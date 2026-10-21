"""Read a newly sealed case only; original scorer/verifier, no model or TEST."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'exp/osft_increment_screen_20261019/v1'
REPORT=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);args=p.parse_args()
    os.nice(10)
    marker=REPORT/(args.case+'_COMPLETE.json')
    assert marker.exists(),'No partial result reading'
    folder=REPORT/'sealed_case_reports_20261022';output=folder/(args.case+'_VERIFIED.json')
    assert not output.exists(),'Do not repeat a completed audit'
    tick=time.monotonic()
    sys.path.insert(0,str(STUDY))
    import dispatch
    from study import training_ast
    assert args.case in dispatch.CASES
    ready=read(REPORT/'READY.json')
    for name,digest in ready['sources'].items():assert sha(ROOT/name)==digest,name
    for name,digest in ready['base_files'].items():assert sha(ROOT/name)==digest,name
    for name,key in [('PROTOCOL.md','protocol_sha256'),('SPLIT.json','split_sha256'),('CPU_GATE.json','CPU_GATE_sha256')]:
        assert sha(REPORT/name)==ready[key],name
    assert training_ast()[1]==ready['train_AST_sha256']
    value=dispatch.verify(args.case,ready)
    assert value and value['complete']
    run=ROOT/value['checkpoint_root']
    events=[json.loads(x) for x in (run/'logs/events.jsonl').read_text().splitlines()]
    assert len(events)==87
    for row in events:
        assert all(math.isfinite(row[k]) for k in ['loss','gradient_l2_preclip','gradient_l2_postclip'])
        assert all(math.isfinite(lr) for lr in row['learning_rate_used'])
        assert any(lr>0 for lr in row['learning_rate_used'])
    rawfiles=sorted((run/'gradients').glob('S*_*.json'))
    assert len(rawfiles)==87
    for file in rawfiles:
        gradients=read(file)['raw'];assert gradients
        assert all(v['finite'] and math.isfinite(v['norm']) for v in gradients.values())
    first={}
    for stage in range(1,20):
        file=run/'gradients'/f'S{stage:02d}_first_raw.pt'
        assert file.exists() and file.stat().st_size>0
        first[str(stage)]={'bytes':file.stat().st_size,'sha256':sha(file)}
    resources=[json.loads(x) for x in (run/'resources.jsonl').read_text().splitlines()]
    before=next(x for x in resources if x['phase']=='training_before')
    after=next(x for x in resources if x['phase']=='training_after')
    assert before['pid']==after['pid']==read(REPORT/(args.case+'_STARTED.json'))['pid']
    train=[x for x in resources if before['unix']<=x['unix']<=after['unix']]
    keys=['VmRSS_bytes','Pss_bytes','cpu_uss_bytes','gpu_allocated_bytes','gpu_reserved_bytes','gpu_process_nvml_bytes']
    def peaks(rows):
        return {key:max(x[key] for x in rows if x.get(key) is not None) if any(x.get(key) is not None for x in rows) else None for key in keys}
    sidecars=[read(run/'checkpoints'/f'stage_{stage:02d}.json') for stage in range(1,20)]
    assert [x['stage'] for x in sidecars]==list(range(1,20))
    final=next(x for x in resources if x['phase']=='run_finished')
    config=read(run/'config.json')
    report={
        'case':args.case,'complete_marker_sha256':sha(marker),'ready_sha256':sha(REPORT/'READY.json'),
        'analysis_sha256':sha(__file__),'original_dispatch_sha256':sha(STUDY/'dispatch.py'),
        'original_independent_verifier_passed':True,'all_100_DEV_ids_all_original_references_and_original_score_AST_verified':True,
        'FIT_QA':1213,'DEV_QA':100,'TEST_QA':0,'native_gate_FIT_exposures':19,
        'positive_finite_updates':len(events),'stage_update_counts':{str(s):sum(x['stage']==s for x in events) for s in range(1,20)},
        'all_19_parameter_change_validities_passed':True,'raw_gradient_stat_files':len(rawfiles),'first_step_raw_gradient_files':first,
        'domains':value['domains'],'domain_macro':value['domain_macro'],'checkpoint':value['checkpoint'],
        'trainable_parameters':config['trainable_parameters'],'numeric_conditions':config['dtype'],
        'costs':{
            'training_interval_wall_seconds':after['unix']-before['unix'],
            'training_interval_scope':'training_before to training_after: includes first 18 checkpoint saves; excludes final stage checkpoint save and DEV',
            'recorded_model_run_wall_seconds':final['unix']-resources[0]['unix'],
            'recorded_model_run_scope':'resource-monitor interval from process_before_model to run_finished, including model, 19 FIT gates, TRAIN, all checkpoint saves and DEV; earlier imports/data checks excluded',
            'sum_update_body_seconds':sum(x['seconds'] for x in events),
            'all_19_checkpoint_serialization_hash_fsync_seconds':sum(x['seconds'] for x in sidecars),
            'all_19_checkpoint_bytes_written':sum(x['bytes'] for x in sidecars),
            'checkpoint_costs_overlap_wall_intervals_do_not_add_twice':True,
            'TRAIN_interval_observed_instantaneous_peaks_bytes':peaks(train),
            'full_recorded_model_RUN_observed_instantaneous_peaks_bytes':peaks(resources),
            'max_recorded_per_stage_TRAIN_CUDA_allocator_peak_bytes':max(x['gpu_peak_allocated_bytes'] for x in events),
            'memory_scope':'sampled RSS/PSS/USS/NVML/reserved peaks are observations; per-stage CUDA allocated peak comes from original reset-and-update logs; neither is deployment cost',
        },
        'training_events_sha256':sha(run/'logs/events.jsonl'),'resource_log_sha256':sha(run/'resources.jsonl'),
        'new_model_forward':0,'new_backward':0,'new_optimizer_updates':0,'GPU_calls':0,
        'not_a_formal_1313_or_TEST_result':True,'not_deployment_reload_verified':True,
        'candidate_selection_must_wait_for_all_four_fixed_conditions':True,
        'verification_seconds':time.monotonic()-tick,
    }
    folder.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'case':args.case,'verified_report':str(output.relative_to(ROOT)),
                      'domain_macro':report['domain_macro'],'domains':report['domains'],
                      'costs':report['costs'],'verification_seconds':report['verification_seconds']}))


if __name__=='__main__':main()
