"""Consume the existing VERIFIED proof once; no model, QA, score or bootstrap."""
from pathlib import Path
from collections import Counter
import json,hashlib,time,datetime

ROOT=Path('/data1/liyapeng/program/LLMCL')
REPORT=ROOT/'summary/final_weight_reduction_20261023/answer_guided_pilot_v1'
WORK=ROOT/'exp/answer_guided_rank_renewal_20261023/v1/private'
VER='90942989b8518e318f82c66ea3abfeb5b06249cbbe8d256aa88f367d4b329596'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def consume():
    tick=time.monotonic()
    assert not (REPORT/'OUTCOME.json').exists(),'One result consumer, do not repeat'
    assert sha(REPORT/'VERIFIED.json')==VER
    v=read(REPORT/'VERIFIED.json');assert v['complete'] and not v['possible_followup_only']
    out=dict(question='Does the answer-guided fixed-size flow preserve DEV quality at a smaller final complete model?',
        consumed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),verified_sha256=VER,
        ready_sha256=v['ready_sha256'],source_sha256=sha(Path(__file__)),
        final_coefficients=v['final_coefficients'],base_coefficients=v['base_coefficients'],
        reduction_percent=100*(1-v['final_coefficients']/v['base_coefficients']),
        old_best_legal_FIT_only_macro=v['old_best_legal_FIT_only_macro'],
        cases={},new_pair=v['new_pair'],binding={},
        decision='closed_negative_fixed_recipe',no_full_confirmation=True,
        no_retry_or_rank_LR_bank_search=True,no_new_model=True,no_QA_or_tensor_read=True,
        no_score_or_bootstrap_repeated=True,no_novelty_or_goal_achievement_claim=True,
        TEST_QA=0,FWT=None,BWT=None)
    for case in ('Spectral','Answer'):
        cp=REPORT/'cases'/case/'COMPLETE.json';assert sha(cp)==v['cases'][case]['complete_sha256']
        c=read(cp);root=WORK/case;stages=[]
        update_seconds=0;event_path=root/'logs/events.jsonl'
        assert sha(event_path)==c['events_sha256']
        events=[json.loads(s) for s in event_path.read_text().splitlines()]
        # Only consume event timing fields, never recompute any quality score.
        timing_keys=sorted({k for e in events for k in e if 'seconds' in k})
        timing={k:sum(e.get(k,0) for e in events) for k in timing_keys}
        saves=[]
        for stage in range(1,20):
            paths={n:root/n/f'S{stage:02d}.json' for n in ('consolidation','layouts')}
            paths['checkpoint']=root/'checkpoints'/f'stage_{stage:02d}.json'
            for p in paths.values():
                rel=p.relative_to(ROOT).as_posix();assert sha(p)==c['file_sha256'][rel]
                out['binding'][rel]=c['file_sha256'][rel]
            z=read(paths['consolidation']);lay=read(paths['layouts']);meta=read(paths['checkpoint']);saves.append(meta)
            stages.append({k:z[k] for k in ('stage','current_QA','old_QA','current_payload_bytes','old_payload_bytes',
                'gradient_QA','candidate_forward_QA','mask_different','selected','fallback','count_before','count_after',
                'final_trainable','bank_after_QA','bank_after_bytes','seconds','memory_before','memory_after',
                'current_coverage','old_coverage','current_rejected_id_sha256')})
            stages[-1]['compact_parameter_bytes']=lay['parameter_bytes']
            stages[-1]['candidate_NLL']={k:z[k] for k in ('parent_gate_NLL','spectral_NLL','answer_NLL') if k in z}
        proof=v['cases'][case]
        out['cases'][case]=dict(quality=proof['quality'],model_RUN_seconds=proof['model_RUN_seconds'],
            gradient_QA=proof['gradient_QA'],candidate_forward_QA=proof['candidate_forward_QA'],
            different_masks=proof['different_masks'],answer_selected=proof['answer_selected'],fallback_counts=proof['fallback_counts'],
            trainable_coordinates_during_stage=6635520,gradient_bytes=26542080,two_CPU_moment_bytes=53084160,
            compact_parameter_bytes=c['final_layout']['parameter_bytes'],
            non_weight_buffer_bytes=sum(z['bytes'] for z in c['final_layout']['buffers'].values()),
            complete_sha256=sha(cp),checkpoint=c['checkpoint'],
            all_weight_digest_before_DEV=c['all_weight_digest_before_DEV'],
            full_checkpoint_reload=c['full_checkpoint_reload'],
            native_gate=dict(NLL_paths=38,short_generations=38,max_new_tokens=4),FIT_QA=1213,DEV_QA=100,
            valid_stages=19,positive_LR_updates=87,stages=stages,
            event_timing_sums=timing,save_seconds=sum(m['seconds'] for m in saves),
            logical_checkpoint_bytes=sum(m['bytes'] for m in saves),
            memory_summary=c['memory_summary'],
            phase_caution='training_after retains temporary AB before the last consolidation; compact boundaries are consolidation_after_saved/layouts')
    out['aggregate_model_RUN_seconds']=sum(c['model_RUN_seconds'] for c in out['cases'].values())
    out['CPU_verification_seconds']=v['CPU_verification_seconds']
    out['CPU_gate_seconds']=3.056610914
    out['whole_freeze_entry_seconds']=23.365613
    out['cost_caution']='Gate time is inside freeze entry. Save/consolidation/event timing overlap RUN; logical checkpoint bytes are not measured physical IO. Observed memory peaks are process-specific and not a fresh controlled speed or baseline memory experiment.'
    out['interpretation']=[
        'Both complete flows meet the final-weight target but fail the quality target by a large margin.',
        'Answer proposed a distinct mask in all 19 stages; all 19 finite NLL comparisons selected the spectral candidate. This is not an empty-bank or identical-mask fallback.',
        'Same CP0 and all-spectral decisions do not establish equal realized training trajectories. Final weight digests differ; this consumption does not identify why.',
        'Initial compression, restricted 87-update adaptation, finite-precision consolidation, and implementation/optimization factors are not causally separated.',
        'Do not generalize this failure to all low-rank representations or all uses of answer information. Do not call either method successful compression with retained quality.',
        'Explored 100 DEV, one seed, 20 QA/domain, 89 known groups; tasks 7/8/9/12 have no DEV. No equivalence, stability, new holdout, training-seed uncertainty, or final TEST result.'
    ]
    out['consumer_seconds_before_write']=time.monotonic()-tick
    (REPORT/'OUTCOME.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(outcome_sha256=sha(REPORT/'OUTCOME.json'),consumer_seconds=out['consumer_seconds_before_write'],
        reduction_percent=out['reduction_percent'],macros={k:c['quality']['domain_macro'] for k,c in out['cases'].items()},
        aggregate_RUN=out['aggregate_model_RUN_seconds'],cost={k:dict(save_seconds=c['save_seconds'],event_timing=c['event_timing_sums'],logical_checkpoint_bytes=c['logical_checkpoint_bytes']) for k,c in out['cases'].items()})))
if __name__=='__main__':consume()
