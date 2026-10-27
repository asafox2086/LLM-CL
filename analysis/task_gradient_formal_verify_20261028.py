"""One TaskGradReuse formal verifier using the unchanged original v2 engine.

CLI --root ABS_SCIENCE_ROOT --binding ABS_BINDING_JSON. No model loading,
generation or training. The only scoring call is the original engine.verify;
its 74 cells/80316 existing answers/full S19 checkpoint SHA checks are retained.
Reports live only in this experiment's own summary directory, never old indexes.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import time
import traceback

METHOD = 'TaskGradReuse'
FRAME = 'exp/task_gradient_formal_20261028'
RUN = FRAME+'/runs/onevision/pct01/TaskGradReuse/training_v1'
PREFLIGHT = FRAME+'/runs/onevision/pct01/TaskGradReuse/preflight_v1'
MANIFEST = 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
ENGINE_SHA = 'd7aac920f738c6ee8fde9a636560f2d2d6ee7ea1ba9de9e030e6548b4b4cad0a'
METRIC_SHA = '16c59998413d12b820d6f57269f9e1f45ac78e08595d20bf1f4fbc967fd94c4b'
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
MODULES = {f'model.layers.{i}.self_attn.{p}_proj' for i in range(24) for p in ('q','v')}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):h.update(block)
    return h.hexdigest()


def check(value, message):
    if not value:raise ValueError(message)


def finite(value):
    check(isinstance(value,(int,float)) and not isinstance(value,bool)
          and math.isfinite(value), 'nonfinite/non-numeric method record')
    return value


def finite_tree(value):
    if isinstance(value,(int,float)) and not isinstance(value,bool):finite(value)
    elif isinstance(value,dict):
        for item in value.values():finite_tree(item)
    elif isinstance(value,list):
        for item in value:finite_tree(item)


def identity(pid):
    try:
        p = Path('/proc')/str(pid);s = (p/'stat').read_text().rsplit(') ',1)[1].split()
        if s[0] == 'Z':return None
        return dict(pid=int(pid),startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]))
    except (OSError,ValueError,IndexError):return None


def gone(w):
    check(identity(w['pid']) != w, 'recorded research identity still live')


def write(path, value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


class Files:
    def __init__(self):self.records = {}

    def raw(self, path, expected=None):
        path = Path(path);s = path.stat();raw = path.read_bytes();h = hashlib.sha256(raw).hexdigest()
        check((path.stat().st_size,path.stat().st_mtime_ns)==(s.st_size,s.st_mtime_ns), 'file changed during read')
        check(expected is None or h == expected, 'source/data SHA mismatch: '+str(path))
        entry = dict(sha256=h,bytes=len(raw),mtime_ns=s.st_mtime_ns)
        check(str(path) not in self.records or self.records[str(path)] == entry, 'file changed between reads')
        self.records[str(path)] = entry
        return raw

    def json(self,path,expected=None):return json.loads(self.raw(path,expected))

    def unchanged(self):
        for path,record in self.records.items():
            s = Path(path).stat()
            check((s.st_size,s.st_mtime_ns)==(record['bytes'],record['mtime_ns']), 'input changed before commit')


def config_contract(config, sources, preflight):
    check(config['experiment_id']==Path(FRAME).name and config['method']==METHOD
          and config['condition']=='pct01' and config['backbone']=='onevision'
          and config['manifest_sha256']==MANIFEST_SHA and config['seed']==42
          and config['epochs']==1 and config['batch_size']==16 and config['micro_batch_size']==1
          and config['train_fraction']==.01 and config['test_fraction']==1
          and config['preflight_only'] is preflight and config['checkpoint_all_stages']
          and config['execution_source_sha256']==sources, 'native formal configuration')


def terminal(root,binding_path,b,files):
    out = root/'run_v1';report = root/'summary/task_gradient_formal_20261028'
    check(b['schema']=='task-gradient-formal-v1' and b['method']==METHOD, 'binding identity')
    check(b['manifest_sha256']==MANIFEST_SHA and len(b['sources'])==29, 'bound data/29-source identity')
    sources = b['sources'];here = root/FRAME
    check({p.name for p in here.glob('*.py')}==set(sources), 'complete bound Python source inventory')
    for name,h in sources.items():
        check(Path(name).name==name and name.endswith('.py'), 'flat source name')
        files.raw(here/name,h)
    protocol = b['protocol'];check(Path(protocol['path']).is_absolute(), 'absolute protocol path')
    files.raw(protocol['path'],protocol['sha256'])
    files.raw(here/'execution_policy.json',b['execution_policy_sha256'])
    check(not (out/'FORMAL_FAILURE.json').exists(), 'formal model failure must remain closed')
    start = files.json(out/'FORMAL_STARTED.json');end = files.json(out/'FORMAL_COMPLETE.json')
    bh = sha(binding_path)
    for v in (start,end):
        check(v['GPU']==1 and v['source_sha256']==sources and v['protocol']==protocol
              and v['binding_sha256']==bh and v['no_retry'] and v['formal_fresh_W0'], 'formal source/start binding')
    check(start['identity']==end['identity'] and end['active_worker'] is None, 'controller terminal identity')
    gone(end['identity'])
    check((start['planned_stages'],start['TRAIN_QA'],start['TEST_QA'],start['planned_updates'],
           start['calibration_backwards'],start['evaluation_cells'])==(19,1313,21454,93,222,74), 'formal finite work budget')
    check(start['preflight_weights_not_reused_for_training'] and end['new_verification_scores']==0,
          'fresh training and no automatic verification')
    check(Path(end['training_run'])==root/RUN and Path(end['preflight_run'])==root/PREFLIGHT,
          'unique native run paths')
    check([v['phase'] for v in end['phases']]==['preflight','reload','training'], 'three native phases')
    for item in end['phases']:
        phase = item['phase'];gone(item['identity'])
        check(item['returncode']==0 and not item['timeout'] and item['identity_absent'], 'real phase exit0')
        outer = files.json(out/('PHASE_'+phase+'_STARTED.json'))
        ex = files.json(out/('PHASE_'+phase+'_EXIT.json'))
        check(ex=={k:v for k,v in item.items() if k not in ('proof_sha256','checkpoint')}, 'phase exit receipt')
        check(outer['identity']==item['identity'] and outer['phase']==phase and outer['GPU']==1
              and outer['native_run']==item['native_run'], 'phase start identity')
        destination = root/(RUN if phase=='training' else PREFLIGHT)
        check(Path(item['native_run'])==destination, 'phase destination')
        config_contract(files.json(destination/'config.json'),sources,phase!='training')
        if phase!='training':
            proof = files.json(destination/('gpu_preflight.json' if phase=='preflight' else 'checkpoint_reload_proof.json'),item['proof_sha256'])
            check(proof['passed'] and proof['execution_source_sha256']==sources, 'native preflight/reload proof')
            if phase=='preflight':check(proof['task_ids']==list(range(1,20)), 'all 19 original preflight tasks')
    if end['last_attempted_worker'] is not None:gone(end['last_attempted_worker'])
    slot = files.json(report/'SLOT.json')
    check(slot['status']=='complete' and slot['released'] is True and slot['no_retry']
          and slot['controller']==end['identity'] and slot['GPU']==1
          and slot['sources']==sources and slot['binding_sha256']==bh
          and slot['science_root']==str(root) and slot['output']==str(out)
          and slot['terminal_path']==str(out/'FORMAL_COMPLETE.json')
          and slot['terminal_sha256']==sha(out/'FORMAL_COMPLETE.json')
          and slot['original_workers_and_controller_absent'] and slot['own_compute_PIDs_absent'], 'original allocator released terminal')
    files.raw(root/'analysis/v2_verified_baseline_report.py',ENGINE_SHA)
    files.raw(root/'analysis/answer_f1.py',METRIC_SHA)
    lock = files.json(root/'summary/primary_metric_20261017/metric_lock.json')
    check(lock['metric_sha256']==METRIC_SHA and not lock['option_mapping']
          and not lock['additional_prefix_or_answer_extraction'], 'original raw F1 contract')
    return end


def module_records(records):
    check(len(records)==48 and {r['module'] for r in records}==MODULES, '48 unique q/v modules')
    finite_tree(records)
    for r in records:
        check(r['scale']==4 and r['merge_dtype']=='torch.float32' and r['output_dtype']=='torch.float16'
              and not r['autocast_enabled'] and not r['matmul_allow_tf32'], 'original GPU fold arithmetic')


def method_records(root,files,manifest):
    run = root/RUN;total_cal = total_updates = 0;diagnostics = []
    for task in manifest['tasks']:
        stage = task['task'];entry = task['files']['train'];rel = PurePosixPath(entry['path'])
        check(not rel.is_absolute() and '..' not in rel.parts, 'TRAIN relative source')
        # Only IDs are used. Calibration is file-order prefix, not the native
        # minibatch SHA sort. No question, answer or TEST row enters this check.
        ids = [json.loads(line)['id'] for line in files.raw(root/entry['path'],entry['sha256']).decode('utf8').splitlines()]
        check(len(ids)==entry['count'] and len(set(ids))==len(ids), 'original TRAIN IDs/count')
        order = sorted(ids,key=lambda x:hashlib.sha256(f'42:{stage}:{x}'.encode()).digest())
        check(hashlib.sha256(''.join(x+'\n' for x in order).encode()).hexdigest()==task['training_order_sha256'], 'native main training order')
        n = min(16,len(ids));total_cal += n;total_updates += math.ceil(len(ids)/16)
        value = files.json(run/f'method_state/task_gradient_stage_{stage:02d}.json')
        finite_tree(value)
        check(value['stage']==stage and value['preflight_only'] is False
              and value['main_optimizer_coordinates']==540672 and value['begin_stage_RNG_restored']
              and value['cumulative_delta_rank_not_bounded_by_r8'], 'method-stage identity')
        module_records(value['fold'])
        c = value['calibration']
        check(c['QA']==c['backwards']==n and c['optimizer_updates']==0
              and c['ordered_IDs_sha256']==hashlib.sha256(''.join(x+'\n' for x in ids[:n]).encode()).hexdigest()
              and len(c['answer_losses'])==n and c['QA_equal_weight'] and c['current_task_TRAIN_only']
              and c['parameter_and_all_buffer_state_restored'] and c['modes_masks_dtype_RNG_restored']
              and c['native_feature_cache_restored'] and c['gradient_dtype']=='torch.float32'
              and c['forward_autocast']=='torch.float16' and c['dense_gradient_coordinates']==22020096
              and c['gradient_bytes']==88080384, 'current-task calibration and complete restoration')
        directions = value['direction']
        check(len(directions)==48 and {r['module'] for r in directions}==MODULES, 'complete direction inventory')
        for r in directions:
            check(r['gradient_shape']==([896,896] if r['module'].endswith('q_proj') else [128,896])
                  and r['B_zero'] and r['svd_dtype']=='torch.float32' and r['svd_driver']=='default'
                  and r['full_matrices'] is False and r['hard_orthogonality_or_row_norm_cutoff'] is False
                  and len(r['singular_values_top_r'])==8
                  and math.isclose(r['A_row_norm_target'],1/math.sqrt(3),rel_tol=0,abs_tol=1e-12), 'raw default SVD A/B0 definition')
            check(r['A_row_norm_max_error']>=0 and r['basis_orthogonality_max_error']>=0, 'nonnegative recorded numerical error')
        # Record the actual errors; the former local 1e-4 cutoff is not a gate.
        diagnostics.append(dict(stage=stage,calibration_QA=n,
            ordered_IDs_sha256=c['ordered_IDs_sha256'],calibration_seconds=c['seconds'],
            A_row_norm_max_error=max(r['A_row_norm_max_error'] for r in directions),
            basis_orthogonality_max_error=max(r['basis_orthogonality_max_error'] for r in directions),
            numerical_error_cutoff_applied=False))
    check(total_cal==222 and total_updates==93, 'original total main/calibration work')
    final = files.json(run/'method_state/task_gradient_final_fold.json')
    check(final['stage']==19 and final['deployed'] is True, 'final dense deployment marker')
    module_records(final['modules'])
    return dict(stage_diagnostics=diagnostics,extra_calibration_backwards=total_cal,
        main_optimizer_calls=total_updates,main_training_QA=1313,final_folded_modules=48,
        source_bound_final_dense_replacement_marker=True,norm_orthogonality_threshold_not_applied=True)


def acquisition(root,proof,manifest,files):
    rows = []
    for task in manifest['tasks']:
        t = task['task'];path = root/RUN/f'evaluations/stage_{t:02d}/T{t:02d}_test_predictions.scores.json'
        rel = path.relative_to(root).as_posix();saved = files.json(path,proof['source_sha256'][rel])
        check(saved['count']==task['files']['test']['count'], 'verified acquired QA count')
        f1 = finite(saved['token_f1']);check(0<=f1<=100, 'verified acquired F1 range')
        rows.append(dict(task=t,domain=task['domain'],native_label=task['native_label'],QA=saved['count'],F1=f1))
    domains = {}
    for d in DOMAINS:
        selected = [r for r in rows if r['domain']==d];n = sum(r['QA'] for r in selected)
        domains[d] = dict(QA=n,F1=sum(r['F1']*r['QA'] for r in selected)/n)
    return dict(native19=rows,domains=domains,AP19=sum(r['F1'] for r in rows)/19,
        domain_macro_F1=sum(domains[d]['F1'] for d in DOMAINS)/5,
        source='Existing R[t,t] aggregate metadata already SHA-verified by original engine; no extra score')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--binding',type=Path,required=True)
    a = p.parse_args();root = a.root.resolve();report = root/'summary/task_gradient_formal_20261028'
    check(a.root.is_absolute() and root.is_dir() and a.binding.is_absolute(), 'absolute root/binding')
    names = ('VERIFY_STARTED.json','VERIFY_FAILURE.json','VERIFIED.json','RESULT.json')
    check(report.is_dir() and not any((report/n).exists() for n in names), 'unique verification claim or prior failure exists')
    began = time.monotonic();files = Files();engine_invoked = False
    own_sha = sha(Path(__file__))
    write(report/'VERIFY_STARTED.json',dict(unix=time.time(),pid=os.getpid(),verifier_sha256=own_sha,
        binding_path=str(a.binding),binding_sha256=sha(a.binding),exclusive_no_retry=True,
        engine_invoked=False,claim_only_not_integrity_success=True,planned_existing_answer_scores=80316))
    try:
        b = files.json(a.binding);end = terminal(root,a.binding,b,files)
        manifest = files.json(root/MANIFEST,MANIFEST_SHA)
        check([t['task'] for t in manifest['tasks']]==list(range(1,20))
              and manifest['condition']=='pct01' and manifest['seed']==42
              and sum(t['files']['train']['count'] for t in manifest['tasks'])==1313
              and sum(t['files']['test']['count'] for t in manifest['tasks'])==21454, 'original full formal manifest')
        cells = [(s,t) for s in range(20) for t in (range(1,20) if s in (0,19) else (s,s+1))]
        check(len(cells)==74 and sum(manifest['tasks'][t-1]['files']['test']['count'] for _,t in cells)==80316,
              'complete original 74-cell evaluation workload')
        method = method_records(root,files,manifest)
        engine_path = root/'analysis/v2_verified_baseline_report.py'
        spec = importlib.util.spec_from_file_location('task_gradient_unchanged_v2_engine',engine_path)
        engine = importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
        check(engine.SOURCE_SHA==ENGINE_SHA, 'original engine identity after import')
        engine.ROOT=root;engine.FRAME=root/FRAME;engine.METHODS=[METHOD];engine.BACKBONES=['onevision']
        engine_invoked=True
        proof = engine.verify(RUN,'old2080')  # Only original scoring invocation.
        check(proof['rescored_cells']==74 and proof['final_QA']==21454
              and proof['positive_finite_training_stages']==19 and proof['candidate']['method']==METHOD,
              'original complete formal verification')
        check(proof['checkpoint_sha256']==end['phases'][-1]['checkpoint']['sha256'], 'same final CP as model terminal')
        acquired = acquisition(root,proof,manifest,files)
        files.unchanged();check(sha(Path(__file__))==own_sha, 'verifier source changed')
        result = dict(status='FORMAL_QUALITY_VERIFIED_RESOURCES_PENDING',method=METHOD,backbone='onevision',condition='pct01',
            domains=proof['domains'],domain_macro_final=proof['domain_macro_final'],aggregate=proof['aggregate'],
            final_native19=proof['final_native19'],acquired=acquired,method_validation=method,
            original_formal_proof=proof,additional_metadata_sha256=files.records,
            source_sha256=own_sha,engine_sha256=ENGINE_SHA,binding_sha256=sha(a.binding),
            formal_complete_sha256=sha(root/'run_v1/FORMAL_COMPLETE.json'),
            final_complete_weight_and_GPU_resources=None,paper_quality_resource_goal_complete=False,
            original_answer_score_checks=80316,additional_scoring_calls_for_acquisition=0,
            new_models=0,new_generations=0,calibration_uses_current_TRAIN_file_order_prefix=True,
            limits=['One seed/fresh 1% TRAIN and complete TEST, not the 10% setting.',
                    'Acquired domain values combine task-specific times; not a single joint model state.',
                    'Norm/orthogonality errors are recorded without a quality or orthogonality cutoff.',
                    'This independent formal definition does not relabel the former strict local comparison as passed.',
                    'Final full weights, strict deployment reload and GPU costs need the separate resource proof.'])
        write(report/'RESULT.json',result)
        write(report/'VERIFIED.json',dict(status='VERIFIED_ONCE_RESOURCES_PENDING',unix=time.time(),
            verifier_sha256=own_sha,engine_sha256=ENGINE_SHA,binding_sha256=sha(a.binding),
            result_sha256=sha(report/'RESULT.json'),formal_complete_sha256=result['formal_complete_sha256'],
            original_formal_cells=74,existing_answer_score_checks=80316,extra_acquisition_scores=0,
            main_optimizer_calls=93,calibration_backwards=222,resources_pending=True,
            no_old_report_or_index_written=True,new_models=0,new_generations=0,seconds=time.monotonic()-began))
        print(json.dumps(dict(report=str(report/'VERIFIED.json'),resources_pending=True)))
    except BaseException:
        write(report/'VERIFY_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),
            verifier_sha256=own_sha,engine_invoked=engine_invoked,no_automatic_retry=True,
            scoring_checks_completed_unknown_if_engine_failed=engine_invoked,new_models=0,new_generations=0))
        raise


if __name__=='__main__':main()
