"""Exercise observer gates with actual completed old controls, not fake GPU results."""
import copy
import json
import tempfile
from pathlib import Path
import torch
import onevision_matched_task_domain_report as observer

ROOT=Path(__file__).resolve().parents[1]


def rejected(call):
    try:call()
    except AssertionError:return True
    return False


def main():
    torch.set_num_threads(2)
    manifest=json.loads((ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json').read_text())
    original={}
    for task in manifest['tasks']:
        p=ROOT/task['files']['test']['path'];assert observer.sha(p)==task['files']['test']['sha256']
        original.update({r['id']:r for r in map(json.loads,p.read_text().splitlines())})
    pointer=json.loads((ROOT/'summary/onevision_mechanisms_20261016/task_domain_attempt.json').read_text())
    run=ROOT/pointer['run'];expected=pointer['spec']['anchor_test_ids'];assert len(expected)==23
    files={};source_hashes={};starts=[]
    for case in sorted((run/'cases').glob('*_complete.json')):
        assert json.loads(case.read_text())['complete']
        name=case.name.removesuffix('_complete.json')
        for stage in [1,3,5]:
            path=run/f'evaluations/{name}_after{stage}/anchor_image.jsonl'
            durable=json.loads((path.parent/'durable_complete.json').read_text())
            assert observer.sha(path)==durable['files'][path.name]
            source_hashes[str(path.relative_to(ROOT))]=observer.sha(path)
            files[name,stage]=observer.checked_rows(path,expected,original)
        starts.append(files[name,1])
    assert len(files)==72
    same=observer.transition(starts[0],starts[0],original)
    assert same['quality_changes']=={'increased':0,'decreased':0,'unchanged':23}
    assert same['delta_Answer_F1']['mean']==0 and same['delta_Answer_F1']['image_cluster_95CI']==[0,0]
    failed_old=sum(rejected(lambda rows=rows:observer.same_initial(starts[0],rows)) for rows in starts[1:])
    assert failed_old>0,'Known numerical same-stage1 confound should be detected'
    ref_name=next(name for name,stage in files if stage==1)
    zero=observer.compare(files[ref_name,1],files[ref_name,5],files[ref_name,1],files[ref_name,5],original)
    assert zero['Answer_F1']['mean']==0 and zero['Answer_F1']['image_cluster_95CI']==[0,0]
    with tempfile.TemporaryDirectory(prefix='llmcl-f1-observer-') as directory:
        path=Path(directory)/'copy.jsonl';rows=copy.deepcopy(starts[0])
        def save():path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        save();assert rejected(lambda:observer.checked_rows(path,list(reversed(expected)),original))
        rows[0]['references']=['INTENTIONALLY WRONG FIXTURE GOLD'];save()
        assert rejected(lambda:observer.checked_rows(path,expected,original))
        rows=copy.deepcopy(starts[0]);rows[0]['scores']['token_f1']+=1;save()
        assert rejected(lambda:observer.checked_rows(path,expected,original))
    case=ref_name;path=run/f'adapters/{case}/stage_01.pt'
    state=observer.checkpoint(path)['adapter'];observer.verify_adapter(state,state)
    changed=copy.deepcopy(state);first=next(iter(changed));changed[first]['B'].view(-1)[0]+=1
    assert rejected(lambda:observer.verify_adapter(state,changed))
    assert not torch.cuda.is_initialized()
    out={'passed':True,'CPU_only':True,'actual_old_prediction_files_verified':72,'old_QA_per_file':23,
         'known_confounded_old_stage1_output_pairs_rejected':failed_old,
         'identical_model_zero_contrast_exact':True,
         'reordered_ids_changed_gold_changed_score_changed_adapter_rejected':True,
         'new_matched_GPU_cases_executed':0,'full_new_case_artifact_ingestion':'pending actual first GPU completion',
         'sources':source_hashes,'observer_sha256':observer.sha(ROOT/'analysis/onevision_matched_task_domain_report.py')}
    (ROOT/'summary/onevision_diagnostics_20261016/matched_task_domain/OBSERVER_CPU_VALIDATION.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='sources'}))


if __name__=='__main__':main()
