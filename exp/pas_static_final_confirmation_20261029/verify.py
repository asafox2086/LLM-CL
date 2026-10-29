"""Unique final confirmation verification; no model load or scoring replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback
from export import validate,sha,read,write,FOLD_DIGEST

def gone(who):
    p=Path('/proc')/str(who['pid'])/'stat'
    return not p.exists() or p.read_text().rsplit(') ',1)[1].split()[19]!=str(who['startticks'])

def close(a,b):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-9*max(1.,abs(a),abs(b))

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--export-dir',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    b,root,original,port,config,cal,probe=validate(a.binding);run=a.run
    assert not (run/'VERIFICATION_STARTED.json').exists()
    if not a.execute:print(json.dumps(dict(model_loaded=False,verification_started=False)));return
    write(run/'VERIFICATION_STARTED.json',dict(binding_sha256=sha(a.binding),verifier_sha256=sha(__file__)))
    try:
        assert not (run/'FAILURE.json').exists() and not (a.export_dir/'FAILURE.json').exists()
        ec=read(a.export_dir/'COMPLETE.json');ex=read(a.export_dir/'EXIT.json');es=read(a.export_dir/'STARTED.json');exported=read(a.export_dir/'EXPORT.json')
        c=read(run/'COMPLETE.json');exit_=read(run/'EXIT.json');start=read(run/'STARTED.json');result=read(run/'RESULT.json');resource=read(run/'RESOURCE_COMPLETE.json');windows=read(run/'windows.json')
        for comp,ext,st,limit in [(ec,ex,es,600),(c,exit_,start,86400)]:
            assert comp['complete'] and ext['returncode']==0 and comp['identity']==ext['identity']==st['identity'] and gone(comp['identity'])
            assert comp['binding_sha256']==st['binding_sha256']==sha(a.binding) and comp['seconds']<=limit
        assert ec['EXPORT_sha256']==sha(a.export_dir/'EXPORT.json') and c['EXPORT_COMPLETE_sha256']==sha(a.export_dir/'COMPLETE.json')
        assert c['RESULT_sha256']==sha(run/'RESULT.json') and c['RESOURCE_COMPLETE_sha256']==sha(run/'RESOURCE_COMPLETE.json')
        assert resource['windows_sha256']==sha(run/'windows.json') and len(windows)==5 and resource['GPU_uuid']==b['GPU_uuid']
        assert resource['raw_and_prepared_match_original_PAS'] and resource['state_unchanged'] and resource['NVML_is_observation_not_peak']
        prior=read(b['pas_memory_path'])
        for current,old in zip(windows,prior):
            for key in ['id','domain','raw_input_sha256','prepared_sha256','transport_sha256','input_shape','input_tensor_shapes','new_tokens']:assert current[key]==old[key],key
            assert current['new_tokens']==32 and current['peak_allocated']>=current['baseline_allocated'] and current['peak_reserved']>=current['peak_allocated']
            assert all(v['process_used_MiB']>=0 for v in current['NVML_observation'].values())
        assert resource['resource_counts']['new_generations']==5 and resource['resource_counts']['new_scores']==0
        assert result['checkpoint_sha256']==exported['checkpoint']['sha256']==start['checkpoint_sha256']
        assert (a.export_dir/'model.pt').stat().st_size==result['checkpoint_bytes']==exported['checkpoint']['bytes']
        assert sha(a.export_dir/'model.pt')==result['checkpoint_sha256'], 'Retained complete model payload no longer matches evaluated checkpoint'
        assert exported['checkpoint']['external_base_weights_required'] is False and exported['checkpoint']['complete_graph']
        assert exported['fold_digest']==result['fold_digest']==FOLD_DIGEST and result['reloaded']['digest']==FOLD_DIGEST
        assert result['reloaded']['strict_complete_reload'] and result['reloaded']['self_contained_meta_construction']
        assert result['invariant']['complete_state_and_buffers_unchanged'] and result['invariant']['digest']==FOLD_DIGEST
        assert result['generation_before']==result['generation_after']==exported['generation_before']
        assert resource['layout']['weights']==dict(coefficients=893675552,bytes=1787351104)
        assert result['score_field']=='token_f1' and result['metric']=='existing raw Answer F1'
        assert result['TEST_QA']==result['quality_generations']==result['new_scores']==21454 and result['resource_generations']==5
        assert result['NLL_requests']==result['optimizer_updates']==result['backwards']==0 and result['FWT'] is None and result['BWT'] is None
        actual_files={str(p.relative_to(a.export_dir/'processor')):sha(p) for p in (a.export_dir/'processor').rglob('*') if p.is_file()};assert actual_files==exported['processor_files']
        manifest=read(b['manifest_path']);derived=[];groups={};seen=set()
        for task in manifest['tasks']:
            path=run/'evaluations/stage_19'/('T%02d_test_predictions.jsonl'%task['task']);relative=str(path.relative_to(run));assert sha(path)==result['outputs'][relative]
            source=root/task['files']['test']['path'];assert sha(source)==task['files']['test']['sha256']
            expected=[json.loads(line) for line in source.read_text().splitlines()];n=0;total=0.
            with path.open() as stream:
                for index,line in enumerate(stream):
                    assert index<len(expected), 'Unexpected extra final prediction'
                    row=expected[index]
                    v=json.loads(line);assert v['id']==row['id'] and v['references']==row['answers'] and v['id'] not in seen
                    assert v['generation_cap']==task['max_new_tokens'];seen.add(v['id']);value=v['scores']['token_f1'];assert math.isfinite(value) and 0<=value<=100
                    total+=value;n+=1
                assert not stream.read()
            assert n==len(expected)==task['files']['test']['count']
            value=dict(task=task['task'],domain=task['domain'],count=n,AnswerF1=total/n);derived.append(value)
            agg=groups.setdefault(task['domain'],dict(count=0,total=0.));agg['count']+=n;agg['total']+=total
        assert len(seen)==21454 and len(derived)==19 and len(groups)==5
        for x,y in zip(derived,result['tasks']):assert x['task']==y['task'] and x['domain']==y['domain'] and x['count']==y['count'] and close(x['AnswerF1'],y['AnswerF1'])
        for domain,v in groups.items():assert v['count']==result['domains'][domain]['count'] and close(v['total']/v['count'],result['domains'][domain]['AnswerF1'])
        assert close(sum(v['AnswerF1'] for v in derived)/19,result['AP19'])
        assert close(sum(v['total']/v['count'] for v in groups.values())/5,result['domain_macro_final'])
        write(run/'VERIFIED.json',dict(binding_sha256=sha(a.binding),RESULT_sha256=sha(run/'RESULT.json'),COMPLETE_sha256=sha(run/'COMPLETE.json'),EXPORT_COMPLETE_sha256=sha(a.export_dir/'COMPLETE.json'),checkpoint_sha256=result['checkpoint_sha256'],checkpoint_bytes=result['checkpoint_bytes'],verifier_sha256=sha(__file__),TEST_QA=21454,final_tasks=19,new_model_loads=0,new_scores=0,full_model_preserved=True,strict_reload=True,fold_digest=FOLD_DIGEST,weights=dict(coefficients=893675552,bytes=1787351104),added_deployment_coefficients=0,domain_macro_final=result['domain_macro_final'],AP19=result['AP19']))
    except BaseException:write(run/'VERIFICATION_FAILURE.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
