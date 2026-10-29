"""One metadata/readout verifier, without model load or rescoring."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def durable(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def close(a,b):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-9*max(1.,abs(a),abs(b))

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True,type=Path);p.add_argument('--run',required=True,type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    b=read(a.binding);run=a.run
    assert b['schema']=='pas-static-function-probe-v1' and sha(__file__)==b['verify_sha256']
    assert sha(b['protocol_path'])==b['protocol_sha256'] and sha(b['selection_path'])==b['selection_sha256']
    selection=read(b['selection_path']);assert selection['schema']=='pas-static-selection-v1'
    assert not (run/'VERIFICATION_STARTED.json').exists()
    if not a.execute:print(json.dumps(dict(model_loaded=False,verification_started=False)));return
    durable(run/'VERIFICATION_STARTED.json',dict(binding_sha256=sha(a.binding),verifier_sha256=sha(__file__)))
    try:
        assert not (run/'FAILURE.json').exists()
        start=read(run/'STARTED.json');complete=read(run/'COMPLETE.json');result=read(run/'RESULT.json');cal=read(run/'CALIBRATION.json');ex=read(run/'EXIT.json')
        assert complete['complete'] and ex['returncode']==0 and ex['identity']==start['identity']==complete['identity']
        identity=start['identity'];proc=Path('/proc')/str(identity['pid'])/'stat'
        assert not proc.exists() or proc.read_text().rsplit(') ',1)[1].split()[19]!=str(identity['startticks'])
        assert start['binding_sha256']==complete['binding_sha256']==sha(a.binding)
        assert start['worker_sha256']==complete['worker_sha256']==b['worker_sha256']
        assert complete['RESULT_sha256']==sha(run/'RESULT.json') and complete['seconds']<=2400
        assert result['calibration_sha256']==sha(run/'CALIBRATION.json')
        assert result['checkpoint_sha256']==b['checkpoint_sha256']=='583ab5d02e187fbd6a54bd4a05a019e33101deb8af2d9533795253140721441a'
        assert result['counts']==dict(calibration_forward=80,readout_CE=300,generations=300,scored=300)
        assert result['protocol_sha256']==b['protocol_sha256'] and result['selection_sha256']==b['selection_sha256']
        assert result['TEST_QA']==result['backwards']==result['optimizer_updates']==result['new_checkpoints']==0
        assert result['calibration_outside_readout'] and result['readout_seen_in_original_training'] and result['not_new_acquisition_evidence'] and result['not_independent_deployment_reload']
        assert cal['readout_loss_not_yet_read'] and cal['weights_and_buffers_unchanged'] and cal['QA']==80 and len(cal['constants'])==168
        assert cal['calibration_ids_sha256']==result['calibration_ids_sha256']==b['calibration_ids_sha256']
        for item in cal['constants'].values():
            assert item['QA']==80 and len(item['c'])==2 and all(math.isfinite(v) and 0<=v<=1 for v in item['c']) and close(sum(item['c']),1.)
            assert item['denominator']>=0 and item['mask']==[1.,1.]
            if sum(v>.5 for v in item['mask'])==2 and item['denominator']>0:assert close(item['c'][0],item['numerator']/item['denominator'])
        arms=['dynamic','constant_unfolded','constant_folded'];assert list(result['arms'])==arms
        records={};expected_ids=None;domains=None
        for arm in arms:
            proof=result['arms'][arm];assert read(run/(arm+'_COMPLETE.json'))==proof and sha(run/(arm+'.jsonl'))==proof['records_sha256']
            rows=[json.loads(s) for s in (run/(arm+'.jsonl')).read_text().splitlines()];assert len(rows)==100 and len({r['id'] for r in rows})==100
            ids=[(r['task'],r['id']) for r in rows]
            digest=hashlib.sha256(''.join(str(t)+':'+i+'\n' for t,i in ids).encode()).hexdigest()
            assert digest==b['readout_ids_sha256']==result['readout_ids_sha256']
            assert [dict(task=r['task'],domain=r['domain'],id_sha256=hashlib.sha256(r['id'].encode()).hexdigest()) for r in rows]==selection['readout']
            if expected_ids is None:expected_ids=ids;domains=list(proof['domains'])
            else:assert expected_ids==ids and domains==list(proof['domains'])
            assert len(domains)==5 and proof['weights_and_buffers_unchanged_during_readout']
            for domain in domains:
                selected=[r for r in rows if r['domain']==domain];assert len(selected)==20
                assert all(math.isfinite(r['AnswerCE']) and r['target_tokens']>0 and 0<=r['scores']['token_f1']<=100 for r in selected)
                f1=sum(r['scores']['token_f1'] for r in selected)/20;ce=sum(r['AnswerCE'] for r in selected)/20
                assert close(f1,proof['domains'][domain]['AnswerF1']) and close(ce,proof['domains'][domain]['AnswerCE'])
            assert close(proof['macro_F1'],sum(v['AnswerF1'] for v in proof['domains'].values())/5)
            records[arm]=rows
        assert result['arms']['dynamic']['layout']==result['arms']['constant_unfolded']['layout']
        assert result['arms']['dynamic']['state_digest']==result['arms']['constant_unfolded']['state_digest']
        assert result['arms']['constant_folded']['layout']['weights']['coefficients']==893675552
        assert result['arms']['constant_folded']['old_PAS_modules_released'] and result['arms']['constant_folded']['training_learner_removed']
        differences={}
        for before,after in zip(arms[:-1],arms[1:]):
            delta=[]
            for x,y in zip(records[before],records[after]):
                assert x['id']==y['id'] and x['references']==y['references'] and x['generation_cap']==y['generation_cap'] and x['target_tokens']==y['target_tokens']
                delta.append(dict(id=x['id'],domain=x['domain'],F1_delta=y['scores']['token_f1']-x['scores']['token_f1'],CE_delta=y['AnswerCE']-x['AnswerCE']))
            differences[before+'__to__'+after]=dict(all_100_pairs=delta,higher=sum(v['F1_delta']>0 for v in delta),lower=sum(v['F1_delta']<0 for v in delta),equal=sum(v['F1_delta']==0 for v in delta))
        durable(run/'VERIFIED.json',dict(binding_sha256=sha(a.binding),COMPLETE_sha256=sha(run/'COMPLETE.json'),RESULT_sha256=sha(run/'RESULT.json'),verifier_sha256=sha(__file__),differences=differences,new_model_loads=0,new_scores=0,scope='calibration-outside 100 previously trained exploration QA; no generalization/significance/new-acquisition/deployment-reload claim'))
    except BaseException:
        durable(run/'VERIFICATION_FAILURE.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
