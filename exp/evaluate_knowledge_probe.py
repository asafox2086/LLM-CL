"""Re-evaluate saved stages on fixed external knowledge questions; no training."""
import argparse
import fcntl
import hashlib
import io
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
import torch
from transformers import Qwen2VLForConditionalGeneration
from common import ROOT,fingerprint,write_json
from cv_data import MODEL,processor
from cv_runtime import batch_tensors,install_sapt
from run_olora import create_method

p=argparse.ArgumentParser();p.add_argument('--method',required=True);args=p.parse_args()
signal.signal(signal.SIGHUP,signal.SIG_IGN)
run=Path(json.loads((ROOT/'exp/CV_result/latest.json').read_text())['run'])
output=run/'knowledge_probe'/args.method;output.mkdir(parents=True,exist_ok=True)
lock=(output/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=time.monotonic()
def status(state,**details):write_json(output/'status.json',{'status':state,'method':args.method,'pid':os.getpid(),**details})
try:
    torch.set_num_threads(4)
    config=json.loads((run/args.method/'config.json').read_text())
    for path in ['code/'+args.method+'.py','exp/cv_runtime.py','exp/cv_data.py','exp/common.py','exp/run_olora.py']:
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==config['implementation'][path],path
    manifest=json.loads((ROOT/'data/knowledge_probe/manifest.json').read_text())
    content=(ROOT/'data/knowledge_probe/probes.jsonl').read_bytes()
    assert hashlib.sha256(content).hexdigest()==manifest['selection_sha256']
    raw=[json.loads(line) for line in content.splitlines()]
    proc=processor();tok=proc.tokenizer
    rows=[]
    for row in raw:
        prompt='Answer this multiple-choice question. Respond only with the letter of the correct answer.\n\n'+row['question']+'\n'
        prompt+='\n'.join(f'{chr(65+i)}. {answer}' for i,answer in enumerate(row['choices']))
        rendered=proc.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        ids=tok.encode(rendered,add_special_tokens=False);assert len(ids)<=1536, len(ids)
        rows.append({**row,'prompt':prompt,'rendered_prompt':rendered,'prompt_ids':ids})
    letters=[tok.encode(letter,add_special_tokens=False) for letter in 'ABCD'];assert all(len(ids)==1 for ids in letters)
    spec={'format':'llmcl_external_knowledge_v1','source_run':str(run),'method':args.method,
        'dataset':manifest,'scoring':'zero-shot next-token logits restricted to letters A/B/C/D; no generation or training',
        'batch_size':8,'prompt_limit':1536,'truncation':'none','selection_counts':{'general':102,'medical':96},
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'config_sha256':fingerprint(config)}
    if (output/'protocol.json').exists():assert json.loads((output/'protocol.json').read_text())==spec
    write_json(output/'protocol.json',spec)
    (output/'source.py').write_bytes(Path(__file__).read_bytes())
    status('running',phase='loading')
    model=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float16,
        device_map={'':0},attn_implementation='sdpa')
    method=create_method(model,config);install_sapt(method)
    token_ids=torch.tensor([ids[0] for ids in letters],device=model.device)
    summary=[]
    for stage in range(10):
        boundary=run/args.method/'checkpoints'/f'stage_{stage:02d}'/'completed.pt'
        state=torch.load(boundary,map_location='cpu',weights_only=False)
        assert state['config_sha256']==fingerprint(config)
        if stage == 0:
            empty = torch.load(io.BytesIO(state['method']), map_location='cpu', weights_only=True)
            assert empty['task_count'] == 0 and method.task_count == 0
            assert all(not weights for weights in empty['layers'].values())
        else:
            method.load(io.BytesIO(state['method']))
        model.eval()
        status('running',phase='evaluation',stage=stage)
        final=output/f'stage_{stage:02d}.jsonl';temp=final.with_suffix('.jsonl.tmp')
        records=[];source=final if final.exists() else temp
        if source.exists():
            for idx,line in enumerate(source.read_text().splitlines()):
                try:record=json.loads(line)
                except json.JSONDecodeError:break
                assert record['instance_id']==rows[idx]['instance_id'] and record['stage']==stage
                assert record['probe_sha256']==manifest['selection_sha256']
                records.append(record)
        if final.exists():assert len(records)==len(rows)
        else:records=records[:len(records)//8*8]
        temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        with temp.open('a') as handle,torch.inference_mode():
            for start in range(len(records),len(rows),8):
                chosen=rows[start:start+8];batch=batch_tensors(chosen,tok,model.device,False)
                if hasattr(method,'prepare_batch'):method.prepare_batch(chosen,tok,batch,False)
                pos,_=model.get_rope_index(batch['input_ids'],attention_mask=batch['attention_mask'])
                hidden=model.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],position_ids=pos,
                    use_cache=False,return_dict=True).last_hidden_state[:,-1]
                logits=model.lm_head(hidden).float()[:,token_ids]
                probabilities=logits.softmax(-1).cpu().tolist()
                predictions=logits.argmax(-1).cpu().tolist()
                for row,prob,prediction in zip(chosen,probabilities,predictions):
                    record={**row,'stage':stage,'method':args.method,'probe_sha256':manifest['selection_sha256'],
                        'prediction':chr(65+prediction),'reference':chr(65+row['answer']),
                        'correct':int(prediction==row['answer']),'choice_probabilities':prob}
                    records.append(record);handle.write(json.dumps(record,ensure_ascii=False)+'\n')
                handle.flush();os.fsync(handle.fileno())
                print(json.dumps({'stage':stage,'method':args.method,'done':len(records),'total':len(rows)}),flush=True)
        temp.replace(final)
        accuracy=lambda records:100*sum(r['correct'] for r in records)/len(records)
        subjects={subject:accuracy([r for r in records if r['subject']==subject]) for subject in manifest['subjects']}
        summary.append({'method':args.method,'stage':stage,'general_accuracy':accuracy([r for r in records if r['group']=='general']),
            'medical_accuracy':accuracy([r for r in records if r['group']=='medical']),
            'macro_accuracy':sum(subjects.values())/57,'subjects':subjects,'count':len(records)})
        write_json(output/'scores.json',summary)
    status('completed',elapsed_seconds=time.monotonic()-started,stage=9)
except BaseException as exc:
    status('failed',error=repr(exc),traceback=traceback.format_exc());raise
