"""Exploratory frozen-base paired probes; never part of official CL scores."""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from run import input_batch, score

ROOT=Path('/data1/liyapeng/program/LLMCL')
EXP='ten_task_cl_20261014_geo_fin'
OUT=ROOT/'summary'/EXP/'failure_diagnosis_20261015'


def main():
    torch.set_num_threads(8)
    torch.manual_seed(42)
    # Physical GPU3 only; no fallback onto reserved GPU0 or a busy GPU.
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    state=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    free,util=map(int,state.strip().split(','))
    assert free>=19000 and util<10,(free,util)
    torch.cuda.set_per_process_memory_fraction(.65,0)
    OUT.mkdir(parents=True,exist_ok=True)
    mp=ROOT/'data/TenTaskCL'/EXP/'manifest.json';sha=hashlib.sha256(mp.read_bytes()).hexdigest()
    m=json.loads(mp.read_text());tasks={x['task']:x for x in m['tasks']}
    config={'exploratory_only':True,'model':'Qwen2-VL-2B-Instruct','frozen':True,'seed':42,
            'gpu':3,'manifest_sha256':sha,'save_checkpoints':False,
            'arms':{'medical_original':'Exact original instruction/question/image',
                    'medical_short_image':'Same question/image; request a word, phrase or number',
                    'medical_short_no_image':'Same short-answer prompt, remove image only',
                    'finance_original':'Exact original FinQA instruction/question/context',
                    'finance_oracle_program':'Add original gold computation program (privileged oracle, no final answer); same numeric output instruction'},
            'caveat':'Oracle program leaks task-specific solution structure; diagnostic positive control, never a benchmark result.'}
    (OUT/'probe_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    model=Qwen2VLForConditionalGeneration.from_pretrained(ROOT/'model/Qwen2-VL-2B-Instruct',torch_dtype=torch.float16,
                                                        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    model.requires_grad_(False)
    processor=AutoProcessor.from_pretrained(ROOT/'model/Qwen2-VL-2B-Instruct',local_files_only=True)
    records=[]
    with (OUT/'probe_predictions.jsonl').open('w',encoding='utf-8') as stream:
        for task_id in [4,7]:
            task=tasks[task_id]
            rows=[json.loads(x) for x in (ROOT/task['files']['test']['path']).read_text().splitlines()]
            assert len(rows)==50
            for index,row in enumerate(rows):
                arms=['medical_original','medical_short_image','medical_short_no_image'] if task_id==4 else ['finance_original','finance_oracle_program']
                for arm in arms:
                    x=copy.deepcopy(row);t=copy.deepcopy(task)
                    if arm.startswith('medical_short'):
                        t['instruction']='Answer the question with only the short answer: a word, phrase, or number. Do not repeat the question. Do not add explanations.'
                        if arm.endswith('no_image'):x['images']=[]
                    if arm=='finance_oracle_program':
                        program=row.get('metadata',{}).get('program_reference')
                        assert program
                        t['instruction']+='\nFor this diagnostic only, use the supplied reference computation program. A #k refers to the result of step k, numbered from zero. Execute the operations and follow the original answer format.\nReference computation program: '+program
                    batch,details=input_batch(x,t,processor,'cuda')
                    cap=64 if task_id==4 else 128
                    start=time.monotonic()
                    with torch.inference_mode():
                        output=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=cap,use_cache=True,
                                              pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
                    response=output[0,batch['input_ids'].shape[1]:]
                    pred=processor.tokenizer.decode(response,skip_special_tokens=True)
                    rec={'task':task_id,'id':row['id'],'arm':arm,'prediction':pred,'references':row['answers'],
                         'scores':score(pred,row['answers']),'output_tokens':len(response),'generation_cap':cap,
                         'seconds':time.monotonic()-start,**details}
                    stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();records.append(rec)
                if (index+1)%5==0:print(json.dumps({'task':task_id,'finished_qa':index+1,'total':50}),flush=True)
    result={}
    for arm in config['arms']:
        rows=[x for x in records if x['arm']==arm]
        assert len(rows)==50
        result[arm]={'n':50,**{s:sum(x['scores'][s] for x in rows)/50 for s in ['exact_match','token_f1','rougeL']},
                     'cap_hits':sum(x['output_tokens']>=x['generation_cap'] for x in rows)}
    (OUT/'probe_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PROBES_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
