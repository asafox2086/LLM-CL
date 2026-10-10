import sys,json,time,hashlib,gc,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'exp'))
import cv_data as D,cv_runtime as C
from run_olora import create_method
torch.set_num_threads(4)
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def append(name,x):
    with (OUT/name).open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
class CachedQwen(Qwen2VLForConditionalGeneration):
    @property
    def device(self):return self.get_input_embeddings().weight.device

def main():
    sel=json.loads((OUT/'selection.json').read_text());cfg=json.loads((ROOT/'exp/configs/seq_lora_qwen2vl_med.json').read_text())
    if not (OUT/'evaluation_protocol.json').exists():
        write('evaluation_protocol.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),created_before_predictions=True,batch_size=4,model='Qwen2-VL-2B-Instruct',metric='Primary candidate letter argmax, strict raw first-token letter accuracy separately; case insensitive',checkpoint_source='Existing VQA-RAD adapters, no OmniMedVQA tuning',visual_tokens=128))
    proc=D.processor();tok=proc.tokenizer
    model=CachedQwen.from_pretrained(D.MODEL,local_files_only=True,torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa').eval()
    cache=OUT/'features128';cache.mkdir(exist_ok=True)
    allrows=sel['primary']+[r for pair in sel['matched_pairs'] for r in pair]+[r for p in sel['probes'] for rs in p['splits'].values() for r in rs]
    ims={r['image_sha256']:r for r in allrows}
    for i,(sha,r) in enumerate(ims.items()):
        p=cache/(sha+'.pt')
        if not p.exists():
            with Image.open(r['image']) as im:packed=proc.image_processor(images=[im.convert('RGB')],return_tensors='pt')
            with torch.inference_mode():f=model.visual(packed['pixel_values'].cuda().half(),grid_thw=packed['image_grid_thw'].cuda())
            torch.save(dict(features=f.cpu(),grid=packed['image_grid_thw'][0].tolist()),p)
        if i%25==0:
            write('status.json',dict(status='running',phase='features',done=i,total=len(ims)));print('FEATURES',i,len(ims),flush=True)
    model.visual.to('cpu');torch.cuda.empty_cache();C.install_cached_vision(model)
    C.feature_path=lambda r:Path(r['_feature'])
    # Ridge readout uses only frozen visual outputs and custom training labels.
    def feature(rs):return np.stack([torch.load(cache/(r['image_sha256']+'.pt'),weights_only=True)['features'].float().mean(0).numpy() for r in rs]).astype('float64')
    for name in ['probe_predictions.jsonl','probe_scores.jsonl']:(OUT/name).write_text('')
    for p in sel['probes']:
        data={s:(feature(rs),np.array([r['probe_label'] for r in rs])) for s,rs in p['splits'].items()}
        x,y=data['train'];mean=x.mean(0);std=x.std(0).clip(.1);xx=(x-mean)/std
        # Add intercept, select regularization on dev only.
        xx=np.column_stack([xx,np.ones(len(xx))]);dv=np.column_stack([(data['dev'][0]-mean)/std,np.ones(len(data['dev'][0]))]);ts=np.column_stack([(data['test'][0]-mean)/std,np.ones(len(data['test'][0]))])
        classes=len(p['classes']);one=np.eye(classes)[y];gram=xx@xx.T
        best=None
        for lam in [.001,.01,.1,1.,10.,100.]:
            weights=xx.T@np.linalg.solve(gram+lam*np.eye(len(xx)),one)
            score=float((np.argmax(dv@weights,1)==data['dev'][1]).mean())
            if best is None or score>best[0]:best=(score,lam,weights)
        dev,lam,w=best;pred=np.argmax(ts@w,1)
        rng=np.random.default_rng(42);shuffle=y.copy();rng.shuffle(shuffle)
        sw=xx.T@np.linalg.solve(gram+lam*np.eye(len(xx)),np.eye(classes)[shuffle]);sp=np.argmax(ts@sw,1)
        for r,a,b in zip(p['splits']['test'],pred,sp):append('probe_predictions.jsonl',dict(source=p['source'],question_id=r['question_id'],image_sha256=r['image_sha256'],label=r['probe_label'],prediction=int(a),shuffle_prediction=int(b),correct=int(a==r['probe_label']),shuffle_correct=int(b==r['probe_label'])))
        append('probe_scores.jsonl',dict(source=p['source'],classes=p['classes'],train_n=len(x),dev_n=len(dv),test_n=len(ts),lambda_selected=lam,dev_accuracy=dev,test_accuracy=float((pred==data['test'][1]).mean()),shuffle_accuracy=float((sp==data['test'][1]).mean())))
        print('PROBE',p['source'],float((pred==data['test'][1]).mean()),flush=True)
    # Preserve immutable query sets; no answer/label in prompts except options.
    suites={}
    suites['primary']=sel['primary'];suites['matched']=[r for pair in sel['matched_pairs'] for r in pair]
    suites['probe_test']=[]
    for p in sel['probes']:
        for r in p['splits']['test']:
            suites['probe_test'].append(dict(r,question='Which diagnostic category best describes this medical image?',choices=p['classes'],answer_index=r['probe_label'],probe_source=p['source']))
    def encode(r,condition):
        hasimage=condition=='image';featurepath=cache/(r['image_sha256']+'.pt');grid=torch.load(featurepath,weights_only=True)['grid'] if hasimage else None
        prompt='Answer the following multiple-choice question'+(' using the image' if hasimage else '')+'. Reply with only the correct option letter.\n'+r['question']+'\n'+'\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(r['choices']))
        return D.tokenize(dict(r,image=r['image'] if hasimage else None,prompt=prompt,references=[chr(65+r['answer_index'])],training_answer=chr(65+r['answer_index']),_feature=str(featurepath)),proc,grid)
    encoded={(suite,condition):[encode(r,condition) for r in rs] for suite,rs in suites.items() for condition in ['image','no_image']}
    letterids=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABCD'];assert all(len(tok.encode(c,add_special_tokens=False))==1 for c in 'ABCD')
    done=set()
    if (OUT/'predictions.jsonl').exists():
        done={(r['arm'],r['suite'],r['condition'],r['question_id']) for r in [json.loads(x) for x in (OUT/'predictions.jsonl').read_text().splitlines()]}
    for arm,seed in [('base',0),('SFT_r8_42',42),('SFT_r8_43',43),('SFT_r8_44',44)]:
        method=create_method(model,{**cfg,'rank':8,'alpha':32,'target_tokens':4})
        if seed:method.load(ROOT/'analysis/research_grounding_20261008/checkpoints'/f'{arm}.pt')
        model.eval()
        for (suite,condition),rs in encoded.items():
            todo=[r for r in rs if (arm,suite,condition,r['question_id']) not in done]
            tick=time.time()
            for start in range(0,len(todo),4):
                batch=todo[start:start+4];b=C.batch_tensors(batch,tok,model.device,False)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    pos,_=model.get_rope_index(b['input_ids'],b.get('image_grid_thw'),attention_mask=b['attention_mask'])
                    h=model.model(inputs_embeds=C.fused_embeddings(model,b),position_ids=pos,attention_mask=b['attention_mask'],use_cache=False,return_dict=True).last_hidden_state[:,-1]
                    logits=model.lm_head(h).float();raw=logits.argmax(-1).tolist();z=logits[:,letterids].cpu().numpy()
                for r,t,zz in zip(batch,raw,z):
                    valid=zz[:len(r['choices'])];pred=int(valid.argmax());first=tok.decode([t]).strip().strip('.,:;!?').casefold()
                    append('predictions.jsonl',dict(arm=arm,seed=seed,suite=suite,condition=condition,question_id=r['question_id'],image_sha256=r['image_sha256'],dataset=r['dataset'],question_type=r['question_type'],modality=r.get('modality_type','unspecified'),question=r['question'],choices=r['choices'],answer_index=r['answer_index'],candidate_prediction=pred,candidate_correct=int(pred==r['answer_index']),letter_logits=valid.tolist(),raw_first_token=first,strict_first_token_correct=int(first==chr(65+r['answer_index']).casefold()),probe_source=r.get('probe_source')))
                if start%40==0:
                    progress=dict(status='running',phase='evaluation',arm=arm,suite=suite,condition=condition,done=start+len(batch),total=len(todo),seconds=time.time()-tick)
                    write('status.json',progress);print(json.dumps(progress),flush=True)
            print('SUITE_DONE',arm,suite,condition,len(todo),time.time()-tick,flush=True)
        for n,l in method.layers.items():parent,child=n.rsplit('.',1);setattr(model.get_submodule(parent),child,l.base)
        del method;gc.collect();torch.cuda.empty_cache()
    write('status.json',dict(status='completed',time=time.time(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
if __name__=='__main__':
    try:main()
    except BaseException:
        write('status.json',dict(status='failed',traceback=traceback.format_exc()));raise
