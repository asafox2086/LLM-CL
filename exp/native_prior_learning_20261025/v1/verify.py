"""No model execution: validate complete trajectory, then score 200 answers once."""
from pathlib import Path
import ast,collections,hashlib,importlib.util,json,math,os,re,time
RUN=Path(os.environ['LLMCL_RUN']);ROOT=Path(os.environ['LLMCL_ROOT']);ASSET=Path(os.environ['LLMCL_ASSET'])
DOMAINS=('natural','medical','geography','finance','traffic');LIMITS={'initial':600,'train':3600,'native_eval':600,'parent_resource':240}
SMALL={'coefficients':507482304,'bytes':1014964608};FULL={'coefficients':893675552,'bytes':1787351104}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (RUN/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def keyed(rows):
    out={v['id']:v for v in rows};assert len(out)==len(rows);return out
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def main():
    tick=time.monotonic();assert not (RUN/'VERIFY_STARTED.json').exists() and not (RUN/'VERIFIED.json').exists() and not (RUN/'FAILURE.json').exists()
    ready=read(RUN/'READY.json');rh=sha(RUN/'READY.json')
    for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
    # Initial worker already hashed the immutable native weight asset once;
    # here verify source/config metadata without a repeated 1GB weight scan.
    for n,h in ready['runtime_files'].items():
        if not n.endswith('.safetensors'):assert sha(ASSET/n)==h,n
    for n,h in ready['prior_files'].items():assert sha(RUN/n)==h,n
    for n,h in ready['environment'].items():assert sha(n)==h,n
    assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
    st=read(RUN/'STARTED.json');ex=read(RUN/'EXIT.json');whole=read(RUN/'COMPLETE.json');ident=st['identity']
    assert ident==whole['identity'] and ident['pid']==ident['pgid']==ex['pid'] and ident['uid']==os.getuid()
    assert not (Path('/proc')/str(ident['pid'])).exists() and ex['returncode']==0
    assert whole['complete'] and st['ready_sha256']==whole['ready_sha256']==rh and 0<whole['seconds']<=5040
    assert whole['new_generations']==210 and whole['backwards']==whole['FIT_model_requests']==1213 and whole['optimizer_updates']==87 and whole['new_scores']==whole['TEST_QA']==0
    phases={};last=st['unix']
    for name,budget in LIMITS.items():
        start=read(RUN/(name+'_STARTED.json'));end=read(RUN/(name+'_COMPLETE.json'))
        assert start['identity']==end['identity']==ident and start['ready_sha256']==end['ready_sha256']==rh
        assert start['unix']>=last and 0<end['seconds']<=budget;last=start['unix']+end['seconds']
        assert sha(RUN/(name+'_COMPLETE.json'))==whole['phase_complete_sha256'][name]
        for n,h in end['files'].items():assert sha(RUN/n)==h,n
        phases[name]=end
    assert sum(v['seconds'] for v in phases.values())<=5040 and ex['unix']>=last
    initial=read(RUN/'INITIAL_NATIVE.json');adapter=read(RUN/'INITIAL_ADAPTER.json');train=phases['train'];cp=whole['checkpoint']
    for phase,name in [('train','FRESH_TRAIN_BOUNDARY.json'),('native_eval','FRESH_EVAL_BOUNDARY.json')]:
        boundary=read(RUN/name);assert boundary==phases[phase]['fresh_boundary']
        assert boundary['prior_payload_released'] and boundary['tracked_references']>0 and boundary['same_process_CUDA_context']
        assert boundary['remaining_allocated_bytes']>=0
    assert initial['model']['layout']['weights']==cp['layout']['weights']==SMALL and cp==train['checkpoint']
    assert initial['model']['vision_attention_backend']=='eager' and initial['model']['language_attention_backend']=='sdpa'
    assert cp['layout']['trainable']==0 and cp['complete_native_graph'] and cp['no_teacher_optimizer_adapter']
    assert cp['digest']==phases['native_eval']['loaded']['digest']==phases['native_eval']['invariants']['digest']
    for p in ['initial','native_eval','parent_resource']:assert phases[p]['invariants']['complete_state_and_buffers_unchanged']
    assert (RUN/'native_complete.pt').stat().st_size==cp['bytes']
    assert adapter['coordinates']==train['coordinates']==819200 and adapter['parameters']==128 and len(adapter['rank_plan'])==64
    assert train['adapter_before']==adapter['digest'] and train['adapter_after']!=adapter['digest'] and train['frozen_unchanged'] and train['buffers_unchanged']
    assert train['training_plan']==ready['training_plan'] and train['training_ast_sha256']==ready['training_ast_sha256']
    assert len(train['fold'])==64 and all(v['merge_dtype']=='torch.float32' and v['dtype']=='torch.float16' and v['autocast_enabled'] is False and v['matmul_allow_tf32'] is False for v in train['fold'])
    assert initial['processor']['official_image_seq_len']==64 and initial['processor']['helper_max_image_size']==2048 and initial['processor']['original_tile_longest_edge']==512
    port=ASSET/'exp/two_backbone_native_cl_20261017_v2';policy=read(port/'execution_policy.json');mp=ASSET/policy['conditions']['pct01']['manifest'];manifest=read(mp)
    sp=ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json';split=read(sp)
    assert sha(mp)==ready['manifest_sha256'] and sha(sp)==ready['split_sha256']
    fit={};gold={};expected_fit=[]
    for t in manifest['tasks']:
        path=ASSET/t['files']['train']['path'];assert sha(path)==t['files']['train']['sha256']
        values={v['id']:v for v in map(json.loads,path.read_text().splitlines())};fi=split['fit_ids'][str(t['task'])];di=split['dev_ids'][str(t['task'])]
        assert set(fi).isdisjoint(di) and set(fi)|set(di)==set(values)
        fit.update({rid:(t,values[rid]) for rid in fi});gold.update({rid:(t,values[rid]) for rid in di})
        expected_fit.extend(sorted(fi,key=lambda rid:hashlib.sha256(f"42:{t['task']}:{rid}".encode()).digest()))
    assert len(fit)==1213 and len(gold)==100 and set(fit).isdisjoint(gold)
    inputs=read(RUN/'FIT_INPUTS.json');assert len(inputs)==1213 and [v['id'] for v in inputs]==expected_fit
    events=[json.loads(v) for v in (RUN/'training/logs/events.jsonl').read_bytes().splitlines()]
    assert len(events)==87 and len(list(RUN.glob('grad_S*.json')))==87 and len(list(RUN.glob('adapter_S*.pt')))==19
    for p in ready['training_plan']:
        stage=p['stage'];end=read(RUN/f'STAGE_{stage:02}_COMPLETE.json');valid=read(RUN/f'training/validity/stage_{stage:02}.json')
        assert end['updates']==p['updates'] and end['all_optimizer_state_GPU'] and end['frozen_unchanged'] and end['bytes']==3276800
        assert valid['passed'] and valid['actual_parameters_changed'] and valid['updates']==valid['positive_lr_updates']==p['updates']
        assert all(read(RUN/f'grad_S{stage:02}_{i:03}.json')==dict(parameters=128,gradient_bytes=3276800,finite=True) for i in range(1,p['updates']+1))
    assert all(finite(v['loss']) and all(finite(lr) and 0<lr<=.0001 for lr in v['learning_rate_used']) for v in events)
    # Reconstruct the common raw-input contract without any model/processor.
    spec=importlib.util.spec_from_file_location('raw_retrieval',port/'retrieval.py');retrieval=importlib.util.module_from_spec(spec);spec.loader.exec_module(retrieval)
    from PIL import Image
    image_cache={}
    def raw(t,row):
        context=row.get('context','');indexes=[]
        if len(context)>6000:
            chunks=[context[i:i+1000] for i in range(0,len(context),800)];indexes=sorted(retrieval.retrieve(row['question'],chunks,5));context='\n'.join(chunks[i] for i in indexes)
        shapes=[];images=[]
        for name in row['images']:
            if name not in image_cache:
                assert sha(ASSET/name)==split['image_sha256'][name]
                with Image.open(ASSET/name) as image:w,h=image.size
                before=[w,h]
                if w*h>401408:
                    ratio=math.sqrt(401408/(w*h));w,h=max(1,int(w*ratio)),max(1,int(h*ratio))
                image_cache[name]=dict(original=before,common_input=[w,h])
            shapes.append(image_cache[name]);images.append(dict(path=name,sha256=split['image_sha256'][name]))
        contract=dict(images=images,instruction=t['instruction'],question=row['question'],original_context=row.get('context',''),resolved_context=context,retrieved_chunk_indexes=indexes,image_shapes=shapes,common_image_area_cap=401408)
        return hashlib.sha256(json.dumps(contract,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for v in inputs:
        t,row=fit[v['id']];assert v['raw_input_sha256']==raw(t,row) and v['task']==t['task'] and v['phase']=='TRAIN'
    prior=read(RUN/'prior_VERIFIED.json');old=read(RUN/'prior_COMPLETE.json')
    assert prior['complete'] and prior['source_COMPLETE_sha256']==sha(RUN/'prior_COMPLETE.json') and prior['ready_sha256']==sha(RUN/'prior_READY.json')
    assert sha(RUN/'prior_parent_predictions.json')==old['files']['parent_predictions.json']
    oldpred=keyed(read(RUN/'prior_parent_predictions.json'));assert set(oldpred)==set(gold)
    predictions={}
    for label in ['initial','final']:
        values=keyed(read(RUN/(label+'_predictions.json')));assert set(values)==set(gold)
        for rid,v in values.items():
            t,row=gold[rid];assert v['task']==t['task'] and v['domain']==t['domain'] and v['raw_input_sha256']==raw(t,row)
            assert v['references']==row['answers']==oldpred[rid]['references'] and v['generation_cap']==t['max_new_tokens']==oldpred[rid]['generation_cap']
            assert isinstance(v['prediction'],str) and len(v['output_tokens'])<=v['generation_cap'] and all(isinstance(x,int) and x>=0 for x in v['output_tokens'])
        predictions[label]=values
    assert all(predictions['initial'][rid]['prepared_sha256']==predictions['final'][rid]['prepared_sha256'] for rid in gold)
    native_mem=read(RUN/'native_memory.json');parent_mem=read(RUN/'parent_memory.json');assert len(native_mem)==len(parent_mem)==5
    for a,b in zip(native_mem,parent_mem):
        assert a['id']==b['id'] and a['domain']==b['domain'] and a['raw_input_sha256']==b['raw_input_sha256'] and a['new_tokens']==b['new_tokens']==32
        for v in [a,b]:assert 0<v['baseline_allocated']<=v['peak_allocated']<=v['peak_reserved']
    assert phases['parent_resource']['loaded']['digest']==old['metadata']['parent']['digest'] and phases['parent_resource']['loaded']['layout']['weights']==FULL
    nodes=[n for n in ast.parse((port/'runtime_utils.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['tokens','score']]
    tree=ast.Module(body=nodes,type_ignores=[]);scorer_sha=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest();assert scorer_sha==prior['scorer_ast_sha256']
    scope=dict(re=re,Counter=collections.Counter);exec(compile(tree,'<original score>','exec'),scope)
    write('VERIFY_STARTED.json',dict(unix=time.time(),new_scores=200,scorer_ast_sha256=scorer_sha,complete_worker_exited=True))
    quality={'parent':prior['quality']['parent']}
    for label,values in predictions.items():
        perid=[dict(id=rid,domain=v['domain'],F1=scope['score'](v['prediction'],v['references'])['token_f1']) for rid,v in sorted(values.items())]
        assert all(finite(v['F1']) and 0<=v['F1']<=100 for v in perid)
        domains={d:dict(F1=sum(v['F1'] for v in perid if v['domain']==d)/20,QA=20) for d in DOMAINS}
        quality[label]=dict(domains=domains,macro=sum(v['F1'] for v in domains.values())/5,per_id=perid)
    floors=all(quality['final']['domains'][d]['F1']>=quality['parent']['domains'][d]['F1'] for d in DOMAINS)
    peaks=all(a['peak_allocated']<b['peak_allocated'] for a,b in zip(native_mem,parent_mem));package=cp['bytes']<old['parent']['bytes']
    passed=floors and peaks and package
    result=dict(complete=True,ready_sha256=rh,source_COMPLETE_sha256=sha(RUN/'COMPLETE.json'),quality=quality,
        decision='retain_native_Q_as_carrier_and_learning_baseline' if passed else 'close_fixed_native_FP16_Q_recipe',
        all_parent_domain_floors=floors,all5_same_card_GPU_peaks_lower=peaks,complete_package_smaller=package,gate_passed=passed,
        checkpoint=cp,memory=dict(native=native_mem,parent=parent_mem),model_seconds=whole['seconds'],phase_seconds={n:p['seconds'] for n,p in phases.items()},
        training_memory=train['training_memory'],process_RSS_highwater_bytes=whole['process_RSS_highwater_bytes'],
        FIT_model_requests=1213,backwards=1213,optimizer_updates=87,adapter_coordinates=819200,new_scores=200,new_generations=210,TEST_QA=0,extra_diagnostic_NLL=0,
        parent_scores_reused_without_rescoring=True,scorer_ast_sha256=scorer_sha,no_new_method_joint_prior_causality_significance_or_formal_success_claim=True,
        verification_seconds=time.monotonic()-tick)
    write('VERIFIED.json',result)
    lines=['# 原生紧凑先验资格：完整结果','','| 域 | 完整父 | 原生初态 | 原生QV末态 |','|---|---:|---:|---:|']
    for d in DOMAINS:lines.append('| '+d+' | '+' | '.join(f"{quality[a]['domains'][d]['F1']:.8f}" for a in ['parent','initial','final'])+' |')
    lines.extend(['| 宏 | '+' | '.join(f"{quality[a]['macro']:.8f}" for a in ['parent','initial','final'])+' |','',result['decision'],
        '同1%数据/87更新不等于同计算；作者已有规模/先验与跨骨干实现差异不计我们创新。初分未用于训练分流。'])
    with (RUN/'FINDINGS.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(verified_sha256=sha(RUN/'VERIFIED.json'),macro={a:q['macro'] for a,q in quality.items()},decision=result['decision'],model_seconds=whole['seconds'])))
if __name__=='__main__':main()
