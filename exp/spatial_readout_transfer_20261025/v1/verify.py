"""Verify all six exited phases, then score exactly400 new answers once."""
from pathlib import Path
import ast,collections,hashlib,importlib.util,json,math,os,random,re,struct,time
from common import RUN,ASSET,ROOT,LIMITS,read,sha,write
DOMAINS=('natural','medical','geography','finance','traffic');ARMS=('native','permuted')
SMALL={'coefficients':591534976,'bytes':1183069952}
def keyed(rows):
    result={r['id']:r for r in rows};assert len(result)==len(rows);return result
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def main():
    tick=time.monotonic();assert not (RUN/'VERIFY_STARTED.json').exists() and not (RUN/'VERIFIED.json').exists()
    assert not list(RUN.glob('*_FAILURE.json'));ready=read(RUN/'READY.json');rh=sha(RUN/'READY.json');all_exit=read(RUN/'ALL_EXIT.json')
    assert all_exit['complete'] and all_exit['ready_sha256']==rh and all_exit['new_scores']==0 and all_exit['new_generations']==415
    for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
    for n,h in ready['prior_files'].items():assert sha(RUN/n)==h,n
    for n,h in ready['environment'].items():assert sha(n)==h,n
    for n,h in ready['runtime_files'].items():
        if not n.endswith('.safetensors'):assert sha(ASSET/n)==h,n
    phases={};starts={};ends={}
    for phase,budget in LIMITS.items():
        st=read(RUN/(phase+'_STARTED.json'));end=read(RUN/(phase+'_COMPLETE.json'));ex=read(RUN/(phase+'_EXIT.json'))
        ident=st['identity'];assert ident==end['identity'] and ident['pid']==ident['pgid']==ex['pid'] and ident['uid']==os.getuid()
        assert not (Path('/proc')/str(ident['pid'])).exists() and ex['returncode']==0
        assert st['ready_sha256']==end['ready_sha256']==rh and 0<end['seconds']<=budget and ex['unix']>=st['unix']+end['seconds']
        assert st['GPU']==ready['phase_GPUs'][phase] and sha(RUN/(phase+'_COMPLETE.json'))==all_exit['source_COMPLETE_sha256'][phase]
        for n,h in end['files'].items():assert sha(RUN/n)==h,n
        phases[phase]=end;starts[phase]=st['unix'];ends[phase]=ex['unix']
    assert starts['train_native']>=ends['fit'] and starts['train_permuted']>=ends['fit']
    assert starts['eval_native']>=max(ends['train_native'],ends['train_permuted'])
    assert starts['eval_permuted']>=ends['eval_native'] and starts['parent_resource']>=ends['eval_permuted']
    assert len({ready['phase_GPUs'][p] for p in ['eval_native','eval_permuted','parent_resource']})==1
    assert sum(p['seconds'] for p in phases.values())<=10440
    alignment=read(RUN/'ALIGNMENT.json');fit=phases['fit']
    assert fit['new_generations']==fit['FIT_backwards']==fit['optimizer_updates']==fit['teacher_decoder_requests']==0
    assert fit['teacher_vision_requests']==fit['student_vision_requests']==325 and fit['groups']==ready['groups']
    assert alignment['FIT_QA']==293 and alignment['held_QA']==32 and alignment['no_decoder_or_answer_supervision']
    permutation=list(range(12288));random.Random(42).shuffle(permutation)
    assert hashlib.sha256(struct.pack('<'+'I'*12288,*permutation)).hexdigest()==alignment['permutation']['permutation_SHA256']
    assert alignment['permutation']['complete_input_column_bijection'] and alignment['permutation']['singular_spectrum_and_row_norms_preserved_by_permutation']
    records=alignment['fit_records']+alignment['held_records'];assert len(records)==325
    assert [[r['group_id'],r['id']] for r in records]==ready['groups']
    assert all(r['phase']==('fit' if i<293 else 'held') and r['tiles']>0 and r['rows']==64*r['tiles'] for i,r in enumerate(records))
    for arm in ARMS:
        f=alignment['fit'][arm];assert f['FIT_QA']==293 and f['statistics_dtype']=='torch.float64' and f['statistics_device'].startswith('cuda:')
        assert f['compiled_weight_shape']==[896,12288] and f['compiled_bias_shape']==[896] and f['compose_dtype']=='torch.float32' and not f['compose_TF32']
        assert finite(f['ridge']) and f['ridge']>0 and f['compiled_SHA256']==fit['compiled'][arm]['digest']
    port=ASSET/'exp/two_backbone_native_cl_20261017_v2';policy=read(port/'execution_policy.json');mp=ASSET/policy['conditions']['pct01']['manifest'];manifest=read(mp)
    split_path=ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json';split=read(split_path)
    assert sha(mp)==ready['manifest_sha256'] and sha(split_path)==ready['split_sha256']
    gold={};fit_rows={};order=[]
    for task in manifest['tasks']:
        path=ASSET/task['files']['train']['path'];assert sha(path)==task['files']['train']['sha256']
        values={r['id']:r for r in map(json.loads,path.read_text().splitlines())};fi=split['fit_ids'][str(task['task'])];di=split['dev_ids'][str(task['task'])]
        assert set(fi).isdisjoint(di) and set(fi)|set(di)==set(values)
        fit_rows.update({rid:(task,values[rid]) for rid in fi});gold.update({rid:(task,values[rid]) for rid in di})
        order.extend(sorted(fi,key=lambda rid:hashlib.sha256(f"42:{task['task']}:{rid}".encode()).digest()))
    assert len(fit_rows)==1213 and len(gold)==100 and set(fit_rows).isdisjoint(gold)
    assert all(r['id'] in fit_rows and fit_rows[r['id']][0]['task']==1 for r in records)
    spec=importlib.util.spec_from_file_location('raw_retrieval',port/'retrieval.py');retrieval=importlib.util.module_from_spec(spec);spec.loader.exec_module(retrieval)
    from PIL import Image
    image_cache={}
    def raw(task,row):
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
        contract=dict(images=images,instruction=task['instruction'],question=row['question'],original_context=row.get('context',''),resolved_context=context,retrieved_chunk_indexes=indexes,image_shapes=shapes,common_image_area_cap=401408)
        return hashlib.sha256(json.dumps(contract,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    predictions={};inputs={}
    for arm in ARMS:
        tr=phases['train_'+arm];ev=phases['eval_'+arm]
        assert tr['new_generations']==100 and tr['FIT_backwards']==1213 and tr['optimizer_updates']==87 and tr['coordinates']==4399104
        assert tr['training_plan']==ready['training_plan'] and tr['training_ast_sha256']==ready['training_ast_sha256']
        assert tr['frozen_unchanged'] and tr['buffers_unchanged'] and tr['adapter_before']!=tr['adapter_after']
        assert tr['initial_checkpoint']['layout']['weights']==tr['checkpoint']['layout']['weights']==SMALL
        assert tr['initial_reload']['strict_complete_reload'] and tr['initial_invariant']['complete_state_and_buffers_unchanged']
        assert tr['checkpoint']['digest']==ev['reloaded']['digest']==ev['invariant']['digest'] and ev['invariant']['complete_state_and_buffers_unchanged']
        assert (RUN/(arm+'_complete.pt')).stat().st_size==tr['checkpoint']['bytes']
        assert ev['new_generations']==105 and ev['FIT_backwards']==ev['optimizer_updates']==0
        assert len(tr['fold'])==168 and all(f['merge_dtype']=='torch.float32' and f['dtype']=='torch.float16' and not f['autocast_enabled'] and not f['matmul_allow_tf32'] for f in tr['fold'])
        initial=read(RUN/(arm+'_ADAPTER_INITIAL.json'));assert initial['coordinates']==4399104 and initial['parameters']==336 and len(initial['rank_plan'])==168
        for p in ready['training_plan']:
            stage=p['stage'];st=read(RUN/f'{arm}_STAGE_{stage:02}_COMPLETE.json');valid=read(RUN/f'{arm}_training/validity/stage_{stage:02}.json')
            assert st['stage']==stage and st['updates']==p['updates'] and st['all_optimizer_state_GPU'] and st['frozen_unchanged'] and st['bytes']==17596416
            assert st['optimizer_state_bytes']==35194176 and valid['passed'] and valid['actual_parameters_changed'] and valid['updates']==valid['positive_lr_updates']==p['updates']
            for i in range(1,p['updates']+1):assert read(RUN/f'{arm}_grad_S{stage:02}_{i:03}.json')==dict(parameters=336,finite=True,gradient_bytes=17596416)
        events=[json.loads(x) for x in (RUN/(arm+'_training/logs/events.jsonl')).read_bytes().splitlines()];assert len(events)==87
        assert all(finite(e['loss']) and all(finite(lr) and 0<lr<=.0001 for lr in e['learning_rate_used']) for e in events)
        inputs[arm]=read(RUN/(arm+'_FIT_INPUTS.json'));assert [r['id'] for r in inputs[arm]]==order
        for rec in inputs[arm]:
            task,row=fit_rows[rec['id']];assert rec['phase']=='TRAIN' and rec['task']==task['task'] and rec['raw_input_sha256']==raw(task,row)
        for state in ['initial','final']:
            values=keyed(read(RUN/(arm+'_'+state+'_predictions.json')));assert set(values)==set(gold)
            for rid,rec in values.items():
                task,row=gold[rid];assert rec['task']==task['task'] and rec['domain']==task['domain'] and rec['references']==row['answers'] and rec['generation_cap']==task['max_new_tokens']
                assert rec['raw_input_sha256']==raw(task,row) and isinstance(rec['prediction'],str) and len(rec['output_tokens'])<=rec['generation_cap']
            predictions[arm+'_'+state]=values
    a=read(RUN/'native_ADAPTER_INITIAL.json')['initialization'];b=read(RUN/'permuted_ADAPTER_INITIAL.json')['initialization']
    assert len(a)==len(b)==168 and all({k:v for k,v in x.items() if k!='device'}=={k:v for k,v in y.items() if k!='device'} for x,y in zip(a,b))
    assert all(a['transport_sha256']==b['transport_sha256'] and a['raw_input_sha256']==b['raw_input_sha256'] for a,b in zip(inputs['native'],inputs['permuted']))
    for rid in gold:
        transports=[predictions[arm+'_'+state][rid]['transport_sha256'] for arm in ARMS for state in ['initial','final']];assert len(set(transports))==1
    reference=read(RUN/'prior_VERIFIED.json');old=read(RUN/'prior_COMPLETE.json');assert reference['complete'] and reference['source_COMPLETE_sha256']==sha(RUN/'prior_COMPLETE.json') and reference['ready_sha256']==sha(RUN/'prior_READY.json')
    oldpred=keyed(read(RUN/'prior_parent_predictions.json'));assert set(oldpred)==set(gold) and sha(RUN/'prior_parent_predictions.json')==old['files']['parent_predictions.json']
    assert all(oldpred[rid]['references']==row['answers'] for rid,(_,row) in gold.items())
    parent=read(RUN/'parent_memory.json');assert len(parent)==5
    memory={arm:read(RUN/(arm+'_memory.json')) for arm in ARMS}
    for arm in ARMS:
        assert len(memory[arm])==5
        for a,b in zip(memory[arm],parent):
            assert a['id']==b['id'] and a['domain']==b['domain'] and a['raw_input_sha256']==b['raw_input_sha256'] and a['new_tokens']==b['new_tokens']==32
            for p in [a,b]:assert 0<p['baseline_allocated']<=p['peak_allocated']<=p['peak_reserved']
    assert phases['parent_resource']['reference']['digest']==old['metadata']['parent']['digest']
    nodes=[n for n in ast.parse((port/'runtime_utils.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['tokens','score']]
    tree=ast.Module(body=nodes,type_ignores=[]);score_sha=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest();assert score_sha==reference['scorer_ast_sha256']
    scope=dict(re=re,Counter=collections.Counter);exec(compile(tree,'<original Answer F1>','exec'),scope)
    write('VERIFY_STARTED.json',dict(unix=time.time(),new_scores=400,scorer_ast_sha256=score_sha,all_workers_exited=True))
    quality={'parent':reference['quality']['parent']}
    for name,rows in predictions.items():
        perid=[dict(id=rid,domain=rec['domain'],F1=scope['score'](rec['prediction'],rec['references'])['token_f1']) for rid,rec in sorted(rows.items())]
        assert all(finite(r['F1']) and 0<=r['F1']<=100 for r in perid)
        domains={d:dict(F1=sum(r['F1'] for r in perid if r['domain']==d)/20,QA=20) for d in DOMAINS}
        quality[name]=dict(domains=domains,macro=sum(v['F1'] for v in domains.values())/5,per_id=perid)
    gates={}
    for arm in ARMS:
        floor=all(quality[arm+'_final']['domains'][d]['F1']>=quality['parent']['domains'][d]['F1'] for d in DOMAINS)
        peaks=all(a['peak_allocated']<b['peak_allocated'] for a,b in zip(memory[arm],parent));cp=phases['train_'+arm]['checkpoint']
        package=cp['bytes']<old['parent']['bytes'] and cp['layout']['weights']['bytes']<old['metadata']['parent']['layout']['weights']['bytes']
        gates[arm]=dict(all_domain_floors=floor,all5_GPU_peaks_lower=peaks,complete_package_and_weights_smaller=package,passed=floor and peaks and package)
    viable=[a for a in ARMS if gates[a]['passed']];chosen=max(viable,key=lambda a:(quality[a+'_final']['macro'],a=='native')) if viable else None
    result=dict(complete=True,ready_sha256=rh,all_exit_sha256=sha(RUN/'ALL_EXIT.json'),source_COMPLETE_sha256=all_exit['source_COMPLETE_sha256'],
        quality=quality,gates=gates,retained_arm=chosen,decision='retain_'+chosen+'_carrier_for_framework_review' if chosen else 'close_fixed_8grid_readout_pair',
        checkpoints={a:phases['train_'+a]['checkpoint'] for a in ARMS},memory=dict(parent=parent,**memory),alignment=alignment,
        phase_seconds={p:e['seconds'] for p,e in phases.items()},model_phase_seconds=sum(e['seconds'] for e in phases.values()),controller_seconds=all_exit['controller_seconds'],
        FIT_backwards=2426,optimizer_updates=174,new_generations=415,new_scores=400,TEST_QA=0,extra_diagnostic_NLL=0,parent_quality_reused_without_rescoring=True,
        pure_spatial_or_pretraining_causality_or_novel_pixelshuffle_or_formal_success_claim=False,verification_seconds=time.monotonic()-tick)
    write('VERIFIED.json',result)
    print(json.dumps(dict(VERIFIED_sha256=sha(RUN/'VERIFIED.json'),macro={k:v['macro'] for k,v in quality.items()},gates=gates,decision=result['decision'],model_phase_seconds=result['model_phase_seconds'])))

if __name__=='__main__':main()
