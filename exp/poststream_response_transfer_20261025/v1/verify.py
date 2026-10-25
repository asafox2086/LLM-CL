"""Validate both complete GPU arms before the single 200-answer scoring pass."""
from pathlib import Path
import ast,collections,hashlib,json,math,os,re,time
RUN=Path(os.environ['LLMCL_RUN']);ROOT=Path(os.environ['LLMCL_ROOT']);ASSET=Path(os.environ['LLMCL_ASSET'])
DOMAINS=('natural','medical','geography','finance','traffic')
LIMITS={'CE_train':1200,'KD_train':1800,'CE_eval':450,'KD_eval':450}
SMALL={'coefficients':789080096,'bytes':1578160192}
INDEXES=[2,5,13,16,17,18,20,21]
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (RUN/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def keyed(rows):
    out={v['id']:v for v in rows};assert len(rows)==len(out);return out

def main():
    tick=time.monotonic();assert not (RUN/'VERIFY_STARTED.json').exists() and not (RUN/'VERIFIED.json').exists()
    ready=read(RUN/'READY.json');rh=sha(RUN/'READY.json')
    for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
    for n,h in ready['runtime_files'].items():assert sha(ASSET/n)==h,n
    for n,h in ready['prior_files'].items():assert sha(RUN/n)==h,n
    assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
    prior=read(RUN/'prior_VERIFIED.json');old=read(RUN/'prior_COMPLETE.json')
    assert prior['complete'] and prior['source_COMPLETE_sha256']==sha(RUN/'prior_COMPLETE.json')
    assert prior['ready_sha256']==sha(RUN/'prior_READY.json')
    assert sha(RUN/'prior_READY.json')=='c61d88028cf0ada9b14d5e477a956af57e72a3a009c4850b685c95f2b6ad4740'
    for n in ['parent_predictions.json','pruned_predictions.json','parent_memory.json','pruned_memory.json']:
        assert sha(RUN/('prior_'+n))==old['files'][n]
    assert old['selected']==INDEXES and old['metadata']['pruned']['layout']['weights']==SMALL
    port=ASSET/'exp/two_backbone_native_cl_20261017_v2'
    policy=read(port/'execution_policy.json');mp=ASSET/policy['conditions']['pct01']['manifest'];manifest=read(mp)
    sp=ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json';split=read(sp)
    assert sha(mp)==ready['manifest_sha256'] and sha(sp)==ready['split_sha256']
    fit={};gold={}
    for t in manifest['tasks']:
        p=ASSET/t['files']['train']['path'];assert sha(p)==t['files']['train']['sha256']
        values={v['id']:v for v in map(json.loads,p.read_text().splitlines())}
        fi=split['fit_ids'][str(t['task'])];di=split['dev_ids'][str(t['task'])]
        assert set(fi).isdisjoint(di) and set(fi)|set(di)==set(values)
        for rid in fi:assert rid not in fit;fit[rid]=(t,values[rid])
        for rid in di:
            assert rid not in gold;gold[rid]=(t,values[rid])
            for image in values[rid]['images']:assert sha(ASSET/image)==split['image_sha256'][image]
    assert len(fit)==1213 and len(gold)==100 and set(fit).isdisjoint(gold)
    memory=read(RUN/'MEMORY.json');chosen={}
    for d in DOMAINS:
        chosen[d]=sorted((tr for tr in fit.values() if tr[0]['domain']==d),
            key=lambda tr:(hashlib.sha256(('response-memory-20261025:'+tr[1]['id']).encode()).digest(),tr[1]['id']))[:32]
    expected=[dict(id=chosen[d][i][1]['id'],task=chosen[d][i][0]['task'],domain=d) for i in range(32) for d in DOMAINS]
    assert memory['order']==expected and len({x['id'] for x in expected})==160
    for d in DOMAINS:
        assert memory['domain_image_groups'][d]==len({tuple(r['images']) for _,r in chosen[d]})
    all_exit=read(RUN/'ALL_EXIT.json');assert all_exit['complete'] and all_exit['no_quality_consumed']
    activation=read(RUN/'ACTIVATION.json');last=None;complete={}
    for case,budget in LIMITS.items():
        assert not (RUN/(case+'_FAILURE.json')).exists()
        st=read(RUN/(case+'_STARTED.json'));end=read(RUN/(case+'_COMPLETE.json'));ex=read(RUN/(case+'_EXIT.json'))
        ident=st['identity'];assert ident==end['identity'] and ident['pid']==ident['pgid']==ex['pid'] and ident['uid']==os.getuid()
        assert not (Path('/proc')/str(ident['pid'])).exists() and ex['returncode']==0
        assert st['boot']==activation['boot'] and st['student_GPU']==activation['student_GPU']
        assert st['teacher_GPU']==(activation['teacher_GPU'] if case=='KD_train' else None)
        assert st['ready_sha256']==end['ready_sha256']==rh and 0<end['seconds']<=budget
        assert ex['unix']>=st['unix'] and (last is None or st['unix']>=last);last=ex['unix']
        assert end['TEST_QA']==end['DEV_used_for_optimization']==0
        for n,h in end['files'].items():assert sha(RUN/n)==h,n
        complete[case]=end
    assert len({v['identity']['pid'] for v in complete.values()})==4
    assert all_exit['unix']>=last and sum(v['seconds'] for v in complete.values())<=3900
    initial={arm:read(RUN/(arm+'_INITIALIZATION.json')) for arm in ['CE','KD']}
    assert initial['CE']['adapter_initial_digest']==initial['KD']['adapter_initial_digest']
    for k in ['frozen_digest','buffer_digest','math']:assert initial['CE'][k]==initial['KD'][k],k
    prior_inputs=keyed(read(RUN/'prior_parent_predictions.json'));pruned_inputs=keyed(read(RUN/'prior_pruned_predictions.json'))
    assert set(prior_inputs)==set(pruned_inputs)==set(gold)
    old_quality={label:prior['quality'][label] for label in ['parent','pruned']}
    for label,inputs in [('parent',prior_inputs),('pruned',pruned_inputs)]:
        q=keyed(old_quality[label]['per_id']);assert set(q)==set(gold)
        for rid in gold:
            assert q[rid]['input_sha256']==inputs[rid]['input_sha256']==prior_inputs[rid]['input_sha256']
    predictions={};memory_values={}
    for arm in ['CE','KD']:
        train=complete[arm+'_train'];ev=complete[arm+'_eval'];cp=train['checkpoint'];init=initial[arm]
        assert train['initial_static_digest']==old['metadata']['pruned']['digest']==init['initial_static']['digest']
        assert train['initial_adapter_digest']==init['adapter_initial_digest']
        assert train['final_adapter_digest']!=train['initial_adapter_digest']
        assert train['frozen_unchanged'] and train['buffers_unchanged']
        assert init['adapter']['modules']==144 and init['adapter']['parameters']==288 and init['adapter']['coordinates']==3293184
        assert train['FIT_model_requests']==train['backwards']==320 and train['optimizer_updates']==20
        assert train['teacher_decoder_requests']==(160 if arm=='KD' else 0)
        assert train['teacher_head_requests']==(320 if arm=='KD' else 0)
        assert train['new_generations']==0 and ev['new_generations']==105
        tr=train['training']
        assert tr['backwards']==tr['student_decoder_requests']==tr['student_head_requests']==320 and tr['optimizer_updates']==20
        assert tr['adapter_coordinates']==3293184 and tr['trainable_parameters']==288
        assert tr['optimizer']['all_state_tensors_cuda'] and tr['optimizer']['capturable']
        assert tr['optimizer']['weight_decay']==.01 and tr['optimizer']['lr']==.0001 and tr['optimizer']['eps']==1e-8
        assert all(ev[k]==0 for k in ['FIT_model_requests','backwards','optimizer_updates','teacher_decoder_requests','teacher_head_requests'])
        assert cp==ev['checkpoint'] and cp['layout']['weights']==SMALL and cp['layout']['trainable']==0
        assert cp['indexes']==INDEXES and cp['embedding_head_tied'] and cp['bridge_aliases']
        assert (RUN/(arm+'_complete.pt')).stat().st_size==cp['bytes']
        assert ev['initial']['digest']==ev['invariants']['digest_before']==ev['invariants']['digest_after']==cp['digest']
        assert ev['invariants']['complete_state_and_buffers_unchanged'] and ev['invariants']['all_parameters_frozen_no_grad']
        assert ev['initial']['all_buffers_digest']==cp['all_buffers_digest']
        fold=train['fold'];assert len(fold)==144
        assert all(v['merge_device'].startswith('cuda') and v['merge_dtype']=='torch.float32' and v['dtype']=='torch.float16'
            and v['matmul_allow_tf32'] is False and v['autocast_enabled'] is False and v['rank']==8 and v['scale']==4 for v in fold)
        qa=[json.loads(v) for v in (RUN/(arm+'_QA.jsonl')).read_bytes().splitlines()]
        updates=[json.loads(v) for v in (RUN/(arm+'_updates.jsonl')).read_bytes().splitlines()]
        assert len(qa)==320 and len(updates)==20
        assert [v['id'] for v in qa]==[v['id'] for v in expected]*2
        assert qa==tr['qa_records'] and updates==tr['update_records']
        assert [v['step'] for v in updates]==list(range(1,21))
        assert all(v['gradient_parameters']==288 and v['gradient_bytes']==13172736 and finite(v['gradient_l2_preclip'])
            and finite(v['gradient_l2_postclip']) and 0<=v['gradient_l2_postclip']<=1.0001
            and all(finite(lr) and 0<lr<=.0001 for lr in v['learning_rate_used']) for v in updates)
        assert all(finite(v['loss']) and finite(v['CE']) and v['backwards']==1 and v['student_decoder_requests']==1
            and (v['KL'] is None if arm=='CE' else finite(v['KL'])) for v in qa)
        assert [v['id'] for v in expected]*2==tr['order_ids']
        values=keyed(read(RUN/(arm+'_predictions.json')));assert set(values)==set(gold)
        for rid,value in values.items():
            t,row=gold[rid];reference=prior_inputs[rid]
            assert value['task']==t['task'] and value['domain']==t['domain']
            assert value['references']==row['answers']==reference['references']
            assert value['generation_cap']==t['max_new_tokens']==reference['generation_cap']
            assert value['prepared_sha256']==reference['input_sha256']
            assert isinstance(value['prediction'],str) and len(value['output_tokens'])<=value['generation_cap']
            assert all(isinstance(x,int) and not isinstance(x,bool) and x>=0 for x in value['output_tokens'])
        predictions[arm]=values
        windows=read(RUN/(arm+'_memory.json'));old_windows=read(RUN/'prior_parent_memory.json');assert len(windows)==len(old_windows)==5
        for a,b in zip(windows,old_windows):
            for k in ['id','domain','input_sha256','prompt_tokens','new_tokens']:assert a[k]==b[k],k
            assert a['new_tokens']==32 and 0<a['baseline_allocated']<=a['peak_allocated']<=a['peak_reserved']
        memory_values[arm]=windows
    cache=read(RUN/'KD_TEACHER.json');records=[json.loads(v) for v in (RUN/'KD_cache.jsonl').read_bytes().splitlines()]
    assert records==cache['cache']['records'] and len(records)==160
    assert [v['id'] for v in records]==[v['id'] for v in expected]
    assert cache['head_identical_to_student'] and cache['head_frozen']
    assert cache['parent']['digest']==old['metadata']['parent']['digest']
    assert cache['cache']['teacher_decoder_requests']==160 and cache['cache']['teacher_head_requests']==0
    assert cache['cache']['hidden_bytes']==sum(v['hidden_bytes'] for v in records)<=147087360
    assert all(v['hidden_dtype']=='torch.float16' and v['hidden_shape']==[1,v['target_tokens'],896]
        and v['hidden_device']==cache['head_device'] and v['post_final_norm_shifted_prediction_positions'] and v['EOS_included'] for v in records)
    nodes=[n for n in ast.parse((port/'runtime_utils.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['tokens','score']]
    assert len(nodes)==2
    tree=ast.Module(body=nodes,type_ignores=[]);scorer_sha=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    assert scorer_sha==prior['scorer_ast_sha256']
    scope=dict(re=re,Counter=collections.Counter);exec(compile(tree,'<original token F1>','exec'),scope)
    write('VERIFY_STARTED.json',dict(unix=time.time(),all_four_phases_exited=True,new_scores=200,scorer_ast_sha256=scorer_sha))
    quality=dict(old_quality)
    for arm,values in predictions.items():
        perid=[dict(id=rid,domain=v['domain'],F1=scope['score'](v['prediction'],v['references'])['token_f1']) for rid,v in sorted(values.items())]
        assert all(finite(v['F1']) and 0<=v['F1']<=100 for v in perid)
        domains={d:dict(F1=sum(v['F1'] for v in perid if v['domain']==d)/20,QA=20) for d in DOMAINS}
        quality[arm]=dict(domains=domains,macro=sum(v['F1'] for v in domains.values())/5,per_id=perid)
    passed={}
    for arm in ['CE','KD']:
        floors=all(quality[arm]['domains'][d]['F1']>=quality['parent']['domains'][d]['F1'] for d in DOMAINS)
        peaks=all(a['peak_allocated']<b['peak_allocated'] for a,b in zip(memory_values[arm],read(RUN/'prior_parent_memory.json')))
        package=complete[arm+'_train']['checkpoint']['bytes']<old['parent']['bytes']
        passed[arm]=dict(all_parent_domain_floors=floors,all5_GPU_peaks_lower=peaks,complete_package_smaller=package,passed=floors and peaks and package)
    chosen='CE' if passed['CE']['passed'] else ('KD' if passed['KD']['passed'] and quality['KD']['macro']>quality['CE']['macro'] else None)
    result=dict(complete=True,ready_sha256=rh,source_complete_sha256={c:sha(RUN/(c+'_COMPLETE.json')) for c in LIMITS},
        quality=quality,criteria=passed,decision='retain_'+chosen+'_component_for_independent_framework_review' if chosen else 'close_fixed_deleted8_memory160_updates20_pair',
        retained_component=chosen,KD_minus_CE={d:quality['KD']['domains'][d]['F1']-quality['CE']['domains'][d]['F1'] for d in DOMAINS},
        KD_minus_CE_macro=quality['KD']['macro']-quality['CE']['macro'],memory=dict(parent=read(RUN/'prior_parent_memory.json'),**memory_values),
        model_seconds={c:v['seconds'] for c,v in complete.items()},checkpoints={a:complete[a+'_train']['checkpoint'] for a in ['CE','KD']},
        training={a:complete[a+'_train']['training'] for a in ['CE','KD']},phase_RSS_highwater_bytes={c:v['process_RSS_highwater_bytes'] for c,v in complete.items()},
        phase_GPU={c:v['phase_GPU'] for c,v in complete.items()},teacher_cache_bytes=cache['cache']['hidden_bytes'],
        new_scores=200,new_generations=210,student_backwards=640,student_optimizer_updates=40,teacher_decoder_requests=160,teacher_head_requests=320,
        prior_scores_reused_without_rescoring=True,scorer_ast_sha256=scorer_sha,TEST_QA=0,NLL_requests=0,
        no_online_CL_unique_mechanism_significance_or_framework_success_claim=True,verification_seconds=time.monotonic()-tick)
    write('VERIFIED.json',result)
    lines=['# 固定删图后的回答监督对照','','| 域 | 父 | 无恢复 | CE | (CE+KL)/2 | KD−CE pp |','|---|---:|---:|---:|---:|---:|']
    for d in DOMAINS:
        lines.append('| '+d+' | '+' | '.join(f"{quality[a]['domains'][d]['F1']:.8f}" for a in ['parent','pruned','CE','KD'])+f" | {result['KD_minus_CE'][d]:+.8f} |")
    lines.append('| 宏 | '+' | '.join(f"{quality[a]['macro']:.8f}" for a in ['parent','pruned','CE','KD'])+f" | {result['KD_minus_CE_macro']:+.8f} |")
    lines.extend(['','Decision: `'+result['decision']+'`.','两臂为同一受损函数增加 post-stream 恢复；非同 87 步在线紧凑 CL，未证明唯一机制或总目标完成。','单 seed / 每域20 / 已探索 DEV；点 F1 不是显著性或等效性证明。'])
    with (RUN/'FINDINGS.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(verified_sha256=sha(RUN/'VERIFIED.json'),macro={a:quality[a]['macro'] for a in quality},decision=result['decision'],model_seconds=sum(result['model_seconds'].values()))))

if __name__=='__main__':main()
