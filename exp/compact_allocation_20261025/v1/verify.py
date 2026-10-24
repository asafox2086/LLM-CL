"""Verify the complete joint study, then score only 280 new answers once."""
from pathlib import Path
import ast,collections,hashlib,json,math,os,re,time
RUN=Path(os.environ['LLMCL_RUN']);ROOT=Path(os.environ['LLMCL_ROOT']);ASSET=Path(os.environ['LLMCL_ASSET'])
DOMAINS=('natural','medical','geography','finance','traffic')
PHASES={'anchor':(450,100),'train':(3000,0),'compact_eval':(450,105),'natural1':(180,20),'medical11':(180,20),'geography15':(180,20),'finance16':(180,20)}
ENDS={'natural1':(1,'natural'),'medical11':(11,'medical'),'geography15':(15,'geography'),'finance16':(16,'finance')}
FULL={'coefficients':893675552,'bytes':1787351104};SMALL={'coefficients':581213056,'bytes':1162426112}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(n,v):
    with (RUN/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def keyed(rows,ids):
    out={r['id']:r for r in rows};assert len(out)==len(rows)==len(ids) and set(out)==set(ids);return out
def main():
    tick=time.monotonic();assert not (RUN/'VERIFY_STARTED.json').exists() and not (RUN/'VERIFIED.json').exists()
    ready=read(RUN/'READY.json');rh=sha(RUN/'READY.json');assert ready['sources']['verify.py']==sha(__file__)
    for name,h in ready['sources'].items():assert sha(RUN/name)==h,name
    for name,h in ready['runtime_files'].items():assert sha(ASSET/name)==h,name
    for name,h in ready['prior_files'].items():assert sha(RUN/name)==h,name
    assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
    assert sha(RUN/'fixed_vision_only.pt')==ready['vision_asset']['sha256']
    cf=read(RUN/'prior_cf_VERIFIED.json');old=read(RUN/'prior_cf_READY.json');ci=read(RUN/'prior_cf_INITIALIZATION.json')
    assert sha(RUN/'prior_cf_VERIFIED.json')=='24fe7263efa78dc6a95dfa3a413059dd39229dfce7f788da66d520e7e5d66a54'
    assert sha(RUN/'prior_cf_READY.json')==cf['ready_sha256'] and cf['complete']
    assert ready['runtime_files']==old['runtime_files'] and ready['vision_asset']==old['vision_asset']
    for n in ('adaptation.py','vision_interface.py','inherited_budget_core.py'):assert ready['sources'][n]==old['sources'][n]
    assert sha(RUN/'prior_cf_INITIALIZATION.json')==read(RUN/'prior_cf_train_COMPLETE.json')['files']['INITIALIZATION.json']
    assert sha(RUN/'prior_cf_train_COMPLETE.json')==cf['source_complete_sha256']['train']
    assert sha(RUN/'prior_cf_compact_eval_COMPLETE.json')==cf['source_complete_sha256']['compact_eval']
    assert sha(RUN/'prior_cf_final_predictions.json')==read(RUN/'prior_cf_compact_eval_COMPLETE.json')['files']['final_predictions.json']
    prior_domain=read(RUN/'prior_domain_VERIFIED.json');assert sha(RUN/'prior_domain_VERIFIED.json')=='23065490fa4f96233d838fad5ff9bef73f09cdb69cc5aa83a150fc9772a925aa'
    port=ASSET/'exp/two_backbone_native_cl_20261017_v2';policy=read(port/'execution_policy.json');mp=ASSET/policy['conditions']['pct01']['manifest'];manifest=read(mp)
    assert sha(mp)==ready['manifest_sha256']==old['manifest_sha256']
    sp=ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json';split=read(sp);assert sha(sp)==ready['split_sha256']==old['split_sha256']
    gold={};fit_count=0
    for t in manifest['tasks']:
        path=ASSET/t['files']['train']['path'];assert sha(path)==t['files']['train']['sha256'];rows=[json.loads(s) for s in path.read_bytes().splitlines()]
        values={r['id']:r for r in rows};fit=split['fit_ids'][str(t['task'])];dev=split['dev_ids'][str(t['task'])]
        assert set(fit).isdisjoint(dev) and set(fit)|set(dev)==set(values);fit_count+=len(fit)
        for rid in dev:
            assert rid not in gold;gold[rid]=(t,values[rid])
            for image in values[rid]['images']:assert sha(ASSET/image)==split['image_sha256'][image]
    assert fit_count==1213 and len(gold)==100 and all(sum(t['domain']==d for t,_ in gold.values())==20 for d in DOMAINS)
    parent_inputs=keyed(read(RUN/'prior_parent_predictions.json'),gold);compact_inputs=keyed(read(RUN/'prior_cf_final_predictions.json'),gold)
    all_exit=read(RUN/'ALL_EXIT.json');assert all_exit['complete'] and all_exit['no_quality_consumed']
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();gpu=int(os.environ['LLMCL_GPU']);last=None;complete={}
    for case,(budget,n) in PHASES.items():
        assert not (RUN/(case+'_FAILURE.json')).exists();s=read(RUN/(case+'_STARTED.json'));e=read(RUN/(case+'_COMPLETE.json'));x=read(RUN/(case+'_EXIT.json'))
        ident=s['identity'];assert ident==e['identity'] and ident['pid']==ident['pgid']==x['pid'] and ident['uid']==os.getuid()
        assert not (Path('/proc')/str(ident['pid'])).exists() and x['returncode']==0
        assert s['GPU']==gpu and s['boot']==boot and s['ready_sha256']==e['ready_sha256']==rh
        assert finite(e['seconds']) and 0<e['seconds']<=budget and e['new_generations']==n
        assert x['unix']>=s['unix'] and (last is None or s['unix']>=last);last=x['unix']
        assert e['NLL_requests']==e['TEST_QA']==0 and e.get('teacher_requests',0)==0
        for name,h in e['files'].items():assert sha(RUN/name)==h,name
        complete[case]=e
    assert len({r['identity']['pid'] for r in complete.values()})==7 and all_exit['unix']>=last
    assert sum(e['seconds'] for e in complete.values())<=4620 and sum(e['new_generations'] for e in complete.values())==285
    extraction=read(RUN/'anchor_extract_COMPLETE.json');anchor=complete['anchor'];train=complete['train'];final=complete['compact_eval'];init=read(RUN/'INITIALIZATION.json')
    assert extraction['source_checkpoint']['sha256']=='e05f7780263e313f5422ee6b547515631ad76bccd1752a51873947ab6658866c'
    assert extraction['asset']==ready['anchor_asset']==anchor['adapter_asset']
    assert sha(RUN/extraction['asset']['path'])==extraction['asset']['sha256']
    assert extraction['extract_source_sha256']==ready['sources']['anchor_extract.py']
    assert extraction['CUDA_initialized'] is False and extraction['model_forward_requests']==extraction['new_scores']==0
    assert anchor['fresh_base_non_AB_exact'] and anchor['copied_complete_cumulative_AB'] and anchor['embedding_head_tied']
    assert anchor['adapter_digest']==extraction['adapter_digest'] and anchor['adapter_parameters']==extraction['adapter_parameters']==336
    assert anchor['adapter_coordinates']==extraction['adapter_coordinates']==4399104 and extraction['adapter_bytes']==17596416
    assert anchor['source_checkpoint']==extraction['source_checkpoint'] and anchor['attention_classes']==extraction['attention_classes']
    assert anchor['nonvision_buffers_digest']==extraction['nonvision_buffers_digest'] and anchor['new_scores']==0
    assert anchor['native_config_file_sha256']==old['runtime_files']['model/LLaVA-OneVision-Qwen2-0.5B-OV/config.json']
    assert anchor['non_adapter_state_digest']==extraction['non_adapter_state_digest'] and anchor['all_buffers_digest']==extraction['all_buffers_digest']
    assert anchor['immutable_before_fold_digest']==anchor['immutable_after_fold_digest'] and anchor['digest_before']==anchor['digest_after']
    assert anchor['layout']['weights']==FULL and anchor['layout']['trainable']==0 and anchor['optimizer_updates']==anchor['backwards']==anchor['FIT_model_requests']==0
    expected={f'model.layers.{i}.{p}' for i in range(24) for p in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj')}
    assert set(init['rank_plan'])==set(extraction['rank_plan'])==expected and all(r==8 for r in init['rank_plan'].values())
    assert init['receipt']==dict(adapter_parameters=336,adapter_coordinates=4399104,identities_bytes_and_RNG_preserved=True)
    assert init['zero_B_modules']==168 and init['initial_new_generations']==0 and init['no_S19_language_loaded']
    for k in ('base_language_digest','frozen_digest','vision_digest','buffer_digest'):assert init[k]==ci[k],k
    assert train['training_plan']==ready['training_plan']==old['training_plan'] and train['training_ast_sha256']==ready['training_ast_sha256']==old['training_ast_sha256']
    assert train['FIT_QA']==train['FIT_model_requests']==train['backwards']==1213 and train['updates']==train['optimizer_updates']==87
    assert train['new_generations']==0 and train['frozen_base_unchanged']
    assert train['DEV_used_for_optimization']==0
    assert train['initial_adapter_digest']==init['adapter_initial_digest']
    assert train['layout']==final['layout'] and train['layout']['weights']==SMALL and train['layout']['trainable']==0
    assert train['digest']==final['digest_before']==final['digest_after']==final['digest'] and final['strict_independent_static_reload']
    assert final['vision_digest']==init['vision_digest'] and final['buffer_digest']==init['buffer_digest']
    for case in PHASES:
        if case!='train':assert all(complete[case][k]==0 for k in ('FIT_model_requests','backwards','optimizer_updates'))
    assert 'anchor_predictions.json' in anchor['files'] and {'final_predictions.json','compact_eval_memory.json'}<=set(final['files'])
    assert (RUN/'compact_complete.pt').stat().st_size==train['checkpoint']['bytes'] and sha(RUN/'compact_complete.pt')==train['checkpoint']['sha256']
    for case in ('anchor','train',*ENDS):
        fold=complete[case]['fold'];assert len(fold)==168 and {r['module'] for r in fold}==expected
        assert all(r['rank']==8 and r['scale']==4 and r['merge_device']=='cpu' and r['merge_dtype']=='torch.float32' and r['dtype']=='torch.float16' for r in fold)
    events=[json.loads(s) for s in (RUN/'training/logs/events.jsonl').read_bytes().splitlines()]
    assert len(events)==87 and [e['stage'] for e in events]==[p['stage'] for p in ready['training_plan'] for _ in range(p['updates'])]
    assert len(list(RUN.glob('grad_S*.json')))==87 and len(list(RUN.glob('adapter_S*.pt')))==19 and len(list(RUN.glob('grad_S*_first.pt')))==19
    for p in ready['training_plan']:
        stage=p['stage'];r=read(RUN/f'STAGE_{stage:02}_COMPLETE.json');assert r['stage']==stage and r['updates']==p['updates']==train['counts'][str(stage)]
        validity=read(RUN/f'training/validity/stage_{stage:02}.json')
        assert validity['passed'] and validity['actual_parameters_changed'] and validity['updates']==validity['positive_lr_updates']==p['updates']
        assert validity['parameters_before_sha256']!=validity['parameters_after_sha256']
        assert f'STAGE_{stage:02}_COMPLETE.json' in train['files'] and f'training/validity/stage_{stage:02}.json' in train['files']
        assert r['adapter_bytes']==17596416 and r['frozen_digest']==init['frozen_digest'] and r['vision_digest']==init['vision_digest']
        assert f'adapter_S{stage:02}.pt' in train['files'] and f'grad_S{stage:02}_first.pt' in train['files']
        for step in range(1,p['updates']+1):
            g=read(RUN/f'grad_S{stage:02}_{step:03}.json');assert g==dict(parameters=336,gradient_bytes=17596416,finite=True)
            assert f'grad_S{stage:02}_{step:03}.json' in train['files']
    assert 'INITIALIZATION.json' in train['files'] and 'training/logs/events.jsonl' in train['files']
    assert all(e['event']=='train_update' and finite(e['loss']) and all(finite(lr) and 0<lr<=.0001 for lr in e['learning_rate_used']) for e in events)
    for case,(stage,domain) in ENDS.items():
        e=complete[case];r=read(RUN/f'STAGE_{stage:02}_COMPLETE.json');restore=e['restore_receipt']
        assert e['stage']==stage and e['domain']==domain and e['layout']==train['layout'] and e['digest_before']==e['digest_after']
        assert e['snapshot_source_sha256']==train['files'][f'adapter_S{stage:02}.pt']
        assert restore['adapter_digest']==restore['snapshot_digest']==r['adapter_digest'] and restore['snapshot_bytes']==17596416
        assert restore['adapter_parameters']==336 and restore['adapter_coordinates']==4399104
        names={n+s for n in expected for s in ('.A.weight','.B.weight')}
        shapes={r['module']+suffix:r[field] for r in init['initialization'] for suffix,field in (('.A.weight','A_shape'),('.B.weight','B_shape'))}
        assert set(restore['snapshot_keys'])==set(restore['before_fold_trainable_names'])==names and restore['stage']==stage
        assert restore['snapshot_shapes']==shapes and restore['snapshot_dtypes']=={n:'torch.float32' for n in names}
        assert restore['snapshot_tensor_bytes']=={n:math.prod(s)*4 for n,s in shapes.items()}
        assert restore['complete_cumulative_snapshot_copied_not_added'] and restore['adapter_parameter_identities_preserved'] and restore['original_trainable_mask_preserved']
        assert restore['frozen_digest_before']==restore['frozen_digest_after']==init['frozen_digest']
        assert e['vision_digest']==init['vision_digest'] and e['buffer_digest']==init['buffer_digest']
        assert case+'_predictions.json' in e['files']
    predictions={}
    for label,path,reference,domain in [('full_all','anchor_predictions.json',parent_inputs,None),('compact_all','final_predictions.json',compact_inputs,None)]+[(c,c+'_predictions.json',compact_inputs,d) for c,(_,d) in ENDS.items()]:
        wanted={rid for rid,(t,_) in gold.items() if domain is None or t['domain']==domain};rows=keyed(read(RUN/path),wanted)
        for rid,r in rows.items():
            t,g=gold[rid];ref=reference[rid];assert r['domain']==t['domain'] and r['task']==t['task'] and r['references']==g['answers']==ref['references']
            assert r['generation_cap']==t['max_new_tokens']==ref['generation_cap']
            for k in ('prepared_sha256','transport_sha256','packed_tokens'):assert r[k]==ref[k]
            assert isinstance(r['prediction'],str) and isinstance(r['output_tokens'],list) and len(r['output_tokens'])<=r['generation_cap']
            assert all(isinstance(x,int) and not isinstance(x,bool) and x>=0 for x in r['output_tokens'])
            if label in ENDS:assert r['stage']==ENDS[label][0]
        predictions[label]=rows
    memory=read(RUN/'compact_eval_memory.json');pm=cf['memory']['parent'];assert len(memory)==len(pm)==5
    for a,b in zip(memory,pm):
        for k in ('id','domain','transport_sha256','text_sha256','input_shape','raw_pixels_sha256','tiles','packed_tokens','new_tokens'):assert a[k]==b[k]
        assert a['new_tokens']==32 and 0<a['baseline_allocated']<=a['peak_allocated']<=a['peak_reserved']
    nodes=[n for n in ast.parse((port/'runtime_utils.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('tokens','score')];assert len(nodes)==2
    scorer_ast=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest();assert scorer_ast==cf['scorer_ast_sha256']
    scope=dict(re=re,Counter=collections.Counter);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-score>','exec'),scope)
    write('VERIFY_STARTED.json',dict(unix=time.time(),new_scores=280,scorer_ast_sha256=scorer_ast,all_seven_phases_exited=True))
    quality={'full_wide':cf['quality']['parent'],'compact_wide':cf['quality']['final']}
    for label,rows in predictions.items():
        values=[dict(id=rid,domain=r['domain'],F1=scope['score'](r['prediction'],r['references'])['token_f1']) for rid,r in sorted(rows.items())]
        assert all(finite(v['F1']) and 0<=v['F1']<=100 for v in values)
        domains={d:dict(F1=sum(v['F1'] for v in values if v['domain']==d)/20,QA=20) for d in DOMAINS if any(v['domain']==d for v in values)}
        quality[label]=dict(domains=domains,per_id=values)
        if len(domains)==5:quality[label]['macro']=sum(v['F1'] for v in domains.values())/5
    differences={d:dict(full_all_minus_wide=quality['full_all']['domains'][d]['F1']-quality['full_wide']['domains'][d]['F1'],compact_all_minus_wide=quality['compact_all']['domains'][d]['F1']-quality['compact_wide']['domains'][d]['F1']) for d in DOMAINS}
    for v in differences.values():v['difference_of_differences']=v['compact_all_minus_wide']-v['full_all_minus_wide']
    domain_end={}
    for case,(stage,domain) in ENDS.items():
        values={v['id']:v['F1'] for v in quality[case]['per_id']};new_final={v['id']:v['F1'] for v in quality['compact_all']['per_id'] if v['domain']==domain}
        prior_values={v['id']:v['domain_end_F1'] for v in prior_domain['domains'][domain]['per_id']}
        end=quality[case]['domains'][domain]['F1'];f=quality['compact_all']['domains'][domain]['F1'];target=quality['full_wide']['domains'][domain]['F1']
        domain_end[domain]=dict(stage=stage,domain_end_F1=end,final_F1=f,parent_final_target=target,final_minus_domain_end=f-end,
            domain_end_minus_prior_wide=end-prior_domain['domains'][domain]['domain_end_F1'],
            target_attained_then_lost=end>=target and f<target,
            per_id=[dict(id=rid,domain_end_F1=values[rid],final_minus_domain_end=new_final[rid]-values[rid],domain_end_minus_prior_wide=values[rid]-prior_values[rid]) for rid in sorted(values)])
    floors=all(quality['compact_all']['domains'][d]['F1']>=quality['full_wide']['domains'][d]['F1'] for d in DOMAINS)
    peaks=all(a['peak_allocated']<b['peak_allocated'] for a,b in zip(memory,pm));package=train['checkpoint']['bytes']<old['parent']['bytes']
    passed=floors and peaks and package
    result=dict(complete=True,ready_sha256=rh,source_complete_sha256={c:sha(RUN/(c+'_COMPLETE.json')) for c in PHASES},quality=quality,allocation_differences=differences,
        gate_passed=passed,all_parent_domain_floors=floors,all5_GPU_peaks_lower=peaks,complete_package_smaller=package,complete_weights_parent=FULL,complete_weights_compact=SMALL,checkpoint=train['checkpoint'],
        decision='retain_for_independent_framework_review' if passed else 'close_fixed_compact_all_r8_recipe',model_seconds={c:e['seconds'] for c,e in complete.items()},
        anchor_CPU_extraction_seconds=extraction['seconds'],new_generations=285,new_scores=280,FIT_QA=1213,optimizer_updates=87,TEST_QA=0,NLL_requests=0,adapter_coordinates=4399104,
        memory=dict(parent=pm,compact=memory),training_memory=train['training_memory'],phase_RSS_highwater_bytes={c:e['process_RSS_highwater_bytes'] for c,e in complete.items()},
        scorer_ast_sha256=scorer_ast,prior_scores_reused_without_rescoring=True,prior_domain_readout=prior_domain['domains'],domain_end=domain_end,domain_boundaries={d:s for s,d in ENDS.values()},
        no_pure_location_learning_significance_or_framework_claim=True,verification_seconds=time.monotonic()-tick)
    write('VERIFIED.json',result)
    lines=['# 固定表示 × 更新分配：全部部署态读数','','| 域 | 完整wide | 完整all | 紧凑wide | 紧凑all | 分配差分之差pp |','|---|---:|---:|---:|---:|---:|']
    for d in DOMAINS:lines.append('| '+d+' | '+' | '.join(f"{quality[q]['domains'][d]['F1']:.8f}" for q in ('full_wide','full_all','compact_wide','compact_all'))+f" | {differences[d]['difference_of_differences']:+.8f} |")
    lines+=['','| 域 | 新all域末 | 新all末态 | 旧wide域末 | 末态−域末pp |','|---|---:|---:|---:|---:|']
    for d,v in domain_end.items():lines.append(f"| {d} | {v['domain_end_F1']:.8f} | {v['final_F1']:.8f} | {prior_domain['domains'][d]['domain_end_F1']:.8f} | {v['final_minus_domain_end']:+.8f} |")
    lines+=['',f"固定出口：{result['decision']}。各域父底线={floors}；完整包更小={package}；五GPU峰均更低={peaks}。",'域末四值见VERIFIED各独立20QA质量，不合并为新模型宏分。旧未fold all和正式负结果保留。','给定训练与FP16合并流程的整包分配对照；不识别纯FFN/容量/舍入，不称显著或新框架成功。','']
    with (RUN/'FINDINGS.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(VERIFIED_sha256=sha(RUN/'VERIFIED.json'),decision=result['decision'],compact_all_macro=quality['compact_all']['macro'],full_all_macro=quality['full_all']['macro'],gate_passed=passed)))
if __name__=='__main__':main()
