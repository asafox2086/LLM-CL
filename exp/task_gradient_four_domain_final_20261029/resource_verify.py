"""Verify four-domain S16 fixed-window resources, no score/model/QA consumption.

CLI --root ABS --output ABS --binding ABS --worker ABS. EXIT schema: worker identity, GPU, returncode, binding_sha256, worker_sha256.
The caller supplies EXIT.json only after this worker's real process has exited.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import statistics
import time

METHOD='TaskGradReuse'
GPU_UUID='GPU-3c72e4ce-0896-0b40-d610-be585d2bb51e'
PRIOR_SHA='45bb9f4ca6d233b139f3c6a9b0255d29eee07e160b085fcef487bad8a3d8d7cd'


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(root,desc):
    p=Path(desc['path']);assert p.is_absolute() or '..' not in p.parts
    p=p if p.is_absolute() else root/p
    assert sha(p)==desc['sha256'],str(p)
    return p


def verify(root,output,binding_path,worker_path):
    assert root.is_absolute() and output.is_absolute() and binding_path.is_absolute()
    assert not (output/'FAILURE.json').exists() and not (output/'VERIFIED.json').exists()
    b=read(binding_path);bh=sha(binding_path)
    assert sha(worker_path)==b['worker_sha256'] and sha(__file__)==b['verifier_sha256']
    assert b['schema']=='task-gradient-four-domain-resource-v1'
    assert b['worker_sha256']=='cde2520f7e0cd4f6764bf655cb63a43f4c87d22c327cf02d9a94afb86f39dd3e'
    result_path=checked(root,b['quality_result']);quality=read(result_path)
    certificate_path=checked(root,b['quality_verified']);certificate=read(certificate_path)
    verifier_path=checked(root,b['quality_verifier'])
    assert quality['verifier_sha256']==certificate['verifier_sha256']==sha(verifier_path)
    assert certificate['result_sha256']==sha(result_path) and certificate['existing_answer_records_checked']==19331
    assert quality['status']==certificate['status']=='FOUR_DOMAIN_NATIVE_SCORE_METADATA_VERIFIED_DEPLOYMENT_PENDING'
    assert certificate['new_score_calls']==quality['new_scoring_calls']==0
    assert quality['method']==METHOD and quality['backbone']=='onevision' and quality['terminal_stage']==16
    fb_path=checked(root,b['quality_binding']);fb=read(fb_path)
    assert sha(fb_path)=='4c7a351c3e9873384652ad4b01d3cfbb0b15acb6b3b2613d80e4b1037863dd89'
    assert fb['schema']=='task-gradient-four-domain-final-v1' and fb['GPU']==1
    assert quality['binding_sha256']==certificate['binding_sha256']==sha(fb_path)
    assert fb['worker_sha256']=='7221a74e7553e7bf3a7ef26ad07e86037e56e4e177de63268d3637e4894daceb'
    complete_path=checked(root,b['quality_complete']);quality_complete=read(complete_path)
    assert sha(complete_path)==quality['worker_complete_sha256']
    assert quality_complete['source_stage']==quality_complete['terminal_stage']==16
    assert quality_complete['new_generations']==19331 and quality_complete['new_training_updates']==0
    assert quality_complete['binding_sha256']==sha(fb_path)
    meta_path=checked(root,b['checkpoint_metadata']);meta=read(meta_path)
    assert meta==quality['final_checkpoint']==quality_complete['checkpoint']
    assert meta['stage']==16 and meta['path']=='checkpoints/stage_16.pt' and meta['method']==METHOD
    assert meta['sha256']==b['checkpoint_sha256']
    logical_run='exp/task_gradient_four_domain_final_20261029/runs/onevision/pct01/final_s16_v1'
    original=root/logical_run
    assert meta_path==original/'checkpoints/stage_16.json' and meta==read(original/'checkpoints/latest.json')
    assert (original/meta['path']).stat().st_size==meta['bytes']
    config_path=original/'config.json';config=read(config_path)
    assert sha(config_path)==b['config_sha256']
    assert config['manifest_sha256']==meta['manifest_sha256']=='6ffae80b7e290b4bcd7b3ef8df7613c176a344221d28e874125f3faf87bd6798'
    assert config['terminal_stage']==16 and config['postprocess_binding_sha256']==sha(fb_path)
    fold_path=checked(root,b['quality_fold']);fold=read(fold_path)
    assert fold_path==original/'FOLD.json' and fold['dense_state']==quality_complete['dense_state']==quality_complete['state_after_evaluation']
    assert fold['terminal_stage']==16 and fold['original_fold_function'] and fold['training_context_cleared']
    for path in [complete_path,config_path,meta_path,fold_path]:
        assert certificate['source_ledger'][str(path)]['sha256']==sha(path)
    prior_path=checked(root,b['prior_windows']);assert sha(prior_path)==PRIOR_SHA
    all_prior=read(prior_path)
    assert len(all_prior)==5 and [x['domain'] for x in all_prior]==['natural','medical','geography','finance','traffic']
    prior=all_prior[:4]
    start,end,ex=(read(output/n) for n in ('STARTED.json','COMPLETE.json','EXIT.json'))
    who=start['identity'];assert who==end['identity']==ex['worker'] and who['pid']==who['pgid']
    p=Path('/proc')/str(who['pid'])/'stat'
    if p.exists():
        fields=p.read_text().rsplit(') ',1)[1].split()
        assert fields[0]=='Z' or fields[19]!=str(who['startticks'])
    assert ex['returncode']==0 and end['complete']
    assert start['binding_sha256']==end['binding_sha256']==ex['binding_sha256']==bh
    assert end['source_sha256']['worker']==ex['worker_sha256']==b['worker_sha256']
    assert end['source_sha256']['quality_binding']==sha(fb_path)
    for k,desc in b['helpers'].items():assert sha(checked(root,desc))==end['source_sha256']['helpers'][k]
    assert start['physical_GPU']==end['physical_GPU']==ex['GPU']==1
    assert start['virtual_GPU']==end['virtual_GPU']==0
    assert b['GPU_uuids']['1']==end['GPU_uuid']==GPU_UUID
    assert start['limit']==600 and math.isfinite(end['seconds']) and 0<end['seconds']<=600
    assert 0<end['checkpoint_load_seconds']<=end['seconds']
    assert end['quality_result_sha256']==sha(result_path)
    assert end['quality_verified_sha256']==sha(certificate_path)
    assert end['checkpoint_sha256']==meta['sha256'] and end['checkpoint_bytes']==meta['bytes']
    assert end['config_sha256']==sha(config_path)==b['config_sha256']
    assert end['backbone']=='onevision' and end['sealed_logical_run']==logical_run and end['original_run']==str(original)
    for n in ('original_full_folded_graph','state_and_registered_buffers_unchanged',
              'generation_public_unchanged','no_reconstruction_or_export','audit_digest_after_captured_peak',
              'final_qv_values_match_saved_fold_digests','no_extra_fold_or_dtype_conversion',
              'original_other_CPU_payload_retained'):
        assert end[n] is True,n
    assert end['checkpoint_deleted_or_rewritten'] is False and end['audit_hashes_in_timing'] is False
    assert end['layout']['weights']==dict(coefficients=893675552,bytes=1787351104)
    assert end['layout']['ordinary_buffers']['bytes']==9032 and end['layout']['trainable']==0
    assert end['dtype_storage']=={'torch.float16':dict(bytes=1787351104,coefficients=893675552)}
    structure=end['adapter_structure']
    assert end['resident_adapter_banks']==structure['resident_adapter_banks']==structure['adapter_coordinates']==0
    assert structure['projection_modules']==len(structure['layers'])==48
    assert structure['no_AB_bank_G_or_TRAIN_context'] and structure['method_owned_tensor_storages']==0
    for kind in ('loaded_payload_references','retained_payload_references'):
        refs=end[kind]
        assert refs['outside_model_CUDA_storages']==0 and refs['actual_payload_retained']
        assert refs['unknown_tensor_bank_storages']==0
        assert all(not r['device'].startswith('cuda') for r in refs['outside_model']['storages'])
        assert all(r['first_path'].startswith('root.rng.') for r in refs['outside_model']['storages'])
        assert all('original_inv_freq' in r['first_path'] or '_llmcl_visual_cache' in r['first_path']
                   for r in refs['classified_model_unregistered']['storages'])
    assert 0<end['checkpoint_load_allocated']<=end['checkpoint_load_peak_allocated']
    assert end['checkpoint_load_reserved']>=end['checkpoint_load_allocated']
    for n,v in dict(new_generations=4,forced_new_tokens=128,new_scores=0,NLL_requests=0,backwards=0,optimizer_updates=0).items():
        assert end[n]==v
    rows=read(output/'windows.json');assert sha(output/'windows.json')==end['windows_sha256']
    assert len(rows)==len(prior)==4
    assert end['measured_domains']==['natural','medical','geography','finance']
    assert end['historical_PAS_comparison_windows']==4
    assert end['historical_PAS_four_window_max_peak_allocated']==max(p['peak_allocated'] for p in prior)
    for d,r,p in zip(('natural','medical','geography','finance'),rows,prior):
        assert r['domain']==p['domain']==d and r['new_tokens']==32
        for k in ('id','raw_input_sha256','prepared_sha256','transport_sha256','image_SHA256',
                  'image_shapes','retrieved_chunk_indexes','common_image_area_cap','input_shape','input_tensor_shapes'):
            assert r[k]==p[k],k
        assert not r['visual_cache_hit'] and not r['audit_hashes_in_timing']
        assert r['finite_logit_guard_in_timing'] and r['synchronized_token_callbacks']
        assert r['input_shape']==[1,r['prompt_tokens']]==r['input_tensor_shapes']['input_ids']
        for k in ('baseline_allocated','peak_allocated','peak_reserved'):assert type(r[k]) is int and r[k]>0
        assert r['baseline_allocated']<=r['peak_allocated']<=r['peak_reserved']
        t=r['timing'];assert t['generated_tokens']==32 and t['ttft_includes_input_preparation']
        for k in ('ttft_seconds','decode_tokens_per_second','decode_interval_seconds'):
            assert math.isfinite(t[k]) and t[k]>0
        assert math.isclose(t['decode_tokens_per_second'],31/t['decode_interval_seconds'],rel_tol=1e-12)
    result=dict(complete=True,backbone='onevision',method=METHOD,condition='pct01',
        source_sha256={n:sha(output/n) for n in ('STARTED.json','COMPLETE.json','EXIT.json','windows.json')},
        binding_sha256=bh,worker_sha256=sha(worker_path),verifier_sha256=sha(__file__),
        quality_result_sha256=sha(result_path),quality_verified_sha256=sha(certificate_path),
        quality_binding_sha256=sha(fb_path),terminal_stage=16,
        final_F1_same_checkpoint=dict(domains=quality['domains'],domain_macro_final=quality['domain_macro_final'],AP16=quality['AP16']),
        checkpoint_sha256=end['checkpoint_sha256'],
        checkpoint_bytes=end['checkpoint_bytes'],GPU_uuid=GPU_UUID,physical_GPU=1,
        layout=end['layout'],dtype_storage=end['dtype_storage'],adapter_coordinates=0,resident_adapter_banks=0,
        loaded_payload_references=end['loaded_payload_references'],retained_payload_references=end['retained_payload_references'],
        checkpoint_load_allocated=end['checkpoint_load_allocated'],checkpoint_load_peak_allocated=end['checkpoint_load_peak_allocated'],
        max_peak_allocated_MiB=max(r['peak_allocated'] for r in rows)/2**20,
        max_peak_reserved_MiB=max(r['peak_reserved'] for r in rows)/2**20,
        window_count=4,forced_new_tokens=128,weight_coefficients=end['layout']['weights']['coefficients'],
        weight_bytes=end['layout']['weights']['bytes'],weight_MiB=end['layout']['weights']['bytes']/2**20,
        historical_PAS_four_window_max_peak_allocated_MiB=max(p['peak_allocated'] for p in prior)/2**20,
        historical_PAS_per_window=[dict(domain=p['domain'],peak_allocated=p['peak_allocated']) for p in prior],
        per_window_allocated_reduction_percent=[dict(domain=r['domain'],percent=100*(1-r['peak_allocated']/p['peak_allocated'])) for r,p in zip(rows,prior)],
        median_TTFT_seconds=statistics.median(r['timing']['ttft_seconds'] for r in rows),
        median_decode_tokens_per_second=statistics.median(r['timing']['decode_tokens_per_second'] for r in rows),
        mean_TTFT_seconds=statistics.fmean(r['timing']['ttft_seconds'] for r in rows),
        mean_decode_tokens_per_second=statistics.fmean(r['timing']['decode_tokens_per_second'] for r in rows),
        windows=rows,new_models=0,new_scores=0,verified_unix=time.time(),
        interpretation='Four fixed OLD GPU1/PAS inputs with equal prepared tensors; evaluated S16 dense graph. '
          'Instrumented serial timing, not full-TEST worst peak, cold starts or production speed. '
          'PAS prior has memory only and was trained over five domains; no matched four-domain quality/resource superiority or PAS TTFT/speed superiority follows.')
    with (output/'VERIFIED.json').open('x',encoding='utf8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return dict(output=str(output),verified_sha256=sha(output/'VERIFIED.json'),new_models=0,new_scores=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('root','output','binding','worker'):parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(verify(args.root,args.output,args.binding,args.worker)))
