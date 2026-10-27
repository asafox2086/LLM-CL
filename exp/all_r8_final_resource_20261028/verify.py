"""Verify the completed original all-r8 fixed-window resource run, no score/model.

Same CLI and EXIT receipt contract as final_seq_resource_20261026/verify.py.
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

METHOD='LoRA_all'
VERIFIED_SHA='abbd2da43753c6bbeb137d7bef535983efbcafdaf08686f61550886b2f18598d'
READY_SHA='a8c4b99aef2e7b5e803d027e45d31aebd5ad60df49fea7c2b459d719249da628'
COMPLETE_SHA='ea9bd2283e2cc7d8948267289ff2e4cc7a20685cdc7b511fc176b628870d6a5c'
CP_SHA='416ed177803b9bf986dacd73a6d1bf18c6c0c59a1eb65f42785d81a628d31519'
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
    proof_path=checked(root,b['original_verified']);assert sha(proof_path)==VERIFIED_SHA
    ready_path=checked(root,b['original_ready']);assert sha(ready_path)==READY_SHA
    complete_path=checked(root,b['original_complete']);assert sha(complete_path)==COMPLETE_SHA
    original=read(proof_path);complete=read(complete_path);ready=read(ready_path)
    assert original['complete'] and complete['complete'] and complete['case']==METHOD
    assert original['quality']==complete['quality'] and original['checkpoint']==complete['checkpoint']
    meta=read(checked(root,b['checkpoint_metadata']));assert meta==original['checkpoint']
    assert meta['stage']==19 and meta['method']==METHOD and meta['sha256']==CP_SHA and meta['bytes']==1941599549
    assert original['ready_sha256']==READY_SHA and original['completion_sha256']==COMPLETE_SHA
    prior_path=checked(root,b['prior_windows']);assert sha(prior_path)==PRIOR_SHA
    prior=read(prior_path)
    start,end,ex=(read(output/n) for n in ('STARTED.json','COMPLETE.json','EXIT.json'))
    who=start['identity'];assert who==end['identity']==ex['worker'] and who['pid']==who['pgid']
    p=Path('/proc')/str(who['pid'])/'stat'
    if p.exists():
        fields=p.read_text().rsplit(') ',1)[1].split()
        assert fields[0]=='Z' or fields[19]!=str(who['startticks'])
    assert ex['returncode']==0 and end['complete']
    assert start['binding_sha256']==end['binding_sha256']==ex['binding_sha256']==bh
    assert end['source_sha256']['worker']==ex['worker_sha256']==b['worker_sha256']
    assert end['source_sha256']['original_READY']==READY_SHA
    assert end['source_sha256']['study']==ready['sources']['exp/osft_increment_screen_20261019/v1/study.py']
    for k,desc in b['helpers'].items():assert sha(checked(root,desc))==end['source_sha256']['helpers'][k]
    assert start['physical_GPU']==end['physical_GPU']==ex['GPU']==1
    assert start['virtual_GPU']==end['virtual_GPU']==0
    assert b['GPU_uuids']['1']==end['GPU_uuid']==GPU_UUID
    assert start['limit']==600 and math.isfinite(end['seconds']) and 0<end['seconds']<=600
    assert 0<end['checkpoint_load_seconds']<=end['seconds']
    assert end['formal_proof_sha256']==VERIFIED_SHA
    assert end['original_COMPLETE_sha256']==COMPLETE_SHA and end['original_READY_sha256']==READY_SHA
    assert end['checkpoint_sha256']==meta['sha256'] and end['checkpoint_bytes']==meta['bytes']
    assert end['config_sha256']==b['config_sha256']=='d346e0174fa6c0f714b94e161f617375268e13c5f1d228c0d03c43af933bfbca'
    assert end['backbone']=='onevision' and end['sealed_logical_run']=='exp/osft_increment_confirmation_20261022/v1/private'
    for n in ('original_full_unfolded_graph','state_and_registered_buffers_unchanged',
              'generation_public_unchanged','no_reconstruction_or_export','audit_digest_after_captured_peak',
              'no_fold_cast_or_export','old_closed_quality_result_unchanged',
              'original_other_CPU_payload_retained'):
        assert end[n] is True,n
    assert end['checkpoint_deleted_or_rewritten'] is False and end['audit_hashes_in_timing'] is False
    assert end['layout']['weights']==dict(coefficients=898074656,bytes=1804947520)
    assert end['layout']['ordinary_buffers']['bytes']==9032 and end['layout']['trainable']==0
    assert end['dtype_storage']=={'torch.float16':dict(bytes=1787351104,coefficients=893675552),
                                  'torch.float32':dict(bytes=17596416,coefficients=4399104)}
    structure=end['adapter_structure']
    assert end['resident_adapter_banks']==structure['resident_adapter_banks']==1
    assert structure['adapter_coordinates']==end['saved_requires_grad_coordinates']==4399104
    assert structure['projection_modules']==len(structure['layers'])==168
    assert structure['method_layer_list_aliases_registered_modules'] and structure['no_additional_method_tensor_bank']
    assert end['inference_parameter_freezing_only'] and end['original_checkpoint_saved_before_TEST']
    assert end['coverage']=='final_only_19_tasks_not_74_cells'
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
    for n,v in dict(new_generations=5,forced_new_tokens=160,new_scores=0,NLL_requests=0,backwards=0,optimizer_updates=0).items():
        assert end[n]==v
    rows=read(output/'windows.json');assert sha(output/'windows.json')==end['windows_sha256']
    assert len(rows)==len(prior)==5
    for d,r,p in zip(('natural','medical','geography','finance','traffic'),rows,prior):
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
        formal_proof_sha256=VERIFIED_SHA,checkpoint_sha256=end['checkpoint_sha256'],
        checkpoint_bytes=end['checkpoint_bytes'],GPU_uuid=GPU_UUID,physical_GPU=1,
        layout=end['layout'],dtype_storage=end['dtype_storage'],adapter_coordinates=4399104,resident_adapter_banks=1,
        loaded_payload_references=end['loaded_payload_references'],retained_payload_references=end['retained_payload_references'],
        checkpoint_load_allocated=end['checkpoint_load_allocated'],checkpoint_load_peak_allocated=end['checkpoint_load_peak_allocated'],
        max_peak_allocated_MiB=max(r['peak_allocated'] for r in rows)/2**20,
        max_peak_reserved_MiB=max(r['peak_reserved'] for r in rows)/2**20,
        median_TTFT_seconds=statistics.median(r['timing']['ttft_seconds'] for r in rows),
        median_decode_tokens_per_second=statistics.median(r['timing']['decode_tokens_per_second'] for r in rows),
        mean_TTFT_seconds=statistics.fmean(r['timing']['ttft_seconds'] for r in rows),
        mean_decode_tokens_per_second=statistics.fmean(r['timing']['decode_tokens_per_second'] for r in rows),
        windows=rows,new_models=0,new_scores=0,verified_unix=time.time(),
        interpretation='Five fixed OLD GPU1/PAS inputs with equal prepared tensors; original S19 unmerged all-r8 graph. '
          'Original unmerged graph, original dtype and closed final19 quality identity; no fold/quality retest. '
          'Instrumented serial timing, not full-TEST worst peak, cold starts or production speed. '
          'PAS prior has memory only; no PAS TTFT/speed superiority follows.')
    with (output/'VERIFIED.json').open('x',encoding='utf8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return dict(output=str(output),verified_sha256=sha(output/'VERIFIED.json'),new_models=0,new_scores=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('root','output','binding','worker'):parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(verify(args.root,args.output,args.binding,args.worker)))
