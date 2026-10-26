"""Verify completed fixed-window resource evidence; no models or scoring."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics
import time


def read(p): return json.loads(Path(p).read_bytes())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(root, desc):
    p=Path(desc['path']);p=p if p.is_absolute() else root/p
    assert sha(p)==desc['sha256'],str(p)
    return p


def verify(root, output, binding_path, worker_path):
    assert not (output/'FAILURE.json').exists()
    assert not (output/'VERIFIED.json').exists()
    b=read(binding_path); bh=sha(binding_path)
    assert sha(worker_path)==b['worker_sha256']
    proof_path=checked(root,b['formal_proof']);proof=read(proof_path)
    prior=read(checked(root,b['prior_windows']))
    start,end,ex=(read(output/name) for name in ('STARTED.json','COMPLETE.json','EXIT.json'))
    who=start['identity']
    assert who==end['identity']==ex['worker'] and who['pid']==who['pgid']
    proc=Path('/proc')/str(who['pid'])/'stat'
    if proc.exists():
        fields=proc.read_text().rsplit(') ',1)[1].split()
        assert fields[0]=='Z' or fields[19]!=str(who['startticks'])
    assert ex['returncode']==0 and end['complete']
    assert start['binding_sha256']==end['binding_sha256']==ex['binding_sha256']==bh
    assert end['source_sha256']['worker']==ex['worker_sha256']==b['worker_sha256']
    for k,desc in b['helpers'].items():
        assert sha(checked(root,desc))==end['source_sha256']['helpers'][k]
    assert start['physical_GPU']==end['physical_GPU']==ex['GPU']
    assert start['virtual_GPU']==end['virtual_GPU']==0
    assert b['GPU_uuids'][str(ex['GPU'])]==end['GPU_uuid']
    assert start['limit']==600 and math.isfinite(end['seconds']) and 0<end['seconds']<=600
    assert 0<end['checkpoint_load_seconds']<=end['seconds']
    assert end['backbone']==b['backbone']==proof['candidate']['backbone']
    assert proof['candidate']['method']=='SeqFT_LoRA' and proof['condition']=='pct10'
    assert end['formal_proof_sha256']==sha(proof_path)
    for name in ('checkpoint_sha256','checkpoint_bytes','config_sha256'):
        assert end[name]==proof[name]
    for name in ('original_full_unfolded_graph','state_and_registered_buffers_unchanged',
                 'generation_public_unchanged','no_reconstruction_or_export','audit_digest_after_captured_peak'):
        assert end[name] is True
    assert end['checkpoint_deleted_or_rewritten'] is False and end['audit_hashes_in_timing'] is False
    assert end['resident_adapter_banks']==end['adapter_structure']['resident_adapter_banks']==1
    assert end['layout']['trainable']==0
    for name,value in dict(new_generations=5,forced_new_tokens=160,new_scores=0,NLL_requests=0,backwards=0,optimizer_updates=0).items():
        assert end[name]==value
    rows=read(output/'windows.json');assert sha(output/'windows.json')==end['windows_sha256']
    assert len(rows)==len(prior)==5
    for domain,r,p in zip(('natural','medical','geography','finance','traffic'),rows,prior):
        assert r['domain']==p['domain']==domain and r['new_tokens']==32
        for key in ('id','raw_input_sha256','image_SHA256','image_shapes','retrieved_chunk_indexes','common_image_area_cap'):
            assert r[key]==p[key],key
        assert not r['visual_cache_hit'] and r['audit_hashes_in_timing'] is False
        assert r['finite_logit_guard_in_timing'] and r['synchronized_token_callbacks']
        assert r['input_shape']==[1,r['prompt_tokens']]==r['input_tensor_shapes']['input_ids']
        for key in ('baseline_allocated','peak_allocated','peak_reserved'):
            assert type(r[key]) is int and r[key]>0
        assert r['baseline_allocated']<=r['peak_allocated']<=r['peak_reserved']
        t=r['timing'];assert t['generated_tokens']==32 and t['ttft_includes_input_preparation']
        for key in ('ttft_seconds','decode_tokens_per_second','decode_interval_seconds'):
            assert math.isfinite(t[key]) and t[key]>0
        assert math.isclose(t['decode_tokens_per_second'],31/t['decode_interval_seconds'],rel_tol=1e-12)
    result=dict(complete=True,backbone=b['backbone'],method='SeqFT_LoRA',condition='pct10',
        source_sha256={n:sha(output/n) for n in ('STARTED.json','COMPLETE.json','EXIT.json','windows.json')},
        binding_sha256=bh,worker_sha256=sha(worker_path),verifier_sha256=sha(__file__),
        formal_proof_sha256=sha(proof_path),checkpoint_sha256=end['checkpoint_sha256'],
        GPU_uuid=end['GPU_uuid'],physical_GPU=end['physical_GPU'],layout=end['layout'],
        adapter_coordinates=end['adapter_structure']['adapter_coordinates'],resident_adapter_banks=1,
        max_peak_allocated_MiB=max(r['peak_allocated'] for r in rows)/2**20,
        max_peak_reserved_MiB=max(r['peak_reserved'] for r in rows)/2**20,
        median_TTFT_seconds=statistics.median(r['timing']['ttft_seconds'] for r in rows),
        median_decode_tokens_per_second=statistics.median(r['timing']['decode_tokens_per_second'] for r in rows),
        mean_TTFT_seconds=statistics.fmean(r['timing']['ttft_seconds'] for r in rows),
        mean_decode_tokens_per_second=statistics.fmean(r['timing']['decode_tokens_per_second'] for r in rows),
        windows=rows,new_models=0,new_scores=0,verified_unix=time.time(),
        interpretation='Five fixed serial windows after one reload; instrumented timing; not full-TEST worst peak, per-question cold starts, or production throughput.')
    with (output/'VERIFIED.json').open('x',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return dict(output=str(output),verified_sha256=sha(output/'VERIFIED.json'),new_models=0,new_scores=0)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','output','binding','worker'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.root,a.output,a.binding,a.worker)))
