"""Verify original Qwen five-window resource evidence; no scoring/model load."""
from pathlib import Path
import hashlib
import json
import math
import os
import time

ROOT = Path(os.environ['LLMCL_ROOT'])
RUN = Path(os.environ['LLMCL_RUN'])


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (RUN / name).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def main():
    tick = time.monotonic()
    assert not any((RUN / n).exists() for n in ('FAILURE.json', 'VERIFY_STARTED.json', 'VERIFIED.json'))
    ready = read(RUN / 'READY.json'); rh = sha(RUN / 'READY.json')
    for field in ('sources', 'prior_files'):
        for name, expected in ready[field].items():
            assert sha(RUN / name) == expected, name
    for field in ('runtime_files', 'environment'):
        for name, expected in ready[field].items():
            path = Path(name); assert sha(path if path.is_absolute() else ROOT / path) == expected, name
    start, end, ex = (read(RUN / n) for n in ('STARTED.json', 'COMPLETE.json', 'EXIT.json'))
    who = start['identity']; assert who == end['identity'] and who['uid'] == os.getuid()
    assert who['pid'] == who['pgid'] == ex['pid'] and ex['returncode'] == 0
    path = Path('/proc') / str(who['pid']) / 'stat'
    if path.exists():
        assert path.read_text().rsplit(') ', 1)[1].split()[19] != str(who['startticks'])
    assert end['complete'] and start['ready_sha256'] == end['ready_sha256'] == rh
    assert start['limit'] == 600 and math.isfinite(end['seconds']) and 0 < end['seconds'] <= 600
    assert ex['unix'] >= start['unix'] + end['seconds']
    assert start['physical_GPU'] == end['physical_GPU'] == 1 and start['virtual_GPU'] == end['virtual_GPU'] == 0
    assert start['GPU_uuid'] == end['GPU_uuid'] == ready['GPU_uuid']
    assert end['checkpoint_sha256'] == ready['checkpoint_sha256'] == 'd3df6622f2de818e3b5cb5209c106cc2bc827ae87a65841af823032dcd3d2f85'
    assert end['checkpoint_bytes'] == ready['checkpoint_bytes'] == 4562999772
    assert end['original_config_sha256'] == ready['config_sha256'] == '9db372397a602a25499fe83c74871fab9696bdadf5442352867a55277a2ccea1'
    formal = read(RUN / 'formal_Qwen_COMPLETE.json')
    assert formal['checkpoint_sha256'] == end['checkpoint_sha256'] and formal['config_sha256'] == end['original_config_sha256']
    assert formal['candidate']['backbone'] == 'qwen' and formal['candidate']['method'] == 'SeqFT_LoRA' and formal['final_QA'] == 21454
    assert end['layout']['weights'] == dict(coefficients=2210075136, bytes=4422329344)
    assert end['layout']['ordinary_buffers']['bytes'] == 7504 and end['layout']['trainable'] == 0
    assert len(end['original_structure']) == 56 and all(x['scale'] == 4 and x['dropout'] == .1 and x['base_dtype'] == 'torch.float16' and x['adapter_dtype'] == 'torch.float32' for x in end['original_structure'].values())
    assert all(end[n] for n in ('state_and_registered_buffers_unchanged', 'original_generation_objects_retained',
        'original_dtype_and_unfolded_graph', 'no_reconstruction_or_export', 'cross_backbone_raw_inputs_equal'))
    assert not end['cross_backbone_prepared_equality_required'] and not end['latency_measured']
    assert end['generation_before']['public'] == end['generation_after']['public']
    for name in ('state_digest', 'all_buffers_digest'):
        assert isinstance(end[name], str) and len(end[name]) == 64
    for name, expected in dict(new_generations=5, forced_new_tokens=160, new_scores=0, NLL_requests=0, backwards=0, optimizer_updates=0).items():
        assert end[name] == expected
    memory = read(RUN / 'memory.json'); assert sha(RUN / 'memory.json') == end['memory_sha256'] and len(memory) == 5
    naive_ready = read(RUN / 'prior_naive_READY.json'); naive = read(RUN / 'prior_naive_VERIFIED.json')
    ev = read(RUN / 'prior_naive_eval_COMPLETE.json'); prior = read(RUN / 'prior_naive_memory.json')
    assert naive['complete'] and naive['ready_sha256'] == sha(RUN / 'prior_naive_READY.json')
    assert sha(RUN / 'prior_naive_eval_COMPLETE.json') == naive['source_COMPLETE_sha256']['eval_qv']
    assert ev['files']['qv_memory.json'] == sha(RUN / 'prior_naive_memory.json') and prior == naive['memory']['plain_r8']
    assert naive['same_GPU_for_final_resource_windows'] == naive_ready['phase_GPUs']['eval_qv'] == 1
    comparisons = []
    for domain, current, old in zip(('natural', 'medical', 'geography', 'finance', 'traffic'), memory, prior):
        assert current['domain'] == old['domain'] == domain and current['new_tokens'] == old['new_tokens'] == 32
        for name in ('id', 'raw_input_sha256', 'image_SHA256', 'image_shapes', 'retrieved_chunk_indexes', 'common_image_area_cap'):
            assert current[name] == old[name], name
        assert not current['visual_cache_hit']
        assert current['input_shape'] == [1, current['prompt_tokens']] == current['input_tensor_shapes']['input_ids']
        for name in ('prepared_sha256', 'transport_sha256', 'raw_input_sha256'):
            assert isinstance(current[name], str) and len(current[name]) == 64
        assert all(type(current[n]) is int and current[n] > 0 for n in ('baseline_allocated', 'peak_allocated', 'peak_reserved'))
        assert current['baseline_allocated'] <= current['peak_allocated'] <= current['peak_reserved']
        comparisons.append(dict(domain=domain, id=current['id'], Qwen_peak_allocated=current['peak_allocated'],
            naive_OneVision_peak_allocated=old['peak_allocated'], OneVision_reduction_percent=100 * (current['peak_allocated'] - old['peak_allocated']) / current['peak_allocated']))
    write('VERIFY_STARTED.json', dict(unix=time.time(), ready_sha256=rh, new_scores=0, new_models=0))
    result = dict(complete=True, ready_sha256=rh, STARTED_sha256=sha(RUN / 'STARTED.json'), COMPLETE_sha256=sha(RUN / 'COMPLETE.json'), EXIT_sha256=sha(RUN / 'EXIT.json'),
        actual_successful_exit=True, physical_GPU=1, virtual_GPU=0, GPU_uuid=end['GPU_uuid'], checkpoint_sha256=end['checkpoint_sha256'],
        checkpoint_bytes=end['checkpoint_bytes'], original_config_sha256=end['original_config_sha256'], formal_proof_sha256=sha(RUN / 'formal_Qwen_COMPLETE.json'),
        prior_naive_VERIFIED_sha256=sha(RUN / 'prior_naive_VERIFIED.json'), memory_sha256=end['memory_sha256'], memory=memory,
        original_layout=end['layout'], original_unfolded_FP16_base_FP32_AB=True, original_state_unchanged=True,
        original_graph_no_reconstruction_or_export=True, naive_OneVision_comparison=comparisons, cross_backbone_prepared_equality_required=False,
        new_generations=5, new_scores=0, NLL_requests=0, backwards=0, optimizer_updates=0, latency_measured=False,
        model_seconds=end['seconds'], verification_seconds=time.monotonic() - tick,
        interpretation='Original sealed strongest-Qwen SeqFT graph, original FP16 base/FP32r8 adapters on the same physical GPU1 five raw-input/32-token windows. Native processor/tokenization/patch features differ across backbones. This is an original unfused implementation resource reference, not optimal folded deployment, full-distribution peak, new F1, latency or evidence that the new method meets the joint goal.')
    write('VERIFIED.json', result)
    print(json.dumps(dict(VERIFIED_sha256=sha(RUN / 'VERIFIED.json'), new_scores=0, model_seconds=end['seconds'])))


if __name__ == '__main__':
    main()
