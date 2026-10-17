"""Full native-task GPU admission, not a formal continual-learning result."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'exp/five_domain_native_cl_20261016_sparse'))
from methods import SeqLinear
from resource_monitor import ResourceMonitor
from runtime_utils import write

MODEL = ROOT / 'model/LLaVA-OneVision-Qwen2-0.5B-OV'
MANIFEST = ROOT / 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def native_input(row, task, processor, model):
    images = []
    content = []
    markers = row['question'].count('<image>')
    if markers:
        assert markers == len(row['images'])
    for rel in row['images']:
        with Image.open(ROOT / rel) as im:
            images.append(im.convert('RGB'))
        if not markers:
            content.append({'type': 'image'})
    text = task['instruction']
    if row.get('context'):
        text += '\nContext: ' + row['context']
    content.append({'type': 'text', 'text': text + '\nQuestion: ' + row['question']})
    prompt = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                            tokenize=False, add_generation_prompt=True)
    assert prompt.count('<image>') == len(images)
    batch = processor(text=prompt, images=images, return_tensors='pt')
    batch = {k: v.cuda().to(torch.float16) if v.is_floating_point() else v.cuda() for k, v in batch.items()}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        features = model.get_image_features(batch['pixel_values'], batch['image_sizes'],
                                            model.config.vision_feature_layer,
                                            model.config.vision_feature_select_strategy)
        features, _ = model.pack_image_features(features, batch['image_sizes'], model.image_newline,
                                                model.config.vision_aspect_ratio)
    ids = batch['input_ids']
    token = model.config.image_token_index
    positions = (ids[0] == token).nonzero().flatten()
    assert len(positions) and len(images) == 1
    assert int(positions[-1] - positions[0] + 1) == len(positions)
    if len(positions) != len(features):
        begin, end = int(positions[0]), int(positions[-1]) + 1
        revised = torch.cat([ids[:, :begin], torch.full((1, len(features)), token, dtype=ids.dtype,
                             device=ids.device), ids[:, end:]], dim=1)
        assert torch.equal(ids[ids != token], revised[revised != token])
        batch['input_ids'] = revised
        batch['attention_mask'] = torch.ones_like(revised)
    return batch, features


def loss_pair(row, task, processor, model, official=False):
    batch, features = native_input(row, task, processor, model)
    target = processor.tokenizer(row['answer'], add_special_tokens=False)['input_ids'][:512]
    target += [processor.tokenizer.eos_token_id]
    labels = torch.tensor([target], device='cuda')
    ids = torch.cat([batch['input_ids'], labels], dim=1)
    mask = (ids == model.config.image_token_index).unsqueeze(-1).expand(-1, -1, model.config.text_config.hidden_size)
    embeddings = model.get_input_embeddings()(ids).masked_scatter(mask, features.to(model.dtype))
    with torch.autocast('cuda', dtype=torch.float16):
        logits = model.language_model(inputs_embeds=embeddings, attention_mask=torch.ones_like(ids),
                                      use_cache=False, num_logits_to_keep=len(target) + 1).logits[:, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.flatten())
        error = None
        if official:
            batch.update(input_ids=ids, attention_mask=torch.ones_like(ids))
            reference = model(**batch, use_cache=False, num_logits_to_keep=len(target) + 1).logits[:, :-1].float()
            native_loss = torch.nn.functional.cross_entropy(reference.reshape(-1, reference.shape[-1]), labels.flatten())
            error = abs(float(native_loss) - float(loss))
            assert error < .015 and torch.isfinite(native_loss)
    return loss, error, int(features.shape[0])


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    assert not subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid',
                                         '--format=csv,noheader'], text=True).strip()
    report = ROOT / 'summary/onevision_native_cl_20261017_pct01'
    report.mkdir(exist_ok=True)
    run = HERE / 'preflight_runs' / __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    (run / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write(report / 'current_attempt.json', {'run': str(run.relative_to(ROOT)), 'pid': os.getpid()})
    monitor = ResourceMonitor(run)
    monitor.snapshot('process_before_model')
    integrity = json.loads((MODEL / 'download_integrity.json').read_text())
    assert integrity['revision'] == '74dd0bf867a4cda7950c17663794267c60cf4b40'
    for asset in integrity['files']:
        assert digest(MODEL / asset['file']) == asset['sha256']
    assert digest(MANIFEST) == '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    manifest = json.loads(MANIFEST.read_text())
    data = {}
    for task in manifest['tasks']:
        for split, info in task['files'].items():
            path = ROOT / info['path']
            assert digest(path) == info['sha256']
            data[task['task'], split] = [json.loads(s) for s in path.read_text().splitlines()]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
                    torch_dtype=torch.float16, attn_implementation='sdpa').cuda()
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    for p in model.parameters():
        p.requires_grad = False
    layers = {}
    for key, base in list(model.named_modules()):
        if key.startswith('language_model.model.layers.') and key.rsplit('.', 1)[-1] in ['q_proj', 'v_proj']:
            parent, child = key.rsplit('.', 1)
            layer = SeqLinear(base)
            setattr(model.get_submodule(parent), child, layer)
            layers[key] = layer
    assert len(layers) == 48
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.language_model.enable_input_require_grads()
    model.eval()
    gates = []
    with torch.no_grad():
        rows = [(t, data[t['task'], 'train'][0]) for t in manifest['tasks']]
        # Original failure example is inference-only; it never enters an optimizer update.
        failure = next(x for x in data[16, 'test'] if x['id'] == 'FinMME:5082')
        rows.append((manifest['tasks'][15], failure))
        for task, row in rows:
            loss, err, count = loss_pair(row, task, processor, model, official=True)
            assert torch.isfinite(loss)
            gates.append({'id': row['id'], 'task': task['task'], 'official_nll_error': err, 'image_tokens': count})
            print(json.dumps({'event': 'official_forward_gate', **gates[-1]}), flush=True)
    model.train()
    monitor.snapshot('training_before', model=model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0)
    # Same-weight/RNG manipulation check: isolate numerical loss scaling from
    # any optimizer, learning-rate, dataset, or architecture changes.
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    scale_checks = []
    first = data[1, 'train'][0]
    for initial_scale in [1024, 1]:
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state_all(rng_cuda)
        optimizer.zero_grad(set_to_none=True)
        probe_scaler = torch.amp.GradScaler('cuda', init_scale=initial_scale)
        loss, _, _ = loss_pair(first, manifest['tasks'][0], processor, model)
        probe_scaler.scale(loss).backward()
        probe_scaler.unscale_(optimizer)
        raw = [p.grad.detach().cpu().clone() if p.grad is not None else None for p in params]
        finite = all(g is not None and bool(torch.isfinite(g).all()) for g in raw)
        torch.save(raw, run / ('scale_' + str(initial_scale) + '_raw_gradients.pt'))
        scale_checks.append({'initial_scale': initial_scale, 'loss': float(loss.detach()),
                             'all_gradients_finite': finite, 'optimizer_updates': 0})
        write(run / 'loss_scaling_control.json', scale_checks)
        del raw
    assert scale_checks[-1]['all_gradients_finite'], 'scale1 still nonfinite; no admission'
    assert scale_checks[0]['loss'] == scale_checks[1]['loss'], 'same-weight/RNG loss differs'
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state_all(rng_cuda)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler('cuda', init_scale=1)
    cases = []
    for task in manifest['tasks']:
        row = data[task['task'], 'train'][0]
        before = {k: l.B.weight.detach().cpu().clone() for k, l in layers.items()}
        optimizer.zero_grad(set_to_none=True)
        loss, _, count = loss_pair(row, task, processor, model)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        write(run / 'current_update_check.json', {'task': task['task'], 'loss': float(loss.detach()),
             'loss_finite': bool(torch.isfinite(loss)), 'gradient_norm_repr': str(float(norm)),
             'gradient_norm_finite': bool(torch.isfinite(norm)), 'scale': scaler.get_scale()})
        assert torch.isfinite(loss) and torch.isfinite(norm)
        assert optimizer.param_groups[0]['lr'] > 0
        scaler.step(optimizer)
        scaler.update()
        changed = sum(not torch.equal(before[k], l.B.weight.detach().cpu()) for k, l in layers.items())
        assert changed > 0
        cases.append({'task': task['task'], 'id': row['id'], 'loss': float(loss.detach()),
                      'preclip_gradient_norm': float(norm), 'actual_lr': 1e-4, 'changed_B_modules': changed,
                      'image_tokens': count})
        write(run / 'partial_cases.json', cases)
        print(json.dumps({'event': 'native_task_update_gate', **cases[-1]}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    monitor.snapshot('training_after', model=model)
    monitor.finish(model)
    write(run / 'gpu_preflight.json', {'passed': True, 'formal_training': False,
          'method': 'SeqFT-LoRA admission only; other author mechanisms still need separate gates',
          'model_revision': integrity['revision'], 'dtype': 'float16', 'task_ids': list(range(1, 20)),
          'official_forward_gates': gates, 'positive_update_cases': cases,
          'initial_loss_scale': 1, 'same_weight_RNG_loss_scaling_control': scale_checks,
          'source_sha256': digest(Path(__file__)), 'QA_unchanged': True,
          'preflight_uses_only_first_TRAIN_per_task': True, 'test_failure_case_inference_only': True})


if __name__ == '__main__':
    main()
