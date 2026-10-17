"""CPU preflight of every frozen matched-study image geometry and prompt.

Use the ACTUAL return_tensors='pt' processor, not a Python-int approximation.
Pack dummy one-channel features with the installed official model method: only
row counts are tested, never visual representations or GPU forward correctness.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import hashlib
import inspect
import json
import types
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers.models.llava_onevision.modeling_llava_onevision import image_size_to_num_patches

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016/matched_task_domain'
MODEL = ROOT / 'model/LLaVA-OneVision-Qwen2-0.5B-OV'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(processor, row, task):
    assert len(row['images']) == 1
    markers = row['question'].count('<image>')
    if markers:
        assert markers == len(row['images'])
    content = [] if markers else [{'type': 'image'}]
    text = task['instruction']
    if row.get('context'):
        text += '\nContext: ' + row['context']
    text += '\nQuestion: ' + row['question']
    content.append({'type': 'text', 'text': text})
    prompt = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                           add_generation_prompt=True, tokenize=False)
    assert prompt.count('<image>') == 1
    return prompt


def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    sources = {}

    def track(path, expected=None):
        value = sha(path)
        if expected is not None:
            assert value == expected, str(path)
        sources[str(path.relative_to(ROOT))] = value
        return path

    spec = json.loads(track(REPORT / 'CPU_PLAN_CHECK.json').read_text())['spec']
    for name, digest in spec['source_sha256'].items():
        track(ROOT / name, digest)
    manifest_path = track(ROOT / 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json',
                          spec['manifest_sha256'])
    manifest = json.loads(manifest_path.read_text())
    original = {}
    for task in manifest['tasks']:
        for split, f in task['files'].items():
            path = track(ROOT / f['path'], f['sha256'])
            data = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(data) == f['count']
            for row in data:
                assert row['id'] not in original
                original[row['id']] = dict(row, _task=task['task'], _split=split)
    for name in ('config.json', 'processor_config.json', 'preprocessor_config.json',
                 'tokenizer_config.json', 'special_tokens_map.json', 'tokenizer.json', 'chat_template.json'):
        path = MODEL / name
        if path.exists():
            track(path)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    assert config.vision_feature_select_strategy == processor.vision_feature_select_strategy == 'full'
    assert config.image_grid_pinpoints == processor.image_processor.image_grid_pinpoints
    assert processor.num_image_tokens == (config.vision_config.image_size // config.vision_config.patch_size)**2
    assert processor.image_processor.size == {'height': 384, 'width': 384}
    roles = {key: sorted({i for ids in spec[key].values() for i in ids}) for key in
             ('TRAIN_ids', 'anchor_probe_ids', 'plasticity_ids', 'final_SeqFT_full_anchor_ids')}
    selected = sorted({i for ids in roles.values() for i in ids})
    train_ids = set(roles['TRAIN_ids'])
    test_ids = {i for role, ids in roles.items() if role != 'TRAIN_ids' for i in ids}
    assert not train_ids & test_ids and all(original[i]['_split'] == 'train' for i in train_ids)
    assert all(original[i]['_split'] == 'test' for i in test_ids)
    control_id = 'FinMME:5082'
    assert control_id not in selected, 'Positive failure control must not silently enter the fixed study'
    groups = defaultdict(list)
    image_metadata = {}
    for identity in selected + [control_id]:
        row = original[identity]
        assert row['answer'] and row['answers']
        assert len(row['images']) == 1
        path = track(ROOT / row['images'][0])
        with Image.open(path) as image:
            w, h = image.size
        image_metadata[row['images'][0]] = {'height': h, 'width': w, 'sha256': sources[row['images'][0]]}
        groups[h, w].append(identity)

    geometries = []
    cases = []
    model_stub = types.SimpleNamespace(config=config)
    real_processor_calls = 0
    for (h, w), ids in sorted(groups.items()):
        representative = ids[0]
        row = original[representative]
        with Image.open(ROOT / row['images'][0]) as image:
            rgb = image.convert('RGB')
        prompt = render(processor, row, manifest['tasks'][row['_task'] - 1])
        batch = processor(text=prompt, images=[rgb], return_tensors='pt')
        real_processor_calls += 1
        assert batch['image_sizes'].tolist() == [[h, w]]
        assert tuple(batch['pixel_values'].shape[-2:]) == (384, 384)
        patches = image_size_to_num_patches(batch['image_sizes'][0], config.image_grid_pinpoints,
                                           config.vision_config.image_size)
        assert batch['pixel_values'].shape[1] == patches
        # Official packing row count depends on shapes/config, not feature values.
        features, lengths = LlavaOnevisionForConditionalGeneration.pack_image_features(
            model_stub, [torch.zeros(patches, processor.num_image_tokens, 1)], batch['image_sizes'],
            torch.zeros(1), config.vision_aspect_ratio)
        packed = features.shape[0]
        assert lengths.tolist() == [packed]
        actual_tokens = int((batch['input_ids'] == config.image_token_index).sum())
        integer_formula = int(processor._get_number_of_features(h, w, 384, 384))
        tensor_formula = int(processor._get_number_of_features(batch['image_sizes'][0, 0],
                                                              batch['image_sizes'][0, 1], 384, 384))
        assert actual_tokens == tensor_formula
        for identity in ids:
            original_row = original[identity]
            full_prompt = render(processor, original_row, manifest['tasks'][original_row['_task'] - 1])
            # Re-run the official token expansion/tokenizer on EACH actual prompt.
            expanded = processor._expand_image_tokens([full_prompt], iter(batch['image_sizes']),
                                                       384, 384, processor.image_token)
            tokens = processor.tokenizer(expanded, return_tensors='pt')['input_ids']
            assert int((tokens == config.image_token_index).sum()) == actual_tokens
            cases.append({'id': identity, 'roles': [role for role, group in roles.items() if identity in group],
                          'height': h, 'width': w, 'processor_tokens': actual_tokens, 'packed_rows': packed,
                          'compatible': actual_tokens == packed, 'positive_failure_control': identity == control_id})
        geometries.append({'height': h, 'width': w, 'representative_id': representative,
                           'QA': len(ids), 'processor_tokens': actual_tokens, 'packed_rows': packed,
                           'python_integer_formula_tokens': integer_formula,
                           'actual_tensor_formula_tokens': tensor_formula})
    fixed = [c for c in cases if not c['positive_failure_control']]
    control = next(c for c in cases if c['positive_failure_control'])
    assert len(fixed) == len(selected) and len({c['id'] for c in fixed}) == len(fixed)
    assert not control['compatible'] and control['processor_tokens'] == 2052 and control['packed_rows'] == 2106
    positive_geometry = next(g for g in geometries if g['representative_id'] == control_id)
    assert positive_geometry['python_integer_formula_tokens'] == 2106
    assert all(c['compatible'] for c in fixed), 'Known structural input mismatch inside fixed cohort'
    for name, digest in sources.items():
        assert sha(ROOT / name) == digest
    assert not torch.cuda.is_initialized()
    result = {'CPU_preflight_passed': True, 'GPU_used': False, 'new_model_inference_or_training': 0,
              'fixed_unique_QA': len(selected), 'fixed_TRAIN_QA': len(train_ids), 'fixed_TEST_QA': len(test_ids),
              'role_QA_counts': {role: len(ids) for role, ids in roles.items()},
              'unique_geometries_including_failure_control': len(geometries),
              'actual_full_CPU_processor_calls': real_processor_calls,
              'per_prompt_expansion_and_tokenization_checks': len(cases),
              'fixed_cohort_mismatch_count': sum(not c['compatible'] for c in fixed),
              'positive_failure_control': control,
              'superseded_initial_check': {'valid': False,
                  'method': 'Python-int _get_number_of_features compared with official dummy pack; zero mismatches including known failed case.',
                  'why_invalid': 'Actual return_tensors=pt processor expands using tensor scalar arithmetic; Python-int path rounds differently for243x441. It missed the positive control.',
                  'reproduced': positive_geometry,
                  'correction': 'Use actual CPU processor return_tensors=pt, verify every real prompt and all geometry groups; retain both integer/tensor calculations.'},
              'cases': cases, 'geometries': geometries, 'image_metadata': image_metadata, 'sources': sources,
              'installed_transformers_methods_sha256': {
                  name: hashlib.sha256(inspect.getsource(method).encode()).hexdigest() for name, method in {
                      'pack_image_features': LlavaOnevisionForConditionalGeneration.pack_image_features,
                      'processor_call': type(processor).__call__,
                      'expand_image_tokens': type(processor)._expand_image_tokens,
                      'get_number_of_features': type(processor)._get_number_of_features}.items()},
              'auditor_sha256': sha(Path(__file__)),
              'limits': ['CPU image geometry/token compatibility only; no vision encoder, model weights, GPU forward/backward or optimizer tested.',
                         'One actual RGB processor call per H/W geometry plus official expansion/tokenization for every real prompt. Feature packing uses dummy width1 representations only.',
                         'Same shapes can still have decoding/content/visual-ability issues. Actual per-case finite updates/exact stage1 gates remain mandatory.',
                         'No QA replacement/filtering, image resize, dependency/source update, new method, score or model improvement.']}
    (REPORT / 'INPUT_GEOMETRY_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 固定核心对照的图像输入结构预检', '',
             f"冻结{len(selected)}条不同QA：TRAIN {len(train_ids)}，TEST {len(test_ids)}。训练8题/任务、锚32题、当前任务8题和SeqFT最终完整735锚题的并集全部覆盖；不把重复角色计为新样本。", '',
             f"实际CPU processor按{len(geometries)}种图像尺寸各执行一次原RGB输入；每条原始问题/提示再执行官方图像占位符扩展与tokenization。官方pack_image_features使用假定宽度1特征只检查输出行数；无模型加载/视觉推理/GPU。固定队列结构不匹配数 **0/{len(selected)}**。", '',
             '## 检查是否真的能发现原故障', '',
             '把不属于本次固定队列的FinMME:5082作为已知失败正对照，实测占位符2052、官方打包2106行，准确复现既有故障。它不进入训练/测试队列，不替换任何题。', '',
             '初次简化检查直接传Python整数尺寸，会算出2106并漏掉该正对照；该初检无效，已作废重做。实际return_tensors=pt的processor使用tensor标量运算，在243×441边界的浮点舍入不同。JSON保留两条路径的数值，避免把“检查通过”当作无条件正确。', '',
             '## 可以排除什么，仍待验证什么', '',
             '- 在本次冻结879条QA范围，未发现已知的占位符/视觉打包行数结构故障；不需要修改原源文件、缩放图像或挑掉样本。',
             '- 不证明图像编码器/梯度/优化器正确，也不说明低分由知识或多域干扰造成。每条实际GPU轨迹仍须有限更新及共同stage1参数/输出精确一致门槛。',
             '- 未新增模型性能或CL方法；这是防止无效实验的准入证据，不是任务/领域根因结论。当前GPU任务与基线不受影响。', '']
    (REPORT / 'INPUT_GEOMETRY_AUDIT.md').write_text('\n'.join(lines).rstrip() + '\n')
    print(json.dumps({'passed': True, 'fixed_QA': len(fixed), 'role_counts': result['role_QA_counts'],
                      'geometries': len(geometries), 'positive_control': control,
                      'GPU_initialized': torch.cuda.is_initialized()}))


if __name__ == '__main__':
    main()
