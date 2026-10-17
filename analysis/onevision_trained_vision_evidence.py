"""CPU retrospective audit of completed visual interventions; no new inference.

Answer F1 is unchanged. Checkpoints/cohorts/views were frozen by the original
probe. Intervals condition on those fitted weights, not on training randomness.
"""
import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from answer_f1 import VERSION, answer_f1

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016'
STATES = ('base', 'generic_natural', 'medical_isolated', 'sequential_final')
DOMAINS = ('natural', 'medical', 'geography')
VIEWS = ('image', 'text', 'wrong')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def check_row(row, original, view, donor, generate):
    assert row['id'] == original['id']
    assert row['references'] == original['answers']
    assert row['native_task'] == original['_task']
    assert row['view'] == view
    assert math.isfinite(row['nll']) and row['nll'] >= 0
    assert row['visual_tokens'] == 0 if view == 'text' else row['visual_tokens'] > 0
    assert ('prediction' in row) == generate
    if view == 'wrong':
        assert row['wrong_image_source_id'] == donor
    else:
        assert 'wrong_image_source_id' not in row
    if generate:
        assert answer_f1(row['prediction'], row['references']) == row['scores']['token_f1']


def paired(values, image_groups):
    assert values and len(values) == len(image_groups)
    groups = defaultdict(list)
    for value, group in zip(values, image_groups):
        assert math.isfinite(value)
        groups[group].append(value)
    clusters = list(groups.values())
    rng = random.Random(42)
    samples = []
    for _ in range(2000):
        selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        samples.append(sum(map(sum, selected)) / sum(map(len, selected)))
    samples.sort()
    return {'QA': len(values), 'recipient_image_clusters': len(clusters),
            'mean': sum(values) / len(values),
            'recipient_image_cluster_95CI': [samples[49], samples[1949]],
            'positive_QA': sum(v > 0 for v in values),
            'negative_QA': sum(v < 0 for v in values),
            'zero_QA': sum(v == 0 for v in values)}


def main():
    sources = {}

    def track(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        sources[str(path.relative_to(ROOT))] = digest
        return path

    def read(path, expected=None):
        return json.loads(track(path, expected).read_text())

    lock = read(ROOT / 'summary/primary_metric_20261017/metric_lock.json')
    track(ROOT / 'analysis/answer_f1.py', lock['metric_sha256'])
    assert lock['version'] == VERSION and not lock['option_mapping']
    proof = read(ROOT / 'summary/onevision_mechanisms_20261016/trained_vision_complete.json')
    attempt = read(ROOT / 'summary/onevision_mechanisms_20261016/trained_vision_attempt.json')
    assert proof['complete'] and proof['optimizer_updates'] == 0 and not proof['formal_full_test']
    assert attempt['run'] == proof['run']
    run = ROOT / proof['run']
    spec = read(run / 'protocol.json')
    assert spec == attempt['spec'] and spec['source_sha256'] == proof['source_sha256']
    assert spec['states'] == list(STATES) and spec['views'] == list(VIEWS)
    for name, digest in spec['source_sha256'].items():
        track(ROOT / name, digest)
    reference = read(REPORT / 'research_current_attempt.json')
    paths = {'generic_natural': ROOT / reference['old_control_run'] / 'adapters/isolated_natural.pt',
             'medical_isolated': ROOT / reference['old_control_run'] / 'adapters/isolated_medical.pt',
             'sequential_final': ROOT / reference['run'] / 'durable_states/stage_19.pt'}
    for name, path in paths.items():
        track(path, spec['checkpoint_sha256'][name])
    manifest = read(ROOT / 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json',
                    '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777')
    original = {}
    for task in manifest['tasks']:
        f = task['files']['test']
        path = track(ROOT / f['path'], f['sha256'])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == f['count']
        for row in rows:
            assert row['id'] not in original
            original[row['id']] = dict(row, _task=task['task'], _domain=task['domain'])
    raw = read(ROOT / 'data/SLAKE/raw/test.json')
    medical_raw = {'SLAKE:' + str(row['qid']): row for row in raw}
    assert len(medical_raw) == len(raw)
    donors = {}
    for domain, ids in spec['test_ids'].items():
        assert len(ids) == len(set(ids)) == 100
        assert all(original[i]['_domain'] == domain for i in ids)
        for index, identity in enumerate(ids):
            donors[identity] = next(ids[(index + shift) % len(ids)] for shift in range(1, len(ids))
                                    if original[ids[(index + shift) % len(ids)]]['images'] != original[identity]['images'])
    cache = {}
    mutation_checks = []
    for state in STATES:
        folder = run / 'evaluations' / ('vision_' + state)
        durable = read(folder / 'durable_complete.json')
        assert durable['source_sha256'] == spec['source_sha256']
        summary = read(folder / 'summary.json')
        assert summary == proof['summaries'][state]
        for domain in DOMAINS:
            for view in VIEWS:
                path = track(folder / f'{domain}_{view}.jsonl', durable['files'][f'{domain}_{view}.jsonl'])
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                assert [row['id'] for row in rows] == spec['test_ids'][domain]
                assert len({row['id'] for row in rows}) == len(rows)
                for row in rows:
                    check_row(row, original[row['id']], view, donors[row['id']],
                              row['id'] in spec['generation_ids'][domain])
                generated = [row for row in rows if 'prediction' in row]
                assert set(row['id'] for row in generated) == set(spec['generation_ids'][domain])
                s = summary[f'{domain}:{view}']
                assert s['count'] == len(rows) and s['generated_count'] == len(generated)
                assert abs(s['mean_nll'] - sum(row['nll'] for row in rows) / len(rows)) < 1e-12
                assert abs(s['token_f1'] - sum(row['scores']['token_f1'] for row in generated) / len(generated)) < 1e-12
                cache[state, domain, view] = rows
    # Reject plausible corruption of an actual completed row, not synthetic scores.
    valid = cache['sequential_final', 'medical', 'wrong'][0]
    for name, edit in [('ID', lambda r: r.update(id='not-the-frozen-ID')),
                       ('gold', lambda r: r.update(references=['altered gold'])),
                       ('donor', lambda r: r.update(wrong_image_source_id=r['id'])),
                       ('score', lambda r: r['scores'].update(token_f1=r['scores']['token_f1'] + 1))]:
        bad = copy.deepcopy(valid)
        edit(bad)
        try:
            check_row(bad, original[valid['id']], 'wrong', donors[valid['id']], True)
        except AssertionError:
            mutation_checks.append(name + ':rejected')
        else:
            raise AssertionError('Mutation accepted: ' + name)

    # Check the actual image bytes/pixels, not merely distinct file names.
    images = {}
    for identity in donors:
        row = original[identity]
        assert len(row['images']) == 1
        name = row['images'][0]
        if name not in images:
            path = track(ROOT / name)
            with Image.open(path) as source:
                rgb = source.convert('RGB')
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + b'\0' + rgb.tobytes()).hexdigest()
            images[name] = {'file_sha256': sources[name], 'RGB_pixel_sha256': pixel_hash, 'size': list(rgb.size)}
    controls = {}
    for domain in DOMAINS:
        ids = spec['test_ids'][domain]
        if domain == 'medical':
            for identity in ids:
                source = medical_raw[identity]
                assert source['question'] == original[identity]['question']
                assert source['answer'] in original[identity]['answers']
                assert source['img_id'] == original[identity]['metadata']['img_id']
        profile = {'QA': len(ids), 'identical_file_bytes': 0, 'identical_decoded_RGB': 0,
                   'different_original_size': 0}
        for identity in ids:
            a = images[original[identity]['images'][0]]
            b = images[original[donors[identity]]['images'][0]]
            profile['identical_file_bytes'] += a['file_sha256'] == b['file_sha256']
            profile['identical_decoded_RGB'] += a['RGB_pixel_sha256'] == b['RGB_pixel_sha256']
            profile['different_original_size'] += a['size'] != b['size']
        image_rows = cache['base', domain, 'image']
        wrong_rows = cache['base', domain, 'wrong']
        profile['different_visual_token_count'] = sum(a['visual_tokens'] != b['visual_tokens'] for a, b in zip(image_rows, wrong_rows))
        if domain == 'medical':
            profile['different_author_modality'] = sum(medical_raw[i]['modality'] != medical_raw[donors[i]]['modality'] for i in ids)
            profile['same_author_modality'] = len(ids) - profile['different_author_modality']
        controls[domain] = profile

    cells = []
    for state in STATES:
        for domain in DOMAINS:
            views = {view: cache[state, domain, view] for view in VIEWS}
            cell = {'state': state, 'domain': domain, 'NLL_QA': 100,
                    'generated_QA': len(spec['generation_ids'][domain]),
                    'Answer_F1': {}, 'mean_NLL': {}, 'comparisons': {}}
            for view, rows in views.items():
                cell['mean_NLL'][view] = sum(row['nll'] for row in rows) / len(rows)
                scores = [row['scores']['token_f1'] for row in rows if 'prediction' in row]
                cell['Answer_F1'][view] = sum(scores) / len(scores)
            for control in ('wrong', 'text'):
                pairs = list(zip(views['image'], views[control]))
                assert all(a['target_tokens'] == b['target_tokens'] and
                           a['target_truncated'] == b['target_truncated'] for a, b in pairs)
                gen = [(a, b) for a, b in pairs if 'prediction' in a]
                cell['comparisons'][control] = {
                    'F1_correct_minus_control_pp': paired(
                        [a['scores']['token_f1'] - b['scores']['token_f1'] for a, b in gen],
                        [tuple(original[a['id']]['images']) for a, _ in gen]),
                    'NLL_control_minus_correct': paired(
                        [b['nll'] - a['nll'] for a, b in pairs],
                        [tuple(original[a['id']]['images']) for a, _ in pairs])}
            cells.append(cell)
    # Retrospective partitions expose resolution/token confounding; retain both
    # groups for every state, without selecting a favorable subgroup or cutoff.
    token_partitions = []
    for state in STATES:
        pairs = list(zip(cache[state, 'medical', 'image'], cache[state, 'medical', 'wrong']))
        for equal in (True, False):
            group = [(a, b) for a, b in pairs if (a['visual_tokens'] == b['visual_tokens']) == equal]
            token_partitions.append({'state': state, 'same_visual_token_count': equal,
                                     'F1_correct_minus_wrong_pp': paired(
                                         [a['scores']['token_f1'] - b['scores']['token_f1'] for a, b in group],
                                         [tuple(original[a['id']]['images']) for a, _ in group]),
                                     'NLL_wrong_minus_correct': paired(
                                         [b['nll'] - a['nll'] for a, b in group],
                                         [tuple(original[a['id']]['images']) for a, _ in group])})
    subgroups = []
    for task in manifest['tasks']:
        if task['domain'] != 'medical':
            continue
        ids = [i for i in spec['test_ids']['medical'] if original[i]['_task'] == task['task']]
        for state in STATES:
            rows = {v: {r['id']: r for r in cache[state, 'medical', v]} for v in VIEWS}
            subgroups.append({'state': state, 'native_task': task['task'], 'native_label': task['native_label'],
                              'QA': len(ids), 'different_author_modality': sum(
                                  medical_raw[i]['modality'] != medical_raw[donors[i]]['modality'] for i in ids),
                              'Answer_F1': {v: sum(rows[v][i]['scores']['token_f1'] for i in ids) / len(ids) for v in VIEWS}})
    result = {'complete_CPU_audit': True, 'original_run': proof['run'], 'primary_metric': 'Answer F1',
              'metric_version': VERSION, 'GPU_updates_or_new_inference': 0,
              'generated_QA_per_state_view': 120, 'NLL_QA_per_state_view': 300,
              'prediction_files': 36, 'unique_generated_QA': 120, 'formal_full_test': False,
              'cells': cells, 'medical_native_subgroups_descriptive': subgroups,
              'medical_visual_token_partitions_posthoc': token_partitions,
              'wrong_image_controls': controls, 'image_audit': images,
              'actual_row_mutation_tests': mutation_checks,
              'sources': sources, 'auditor_sha256': sha(Path(__file__)),
              'limits': [
                  'Retrospective analysis of fixed original checkpoints/cohorts, not a new predeclared experiment or method.',
                  'Within-weight views change only inference image input; no optimizer, answer editing or new scoring.',
                  'Checkpoint states differ in training budgets/dropout streams. Cross-state changes do not isolate CL, knowledge or vision-training causality.',
                  'Paired recipient-image bootstrap:2000 resamples,seed42,uncorrected95CI; fixed weights/single training seed. Shared donor dependencies are not modeled.',
                  'Natural/geography have only10 generated QA each; medical100 is diagnostic, not full2094 or full21454.',
                  'Wrong-image donors differ in resolution/visual-token count. This ablates the image-input package, not matched-token semantic content alone.',
                  'Distinct RGB pixels and different modality do not prove all wrong images contradict the answer; same-modality images may also be wrong.',
                  'No image benefit cannot establish absent knowledge or a broken encoder. Positive image benefit does not establish sufficient vision or medical knowledge.',
                  'Answer F1 is lexical overlap. No option mapping, prefix stripping, semantic judgment or test-driven model selection.']}
    for name, digest in sources.items():
        assert sha(ROOT / name) == digest
    (REPORT / 'TRAINED_VISION_F1_EVIDENCE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 训练后的模型是否真正利用图像？', '',
             '只读已完成视觉消融，同一权重分别输入原图、固定错图、无图；不增加推理/训练，不更换评分。唯一性能指标Answer F1，NLL仅作机制辅助。四组权重和测试ID均由原协议固定，不按结果挑模型。', '',
             '## 医学100题：同一权重的配对对照', '',
             '| 权重 | 原图F1 | 错图F1 | 无图F1 | 原图−错图pp [95%区间] | 原图−无图pp [95%区间] |',
             '| --- | ---: | ---: | ---: | --- | --- |']
    for cell in cells:
        if cell['domain'] != 'medical':
            continue
        f = cell['Answer_F1']
        pairs = []
        for view in ('wrong', 'text'):
            p = cell['comparisons'][view]['F1_correct_minus_control_pp']
            ci = p['recipient_image_cluster_95CI']
            pairs.append(f"{p['mean']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
        lines.append(f"| {cell['state']} | {f['image']:.3f} | {f['wrong']:.3f} | {f['text']:.3f} | {pairs[0]} | {pairs[1]} |")
    lines += ['', '## 证明与证伪的边界', '',
              '- 排除：不能把基座医学生成中较弱的图像收益，推广成最终训练权重完全不利用图像。最终权重在固定医学100题的原图F1优于两种对照，区间与逐题方向见上表。',
              '- 支持：该组最终权重的输出质量依赖图像输入包（像素、分辨率、视觉token），低分不能全部归因为图像输入完全不起作用。这只是同权重输入干预，不证明视觉语义能力已经足够。',
              '- 未证明：顺序学习本身导致图像利用提升；四组权重训练预算/dropout不同，跨权重不作严格方法排名。也不能据此认定知识不足或干扰是唯一原因。', '',
              '## 错配图是否有效？', '',
              '| 域 | 对照QA | 相同原文件字节 | 相同解码RGB | 原图/错图尺寸不同 | visual token数不同 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for domain, p in controls.items():
        lines.append(f"| {domain} | {p['QA']} | {p['identical_file_bytes']} | {p['identical_decoded_RGB']} | {p['different_original_size']} | {p['different_visual_token_count']} |")
    p = controls['medical']
    lines += ['', f"医学错图{p['different_author_modality']}/100来自不同作者标注模态，{p['same_author_modality']}/100同模态。原生题型细项全部保留在JSON，不挑显著组。不同尺寸/token数仍是图像干预的一部分，不把本实验写成严格等token的纯语义消融。", '',
              '事后按原图/错图visual token数相同与不同作二分，保留全部四权重两组，不按测试选阈值或只报好组。最终权重的结果：', '',
              '| visual token数 | QA | 原图−错图F1 pp [95%区间] |',
              '| --- | ---: | --- |']
    for part in token_partitions:
        if part['state'] != 'sequential_final':
            continue
        p = part['F1_correct_minus_wrong_pp']
        ci = p['recipient_image_cluster_95CI']
        lines.append(f"| {'相同' if part['same_visual_token_count'] else '不同'} | {p['QA']} | {p['mean']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] |")
    lines += ['', '两组题目/图像构成不同，该二分不能估计token数的独立因果作用；相同token数也不等于相同分辨率/裁剪或视觉语义。', '',
              '## 范围、复现与下一步', '',
              '36个原预测文件、4个完成证明、3个实际adapter文件、原QA/金标/ID、固定循环错图规则和原F1逐项校验；篡改真实行的ID/金标/错图来源/F1四种情况均被拒绝。逐文件SHA和图像字节/RGB检查见JSON。', '',
              '每组权重300题NLL、120题生成（医学100、自然10、遥感10），四权重三视图的重复测量不是1440个独立QA。不是完整测试或正式双骨干V2主表。配对区间按收图端原图聚类，固定权重、未校正多重比较，未建模重复错图来源依赖。', '',
              '改进的是诊断结论：补齐训练后视觉证据，撤去“基座弱收益等于训练后不会看图”的推断；无新增方法或性能提升。下一步仍按原顺序完成已有完整确认，再做冻结的48组任务/领域控制，不启动新算法。', '']
    (REPORT / 'TRAINED_VISION_F1_EVIDENCE.md').write_text('\n'.join(lines).rstrip() + '\n')
    print(json.dumps({'medical': [c for c in cells if c['domain'] == 'medical'],
                      'wrong_controls': controls, 'sources': len(sources), 'mutation_checks': mutation_checks}, ensure_ascii=False))


if __name__ == '__main__':
    main()
