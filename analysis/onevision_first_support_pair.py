"""First predeclared one-QA initialization pair, all fixed budget endpoints."""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from answer_f1 import answer_f1

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'summary/onevision_diagnostics_20261016/cold_data_factorial_20261018'
REPORT = STUDY/'first_support_pair'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def quantile(values, p):
    values = sorted(values);x = p*(len(values)-1);a = int(x);b = min(a+1, len(values)-1)
    return values[a]+(values[b]-values[a])*(x-a)


def main():
    assert not (REPORT/'EVIDENCE.json').exists(), 'Do not repeat unchanged completed evidence'
    inputs = {}
    def read(p):
        inputs[str(p.relative_to(ROOT))] = sha(p)
        return json.loads(p.read_text())
    plan = read(STUDY/'CPU_PLAN_CHECK.json')['spec']
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT/name) == expected
    manifest = read(ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json')
    task = next(t for t in manifest['tasks'] if t['task'] == 2)
    testfile = ROOT/task['files']['test']['path'];assert sha(testfile) == task['files']['test']['sha256']
    gold = {r['id']: r for r in map(json.loads, testfile.read_text().splitlines())}
    trainfile = ROOT/task['files']['train']['path'];assert sha(trainfile) == task['files']['train']['sha256']
    train = {r['id']: r for r in map(json.loads, trainfile.read_text().splitlines())}
    inputs[str(testfile.relative_to(ROOT))] = sha(testfile)
    inputs[str(trainfile.relative_to(ROOT))] = sha(trainfile)
    scores, statistics, cases = {}, {}, []
    for index, start in [(0, 'cold_base'), (1, 'warm_native_T01')]:
        proof = read(STUDY/f'case_{index:02d}_complete.json')
        assert proof['complete'] and proof['valid_training'] and proof['spec'] == plan
        case = proof['case'];assert case == plan['cases'][index]
        assert case['task'] == 2 and case['start'] == start and case['unique_TRAIN'] == 1
        assert case['effective_batch'] == 44 and len(set(case['exposure_ids_per_update'])) == 1
        cases.append(case)
        run = ROOT/proof['run']
        for rel, expected in proof['artifact_sha256'].items():
            p = run/rel;assert p.resolve().is_relative_to(run.resolve()) and sha(p) == expected
        assert read(run/'initial_state_gate.json')['passed']
        events = read(run/'training.json')['records']
        assert [e['update'] for e in events] == [1, 2, 3, 4]
        assert all(e['LR'] == 1e-4 and math.isfinite(e['loss']) and e['gradient_L2_preclip'] > 0 for e in events)
        scores[start], statistics[start] = {}, {}
        for label in ['initial', 'endpoint_1', 'endpoint_4']:
            path = run/f'evaluations/{label}/native_image.jsonl'
            inputs[str(path.relative_to(ROOT))] = sha(path)
            rows = list(map(json.loads, path.read_text().splitlines()))
            assert [r['id'] for r in rows] == case['TEST_ids'] and len(rows) == 228
            values = {}
            for row in rows:
                assert row['references'] == gold[row['id']]['answers']
                value = answer_f1(row['prediction'], row['references'])
                assert value == row['scores']['token_f1']
                values[row['id']] = value
            scores[start][label] = values
            histogram = Counter(r['prediction'] for r in rows)
            statistics[start][label] = {'Answer_F1': sum(values.values())/len(values),
                'distinct_raw_predictions': len(histogram), 'largest_identical_prediction_count': max(histogram.values()),
                'largest_identical_prediction_fraction': max(histogram.values())/len(rows)}
    assert cases[0]['TRAIN_ids'] == cases[1]['TRAIN_ids'] and cases[0]['TEST_ids'] == cases[1]['TEST_ids']
    ids = cases[0]['TEST_ids'];contrasts = {}
    for label in ['initial', 'endpoint_1', 'endpoint_4']:
        contrasts['warm_minus_cold_'+label] = {i: scores['warm_native_T01'][label][i]-scores['cold_base'][label][i] for i in ids}
    contrasts['change_in_warm_advantage_4_minus_1'] = {i: contrasts['warm_minus_cold_endpoint_4'][i]-contrasts['warm_minus_cold_endpoint_1'][i] for i in ids}
    for start in scores:
        contrasts[start+'_4_minus_1'] = {i: scores[start]['endpoint_4'][i]-scores[start]['endpoint_1'][i] for i in ids}
    groups = defaultdict(list)
    for i in ids:
        groups[tuple(gold[i]['images'])].append(i)
    blocks = list(groups.values());rng = random.Random(42);boot = {name: [] for name in contrasts}
    for _ in range(10000):
        drawn = [i for _ in blocks for i in blocks[rng.randrange(len(blocks))]]
        for name, values in contrasts.items():
            boot[name].append(sum(values[i] for i in drawn)/len(drawn))
    estimates = {name: {'mean_pp': sum(values.values())/len(values),
        'image_cluster_CI95': [quantile(boot[name], .025), quantile(boot[name], .975)]} for name, values in contrasts.items()}
    REPORT.mkdir(parents=True, exist_ok=True)
    out = {'CPU_only': True, 'new_model_or_optimizer_operations': 0,
        'predeclared_cases': [0, 1], 'TRAIN_QA': 1, 'TRAIN_id': cases[0]['TRAIN_ids'][0],
        'TRAIN_answer_for_description_only': train[cases[0]['TRAIN_ids'][0]]['answer'],
        'TEST_QA': 228, 'image_clusters': len(blocks), 'statistics': statistics, 'contrasts': estimates,
        'extra_exposures_4_vs_1': 132, 'main1313_protocol_changed': False,
        'all18_required': True, 'TEST_chosen_hyperparameters_or_methods': False,
        'input_sha256': inputs, 'source_sha256': sha(Path(__file__)),
        'limits': ['One nested TRAIN QA/one task/one seed, all remaining supports and tasks pending.',
                   'Update count, repeated exposure and Adam state change together; not isolated update-only causality.',
                   'Warm inherits347 natural TRAIN history; initial functions and optimization coordinates differ.',
                   'Image bootstrap is fixed-weight uncorrected uncertainty, not training-seed variability.',
                   'Raw-output repetition counts do not prove knowledge gain or diagnose a unique mechanism.']}
    (REPORT/'EVIDENCE.json').write_text(json.dumps(out, indent=2)+'\n')
    text = '# 第一组固定单QA：初始化优势随更新／曝光预算改变\n\n'
    text += '- **探究的问题：** 冷启动劣势是否在增加预算后仍同样存在？\n'
    text += '- **做的实验：** 原预定case00/01、相同1TRAIN/44次曝光每步、同228原TEST，冷基座与原生T01暖起点；保留同一Adam轨迹的1/4步，不改变正式1313训练协议。\n'
    text += '- **结果：**\n\n| 起点 | 初始F1 | 1步F1 | 4步F1 |\n|---|---:|---:|---:|\n'
    for start, points in statistics.items():
        text += '| '+start+' | '+' | '.join(f"{points[k]['Answer_F1']:.3f}" for k in ['initial', 'endpoint_1', 'endpoint_4'])+' |\n'
    text += '\n| 对比 | 差值pp | 图像95%区间 |\n|---|---:|---|\n'
    for name, v in estimates.items():
        text += f"| {name} | {v['mean_pp']:.3f} | [{v['image_cluster_CI95'][0]:.3f}, {v['image_cluster_CI95'][1]:.3f}] |\n"
    text += '\n- **证明了什么：** 本例初始化优势依赖更新／曝光预算，不能将一步弱习得解释为永久的低能力上限；是否稳定反转需看区间。不是少数据交互定律、知识充分或全部方法冷启动原因。独特样本量仍固定1，未齐格不能推断1vs4vs全TRAIN规律。\n'
    text += '- **改进的方法：** 新候选保持连续习得并有界保存功能；仍保留所有既定预算和初始化对照，不根据本TEST结果增加主训练轮数或挑赢家。\n'
    text += '- **改进的效果：** 此处只有控制实验，未取得新候选F1；4步相对1步额外132次重复曝光，不能写成免费收益。原始输出重复度、全部逐题区间和源/数据SHA见[EVIDENCE](EVIDENCE.json)。\n'
    (REPORT/'RESEARCH_FINDINGS.md').write_text(text)
    print(json.dumps({'statistics': statistics, 'contrasts': estimates}, ensure_ascii=True))


if __name__ == '__main__':
    main()
