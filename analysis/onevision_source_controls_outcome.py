"""CPU-only synthesis of sealed source controls, resources and all pairwise effects.

Run the original score and TRAIN-geometry auditors on a new sealed proof first.
No model inference, optimizer, experiment selection or background daemon.
"""
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    scores = read(REPORT/'VERIFIED_EVIDENCE.json')
    geometry = read(REPORT/'TRAIN_PARAMETER_GEOMETRY.json')
    proofs = {p.name: sha(p) for p in REPORT.glob('*_complete.json')}
    assert proofs == scores['proof_sha256'] == geometry['proof_sha256'], 'Audit newly sealed proofs first'
    assert geometry['analyzer_sha256'] == sha(ROOT/'analysis/onevision_initialization_source_geometry.py')
    assert scores['report_source_sha256'] == sha(ROOT/'analysis/onevision_initialization_source_report.py')
    destination = REPORT/'CONTROL_OUTCOME.json'
    if destination.exists():
        previous = read(destination)
        if previous['proof_sha256'] == proofs and previous['analyzer_sha256'] == sha(Path(__file__)):
            return
    spec = read(REPORT/'CPU_PLAN_CHECK.json')['spec']
    manifest_path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest_path) == spec['manifest_sha256']
    info = read(manifest_path)['tasks'][3]['files']['test'];gold_path = ROOT/info['path']
    assert sha(gold_path) == info['sha256']
    gold = [json.loads(line) for line in gold_path.read_text().splitlines()]
    ids = [row['id'] for row in gold];assert len(ids) == len(set(ids)) == 507
    conditions = {};values = {}
    resources = dict(unit='bytes',CPU_only_verbatim_aggregation=True,conditions={},
        analyzer_sha256=sha(Path(__file__)),limits=[
            'Before/after within same PID; independent PID occupancy differences are not method memory growth.',
            'Pure TRAIN includes gradient/checkpoint persistence, excludes TEST/admission; initial32 already evaluated.',
            'CUDA peak counters reset; CPU endpoints do not establish pure TRAIN CPU peak.',
            'Warm source12 exposures/2updates charged separately; no comparative memory superiority inferred.'])
    for name in spec['cases']:
        pointer = REPORT/(name+'_complete.json')
        if not pointer.exists():
            continue
        proof = read(pointer);run = ROOT/proof['run'];assert proof['complete'] and proof['spec'] == spec
        assert run.resolve().is_relative_to((ROOT/'exp/onevision_initialization_source_20261018/v1').resolve())
        def bound(rel):
            p = run/rel;assert p.resolve().is_relative_to(run.resolve())
            assert sha(p) == proof['artifact_sha256'][rel]
            return read(p)
        data_path = run/'evaluations/target_full507/medical_image.jsonl'
        assert sha(data_path) == proof['artifact_sha256'][str(data_path.relative_to(run))]
        rows = [json.loads(line) for line in data_path.read_text().splitlines()]
        assert [r['id'] for r in rows] == ids
        assert all(a['references'] == b['answers'] for a,b in zip(rows,gold))
        values[name] = [r['scores']['token_f1'] for r in rows]
        record = next(row for row in scores['rows'] if row['condition'] == name)
        assert sum(values[name])/507 == record['target_full507']['F1']
        conditions[name] = dict(scores=record,parameters=geometry['targets'][name])
        payload = dict(whole_control=bound('memory_summary.json'),pure_TRAIN=bound('pure_TRAIN_windows.json'),
            whole_control_sha256=sha(run/'memory_summary.json'),pure_TRAIN_sha256=sha(run/'pure_TRAIN_windows.json'),
            run=proof['run'],completion_sha256=sha(pointer))
        for window in payload['pure_TRAIN']['windows']:
            assert window['before']['pid'] == window['after']['pid'] == payload['whole_control']['pid']
        resources['conditions'][name] = payload
    contrasts = []
    # Every available pair, including negative effects; no winner or source selection.
    for reference,target in itertools.combinations([n for n in spec['cases'] if n in values],2):
        blocks = defaultdict(list)
        for row,a,b in zip(gold,values[reference],values[target]):
            blocks[tuple(row['images'])].append(b-a)
        groups = list(blocks.values());rng = random.Random(42);draws = []
        for _ in range(10000):
            sample = [groups[rng.randrange(len(groups))] for _ in groups]
            draws.append(sum(map(sum,sample))/sum(map(len,sample)))
        draws.sort();contrasts.append(dict(target=target,reference=reference,
            delta_F1=sum(map(sum,groups))/507,image_95CI=[draws[249],draws[9749]],uncorrected=True))
    for contrast in contrasts:
        if contrast['reference'] == 'cold':
            original = next(c for c in scores['contrasts'] if c['condition'] == contrast['target'])
            assert contrast['delta_F1'] == original['delta_F1']
            assert contrast['image_95CI'] == original['image_95CI']
    result = dict(CPU_only=True,new_predictions=0,new_optimizer_updates=0,analyzer_sha256=sha(Path(__file__)),
        proof_sha256=proofs,score_audit_sha256=sha(REPORT/'VERIFIED_EVIDENCE.json'),
        TRAIN_parameter_audit_sha256=sha(REPORT/'TRAIN_PARAMETER_GEOMETRY.json'),conditions=conditions,
        all_available_pairwise_contrasts=contrasts,missing_conditions=[n for n in spec['cases'] if n not in conditions],
        no_TEST_selection=True,limits=[
            'One target, two source tasks, one seed; source content/type differ. Not universal domain causality.',
            'All target26QA/2updates same; warm conditions additionally use source6QA repeated twice/2updates.',
            'Full vs A-only changes old output and first-step A-gradient access; not pure knowledge separation.',
            'Source A change is small; A-only negatives cannot universally reject basis reuse.',
            'Initial32 is not full507 initial score or validation; do not subtract their means.',
            'Uncorrected paired image bootstrap describes fixed fitted models, not multi-seed uncertainty.',
            'Independent single-task507 does not establish five-domain CL AP/FWT/BWT or method success.'])
    destination.write_text(json.dumps(result,indent=2)+'\n')
    (REPORT/'RESOURCE_SNAPSHOTS.json').write_text(json.dumps(resources,indent=2)+'\n')
    text = ['# 来源控制：旧输出与低秩方向', '',
        '**探究的问题：** 少数据下，应该继承同域或跨域的旧输出，还是只保留其低秩方向？', '',
        '**做的实验：** 固定Organ原26TRAIN／2更新，五种初始化，每组最终原507题；源各6QA重复两轮／2更新。初始32仅记录函数，不能与最终507均值相减。', '',
        f"**结果：** 已封存{len(conditions)}/5组，未完成留空，不根据分数取消条件。", '',
        '| 条件 | 最终507 Answer F1 | 最终507 NLL | 首步A梯度范数 | 第二步A梯度范数 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in spec['cases']:
        c = conditions.get(name)
        if c:
            final = c['scores']['target_full507'];g = c['parameters']['raw_preclip_gradients']
            text.append(f"| {name} | {final['F1']:.3f} | {final['NLL']:.4f} | {g[0]['preclip_A_gradient_l2']:.3f} | {g[1]['preclip_A_gradient_l2']:.3f} |")
        else:
            text.append('| '+name+' | — | — | — | — |')
    text += ['', '| 全部可用配对对比（前者−后者） | F1差 | 图像95%区间 |', '| --- | ---: | --- |']
    for c in contrasts:
        lo,hi = c['image_95CI']
        text.append(f"| {c['target']} − {c['reference']} | {c['delta_F1']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    text += ['', '**证明了什么：** 当前表只支持相应固定源／目标的局部差异；参数与梯度证明干预是否启用，不直接证明差距的唯一原因。初始函数、源历史与首步优化通道未完全分开。', '',
        '**改进的方法：** 完成所有五格后从结果提出最简候选；现阶段不新增模块、路由或测试驱动超参选择。', '',
        '**改进的效果：** 未完成组尚待；该单任务控制不是新的五域持续学习算法，也不填正式主表。', '',
        '[原始评分核验](VERIFIED_RESULTS.md) · [全部配对／参数／SHA](CONTROL_OUTCOME.json) · [内存及TRAIN窗口](RESOURCE_SNAPSHOTS.md)']
    (REPORT/'CONTROL_OUTCOME.md').write_text('\n'.join(text)+'\n')
    lines = ['# 初始化来源控制：真实内存与TRAIN窗口', '',
        '单位MiB；原始bytes及RSS/PSS/USS/allocated/reserved/NVML见JSON。每行同PID目标TRAIN窗口，含梯度／checkpoint持久化，不含TEST／模型准入，初始32题已在TRAIN前评测。', '',
        '| 条件 | PID | CPU RSS 前→后 | CPU USS 前→后 | GPU allocated 前→后 | GPU TRAIN allocated峰值 | TRAIN窗口秒 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,a in resources['conditions'].items():
        w = a['pure_TRAIN']['windows'][0];before,after = w['before'],w['after'];unit = 2**20
        numbers = [f"{before[k]/unit:.3f} → {after[k]/unit:.3f}" for k in ['VmRSS_bytes','cpu_uss_bytes','gpu_allocated_bytes']]
        lines.append('| '+name+f" | {before['pid']} | "+' | '.join(numbers)+f" | {w['cuda_peak_allocated_bytes']/unit:.3f} | {after['unix']-before['unix']:.3f} |")
    lines += ['', '每格540672个可训练LoRA参数；参数字节不等于进程内存。温组另计源12QA曝光／2更新。跨PID的预热／分配器历史不同，不将端点差称算法内存优势。CPU纯TRAIN峰值未报告；全控制采样峰值／HWM另存，不能偷换边界。']
    (REPORT/'RESOURCE_SNAPSHOTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(accepted=list(conditions),pending=result['missing_conditions'],contrasts=contrasts)))


if __name__ == '__main__':
    main()
