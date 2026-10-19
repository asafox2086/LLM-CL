"""Report and independent low-rank trace verification of saved CPU geometry."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'summary/two_backbone_pct01_20261017/partial_full_vs_pct01_20261019'
OUT = BASE/'parameter_geometry/v2'
WORK = ROOT/'exp/qwen_full_pct01_parameter_geometry_20261019/v2'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    torch.set_num_threads(2)
    e = json.loads((OUT/'EVIDENCE.json').read_text())
    scores = {x['cell']: x for x in json.loads((BASE/'EVIDENCE.json').read_text())['comparisons']}
    states = {(c, s): torch.load(WORK/f'{c}_S{s:02}.pt', map_location='cpu', weights_only=True)
              for c in ['full', 'pct01'] for s in range(1, 4)}
    def terms(c, s, mode, n):
        a = states[c, s][n]
        result = [(a['B'].double()*a['scale'], a['A'].double())]
        if mode == 'stage_increment' and s > 1:
            a = states[c, s-1][n]
            result.append((-a['B'].double()*a['scale'], a['A'].double()))
        return result
    def inner(f, p):
        # <BA,DC> = trace(B^T D C A^T); never forms the dense update.
        return sum(float(((b.T@d)*(a@c.T)).sum()) for b,a in f for d,c in p)
    max_relative_error = 0.
    checks = 0
    derived = []
    for r in e['comparisons']:
        stage, mode = r['stage'], r['mode']
        for m in r['modules']:
            f, p = terms('full',stage,mode,m['module']), terms('pct01',stage,mode,m['module'])
            for expected, actual in [(m['full_norm2'],inner(f,f)),
                                     (m['pct01_norm2'],inner(p,p)), (m['dot'],inner(f,p))]:
                err = abs(actual-expected)/max(abs(expected),1e-12)
                assert err < 1e-9, (stage,mode,m['module'],err)
                max_relative_error = max(max_relative_error,err)
                checks += 1
        if mode != 'stage_increment':
            continue
        d = {'stage': stage}
        for c in ['full','pct01']:
            k = c+'_norm2'
            total = sum(m[k] for m in r['modules'])
            d[c] = {
                'last7_of28_layers_energy_fraction': sum(m[k] for m in r['modules'] if m['layer'] >= 21)/total,
                'mean_per_module_rank90': sum(m[c+'_spectrum']['rank90'] for m in r['modules'])/56,
                'mean_per_module_stable_rank': sum(m[c+'_spectrum']['stable_rank'] for m in r['modules'])/56,
            }
            if stage > 1:
                for factor in ['A','B']:
                    changed = count = norm2 = oldnorm2 = 0
                    for n in states[c,stage]:
                        a, b = states[c,stage][n][factor], states[c,stage-1][n][factor]
                        changed += int((a != b).sum());count += a.numel()
                        norm2 += float((a.double()-b.double()).square().sum())
                        oldnorm2 += float(b.double().square().sum())
                    d[c][factor] = {'changed_coordinates':changed,'total_coordinates':count,
                                   'changed_fraction':changed/count,'change_l2':math.sqrt(norm2),
                                   'relative_l2':math.sqrt(norm2/oldnorm2)}
        derived.append(d)
    proof = {'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'geometry_evidence_sha256':hashlib.sha256((OUT/'EVIDENCE.json').read_bytes()).hexdigest(),
             'independent_gram_trace_checks':checks,'max_relative_error':max_relative_error,
             'derived':derived,'cuda_initialized':torch.cuda.is_initialized(),
             'new_model_forwards':0,'new_optimizer_steps':0,
             'checkpoint_mutations':False,'metric_or_data_changes':False}
    assert not proof['cuda_initialized']
    (OUT/'INDEPENDENT_CHECK.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    increments = {r['stage']:r for r in e['comparisons'] if r['mode']=='stage_increment'}
    labels = {1:'自然问答',2:'医学：影像模态',3:'医学：身体位置'}
    lines = ['# 100%比1%多更新了什么', '',
        '**已完成：实际读取两条Qwen顺序LoRA运行各S1/S2/S3，共六个checkpoint。比较的是有效算子4BA及相邻阶段之差，不是只比较A、B的数值大小。全程CPU，无训练或新推理。**', '',
        '| 探究的问题 | 做了什么实验、结果是什么 | 结论 |',
        '|---|---|---|',
        '| 100%是否多训练了模型参数？ | 六个checkpoint均为28层、56个q/v LoRA、rank8、1,089,536个可训练参数；其余2,208,985,600个冻结参数的实际字节摘要完全一致。 | 没有多解冻参数或新增LoRA，差别在同一组LoRA的实际更新。 |',
        '| 1%是否根本没有更新？ | 有效1%医学两阶段分别有1、2次正学习率更新；两个阶段A/B绝大多数坐标均发生真实变化，梯度有限。 | 不是零学习率或完全没动；但更新机会很少。 |',
        '| 100%是否只是把1%的变化放大？ | 医学两阶段更新范数分别为1%的17.79、12.93倍，但整体方向余弦仅0.166、0.030。 | 同时有幅度和方向差异；单个倍数不能复制100%的参数变化，不代表放大更新一定不能提高分数。 |',
        '| 更新越大，效果就越好吗？ | 自然任务100%更新大7.76倍，但Answer F1为85.817，对1%的86.050几乎持平。 | 单纯鼓励参数多变化，没有充分依据。 |',
        '| 100%是否用了更多LoRA维度？ | 医学两阶段每模块覆盖90%更新能量所需奇异方向，100%平均3.89／4.09，1%为5.84／5.46；名义rank始终相同。 | 100%的变化反而更集中，并非通过更多专家或更大rank获得优势。这是几何描述，不等于方向更有用的因果证明。 |', '',
        '## 主要量化结果', '',
        '| 刚学完的任务 | 正LR更新次数（100%／1%） | 本阶段更新范数比（100%÷1%） | 更新方向余弦 | Answer F1（100%／1%） |',
        '|---|---:|---:|---:|---:|']
    for stage in range(1,4):
        r = increments[stage]['global'];f = e['training']['full'][str(stage)];p = e['training']['pct01'][str(stage)]
        sc = scores[f'S{stage}/T{stage:02}']
        lines.append(f"| {labels[stage]} | {f['positive_used_lr_updates']}／{p['positive_used_lr_updates']} | {r['norm_ratio']:.2f}倍 | {r['cosine']:.3f} | {sc['legacy_full']:.3f}／{sc['valid_pct01']:.3f} |")
    lines += ['', '更新范数为全部56个q/v模块的本阶段有效增量之Frobenius范数，医学阶段用本阶段末减本阶段初，避免把此前累积更新全部算入当前任务。余弦把所有模块连接成一个整体算子向量；不是语义相似度或知识量。', '',
        '## 更新集中在哪里', '',
        '| 阶段 | 100%后7层的更新能量占比 | 1%后7层占比 | 100%主要更新的5层（从0编号） |',
        '|---|---:|---:|---|']
    for d in derived:
        r=increments[d['stage']]
        layers='、'.join(str(x['layer']) for x in r['top_full_increment_layers'])
        lines.append(f"| {labels[d['stage']]} | {d['full']['last7_of28_layers_energy_fraction']*100:.2f}% | {d['pct01']['last7_of28_layers_energy_fraction']*100:.2f}% | {layers} |")
    lines += ['', '这里只能定位注意力投影的更新，不能据此将某层命名为“医学知识层”。两个医学阶段中，v投影相对1%的幅度比为22.02／16.49，q为16.98／12.22；q/v参数尺寸不同，原始总能量不能直接解释为重要性。', '',
        '## 对后续改进的意义', '',
        '当前最值得检验的是：在少数据、少步数下，如何让同一小适配器形成有效更新，而不是先加专家或只奖励更新幅度。现有连续LoRA优化器对照与TRAIN-only更新策略试验继续保留；本次分析没有用TEST挑层、秩、学习率或新方案。', '',
        '## 证据边界', '',
        '100%使用旧图像输入／缓存和预热／scaler框架，有效1%为V2；医学起点已因此前训练而不同，数据量与更新次数也同时变化。因此不能把这些差别全归因于数据量、梯度裁剪、某个层或某个优化器，更不能称已找到唯一性能原因。此处没有受控干预，不把参数几何当作F1的因果贡献。', '',
        '自然任务是重要反例：参数改变更多、方向不同，也可能几乎没有分数收益。医学全量训练的loss下降可在日志中看到，但不同样本的训练loss不能直接作为同一验证集的收敛比较。', '',
        '旧零LR的1%运行未用作这里的主对照。100%只完成S1–S3，不冒充完整19阶段实验。本次CPU分析的显存为零；加载进程约5.4–5.8GB峰值RSS只是分析开销，不是模型训练内存。', '',
        f"实际数值独立验证：使用低秩Gram／trace公式复核全部{checks}项，与稠密4BA计算最大相对误差{max_relative_error:.3g}。", '',
        '证据：[参数与训练日志](EVIDENCE.json)；[独立核验及坐标变化](INDEPENDENT_CHECK.json)；[已有完整测试预测的阶段比较](../../REPORT.md)。六份小参数快照仅在exp下，未提交Git。', '',
        '分析脚本第一版误用了另一骨干的参数数量断言，CPU加载后即被拦截；旧源码与[V1_FAILURE.json](../V1_FAILURE.json)保留。独立v2改为实际Qwen维度，没有修改训练或checkpoint。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
