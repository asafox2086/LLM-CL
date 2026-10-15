"""Build a reproducible, CPU-only review from completed immutable runs."""
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'plot_templates'))
from style import setup_style, polish_axes, save_png_pdf, darken_color
from result_tables import metrics

EXP = HERE.name
OUT = ROOT / 'summary' / EXP
REVIEW = OUT / 'results_review'
DISPLAY = {'PASs_MoE': 'PASs-MoE', 'Hyper_LLaVA': 'Hyper-LLaVA',
           'FV_guided_Training': 'FV-guided Training'}
ARCH = {'SMoLoRA', 'PASs_MoE', 'Hyper_LLaVA'}
MECHANISM = {
    'SMoLoRA': '双路路由与低秩专家，区分任务指令和视觉信息。',
    'PASs_MoE': '路径激活子空间约束路由器与专家的协同变化。',
    'Hyper_LLaVA': '双曲不确定性与模态平衡驱动专家路由。',
    'PaRSP': '区域定位及历史输入子空间的零空间梯度投影。',
    'OSFT': '约束全量微调的可更新子空间，以正交约束保护已有知识。',
    'SpaRTA': '秩感知的任务适配及正交约束，分离可塑与稳定部分。',
    'FV_guided_Training': '函数向量引导及 KL 约束，稳定任务相关的内部功能。',
    'ALER': '对抗潜在嵌入搜索及修复约束，抑制遗忘。',
}
TASKS = [('自然／通用', '文本', 'SQuAD 1.1'),
         ('自然／通用', '图像', 'VQAv2 / GQA / TextVQA'),
         ('医学', '文本', 'MedQuAD'), ('医学', '图像', 'VQA-RAD / PathVQA / SLAKE'),
         ('地理／遥感', '文本', 'GeoQuestions1089'), ('地理／遥感', '图像', 'GeoChat'),
         ('金融', '文本', 'FinQA'), ('金融', '图像', 'MME-Finance PC'),
         ('艺术／文化', '文本', 'AQUA knowledge'), ('艺术／文化', '图像', 'AQUA visual')]
SCORES = {'rougeL': 'ROUGE-L', 'exact_match': 'Normalized EM', 'token_f1': 'Token F1'}


def dump_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    REVIEW.mkdir(parents=True, exist_ok=True)
    status = json.loads((OUT / 'run_status.json').read_text())
    manifest_path = ROOT / 'data' / 'TenTaskCL' / EXP / 'manifest.json'
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert digest == status['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    expected = {}
    for task in manifest['tasks']:
        p = ROOT / task['files']['test']['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == task['files']['test']['sha256']
        expected[int(task['task_id'][1:])] = {
            r['id']: r for r in map(json.loads, p.read_text().splitlines())}
    snapshot = {'experiment_id': EXP, 'manifest_sha256': digest,
                'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                'status_snapshot': status, 'methods': {}}
    final_rows = []
    for method, job in status['jobs'].items():
        if job['status'] != 'completed':
            continue
        run = ROOT / job['run_dir']
        cfg = json.loads((run / 'config.json').read_text())
        assert cfg['experiment_id'] == EXP and cfg['manifest_sha256'] == digest
        assert cfg['seed'] == 42 and cfg['counts_per_task']['train'] == 100
        assert cfg['counts_per_task']['test'] == 50
        matrices = json.loads((run / 'score_matrices.json').read_text())
        hashes = {}
        for stage in range(11):
            for task in range(1, 11):
                p = run / 'evaluations' / f'stage_{stage:02d}' / f'T{task:02d}_test_predictions.jsonl'
                rows = list(map(json.loads, p.read_text().splitlines()))
                assert len(rows) == 50 and {r['id'] for r in rows} == set(expected[task])
                assert len({r['id'] for r in rows}) == 50
                hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
                for score in SCORES:
                    a = np.asarray(matrices[score], dtype=float)
                    assert a.shape == (11, 10) and np.isfinite(a).all()
                    assert ((a >= 0) & (a <= 100)).all()
                    value = sum(r['scores'][score] for r in rows) / 50
                    assert math.isclose(value, a[stage, task - 1], abs_tol=1e-7), (method, score, value)
        result = {score: metrics(matrices[score]) for score in SCORES}
        recorded = json.loads((run / 'metrics_by_score.json').read_text())
        for score, values in result.items():
            for key, value in values.items():
                assert math.isclose(value, recorded[score][key], abs_tol=1e-7)
            final_rows.append({'method': DISPLAY.get(method, method),
                               'category': 'architecture' if method in ARCH else 'regularization',
                               'score': score, **values, 'seed': 42, 'run_dir': job['run_dir']})
            dump_csv(REVIEW / f'{method}_{score}_11x10.csv',
                     [dict(stage=i, **{f'T{j+1:02d}': v for j, v in enumerate(row)})
                      for i, row in enumerate(matrices[score])],
                     ['stage'] + [f'T{j:02d}' for j in range(1, 11)])
        snapshot['methods'][method] = {'run_dir': job['run_dir'], 'config': cfg,
                                      'matrices': matrices, 'metrics': result,
                                      'verified_cells': 110, 'verified_predictions': 5500,
                                      'prediction_sha256': hashes}
        # Each line is one fixed test task across the eleven training stages.
        setup_style('line')
        fig, ax = plt.subplots(figsize=(11.8, 6.8), dpi=300)
        colors = [matplotlib.colors.to_hex(plt.cm.tab10(i)) for i in range(10)]
        markers = ['o', 'D', '^', 's', 'v', 'P', 'X', '<', '>', 'h', '*']
        task_labels = ['General text', 'General image', 'Medical text', 'Medical image',
                       'Geography text', 'Remote sensing image', 'Finance text',
                       'Finance chart', 'Art/culture text', 'Art/culture image']
        for task, row in enumerate(np.asarray(matrices['rougeL']).T):
            ax.plot(range(11), row, color=colors[task],
                    label=f'T{task+1}: {task_labels[task]}',
                    marker=markers[task], markersize=5.8, markerfacecolor=colors[task],
                    markeredgecolor=darken_color(colors[task], .65), markeredgewidth=.8,
                    linewidth=1.8, linestyle='--' if task % 2 == 0 else '-', zorder=3)
        ax.set(xlim=(-.25, 10.25), ylim=(0, 100), xticks=range(11),
               xticklabels=[f'S{i}' for i in range(11)],
               xlabel='Completed training stage (S0 = initial; Si = after training Ti)', ylabel='ROUGE-L F1 (0–100)')
        category = 'Architecture' if method in ARCH else 'Regularization'
        ax.set_title(f'{DISPLAY.get(method, method)} | Continual fine-tuning / {category}',
                     fontsize=16, fontweight='bold', pad=16)
        ax.legend(loc='center left', bbox_to_anchor=(1.01, .5), frameon=False, fontsize=10)
        polish_axes(ax)
        fig.subplots_adjust(left=.09, right=.72, bottom=.17, top=.89)
        save_png_pdf(fig, str(REVIEW / f'{method}_task_trajectories_rougeL'))
        plt.close(fig)
    dump_csv(REVIEW / 'final_metrics_all_scores.csv', final_rows,
             ['method', 'category', 'score', 'AP', 'ForgetRate', 'FWT', 'BWT', 'seed', 'run_dir'])
    (REVIEW / 'verified_snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 已完成结果：逐阶段十任务性能与最终指标', '',
             f"生成时间：{snapshot['created_at']}。本报告是独立快照，实时队列不会覆盖。", '',
             f"已完成 **{len(snapshot['methods'])}/8** 个方法，每个方法核验 **110 个评测单元、5500 条逐题预测**，ID 与冻结测试集一致；逐题分数均值与矩阵、最终指标交叉核对通过。", '',
             '基座 Qwen2-VL-2B-Instruct；每任务训练 100／测试 50，无验证集；一轮训练；仅 seed 42；不保存 checkpoint。所有方法使用相同冻结样本及训练顺序。', '',
             '## 图怎么读', '',
             '每个完成的方法一张图，**10 条折线分别代表 T1–T10，每条 11 个节点**。横轴 S0–S10 是训练阶段；S0 是训练前，Si 是完成任务 Ti 训练后的评测。颜色和标记在所有方法间一致；纵轴统一为 0–100 的 ROUGE-L F1，越高越好。', '',
             '沿同一条线从左向右看，就是固定任务随持续学习的真实性能轨迹。Sj 是任务 Tj 刚学完的位置，之后的下降体现遗忘；之前的变化反映未学习任务的迁移。未学习任务也保留实际评测值。没有插值、平滑或缺失值补齐。此前以任务为横轴的阶段快照图不用于判断时间变化，主图现已纠正。', '',
             '| 任务 | 领域 | 模态 | 数据集 |', '|---|---|---|---|']
    lines += [f'| T{i} | {d} | {m} | {s} |' for i, (d, m, s) in enumerate(TASKS, 1)]
    lines += ['', '## 论文二维表中的定义', '',
              '本轮方法均属于二维表的**持续微调**大类。基于架构的方法主要通过专家、模块或路由结构分离和组合任务能力；基于正则化的方法通过子空间、梯度或表征／输出约束保护已有知识。这里沿用仓库论文表分类，具体方法可同时包含多种组件。', '',
              '| 方法 | 大类 → 小类 | 主要机制 | 当前状态 |', '|---|---|---|---|']
    for m, j in status['jobs'].items():
        state = '已完成 11×10' if m in snapshot['methods'] else '尚未完成，不发布最终指标'
        lines.append(f"| {DISPLAY.get(m,m)} | 持续微调 → {'基于架构' if m in ARCH else '基于正则化'} | {MECHANISM[m]} | {state} |")
    lines += ['', '论文链接见[仓库二维表](../../README.md)，实现与原作者设置的差异见[方法适配记录](../../exp/' + EXP + '/METHOD_ADAPTATIONS.md)。以下是统一小模型、小样本设置下的适配结果，不是原论文排行榜数值。OSFT、ALER 包含更重的全量参数更新，不能据此假定各方法训练成本相同。', '', '## 最终指标汇总']
    for score, label in SCORES.items():
        lines += ['', f'### {label}', '', '| 方法 | AP ↑ | ForgetRate ↓ | FWT ↑ | BWT ↑ |', '|---|---:|---:|---:|---:|']
        for row in final_rows:
            if row['score'] == score:
                lines.append('| ' + row['method'] + ' | ' + ' | '.join(f"{row[k]:.3f}" for k in ['AP', 'ForgetRate', 'FWT', 'BWT']) + ' |')
    lines += ['', '### 指标定义与解释', '',
              '令 R[i,j] 为完成第 i 阶段后在任务 j 上的分数（i=0…10，j=1…10）。每一种评分分别从自己的完整 11×10 矩阵计算：', '',
              '- AP = mean(j=1…10) R[10,j]：最终平均性能。',
              '- ForgetRate = mean(j=1…9) (max(k=j…9) R[k,j] − R[10,j])：沿用当前实验定义的历史最佳与最终性能之差，不截断负值。',
              '- BWT = mean(j=1…9) (R[10,j] − R[j,j])：后续训练对已学任务的影响。',
              '- FWT = mean(j=2…10) (R[j−1,j] − R[0,j])：学习某任务前，相对本方法初始模型的变化。', '',
              '**ForgetRate 是分数点差，不是归一化比例**；FWT、BWT 也为分数点差。负 BWT 表示平均遗忘，负 FWT 表示尚未学习的任务平均低于初始参考。统一 EM、词面 Token F1、ROUGE-L 不等于各原始基准的全部官方指标；EM 的正确数／总数保留在任务及数据集明细中。只有一个 seed，不报告统计显著性或跨 seed 误差条。', '',
              '## 各方法折线图']
    for m in snapshot['methods']:
        lines += ['', f'### {DISPLAY.get(m,m)}', '',
                  f"二维表定位：持续微调 → {'基于架构' if m in ARCH else '基于正则化'}。{MECHANISM[m]}", '',
                  f'![{DISPLAY.get(m,m)} 的 10 条任务曲线，每条 11 个训练阶段节点](results_review/{m}_task_trajectories_rougeL.png)', '',
                  f'[矢量 PDF](results_review/{m}_task_trajectories_rougeL.pdf) · [ROUGE-L 11×10 数据](results_review/{m}_rougeL_11x10.csv) · [EM 矩阵](results_review/{m}_exact_match_11x10.csv) · [Token F1 矩阵](results_review/{m}_token_f1_11x10.csv)']
    lines += ['', '## 可追溯文件', '',
              '- [全部最终指标 CSV](results_review/final_metrics_all_scores.csv)：三种评分各自的 AP、ForgetRate、FWT、BWT，保留完整精度。',
              '- [核验快照与绘图原始数值](results_review/verified_snapshot.json)：实际 run 路径、配置、完整矩阵、逐预测文件 SHA-256。',
              '- [按任务明细](task_metrics.csv)及[按来源数据集明细](dataset_metrics.csv)：EM 正确数／总数及其他分数；这两个文件随队列更新。',
              f'- [可复现生成脚本](../../exp/{EXP}/build_results_review.py)：CPU 绘图，不占用训练 GPU。', '',
              f'冻结 manifest SHA-256：`{digest}`。未完成方法不使用旧运行、旧划分或中断阶段拼接成最终结果。', '']
    (OUT / 'results_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'completed': list(snapshot['methods']), 'report': str(OUT / 'results_report.md'),
                      'verified_predictions': 5500 * len(snapshot['methods'])}))


if __name__ == '__main__':
    main()
