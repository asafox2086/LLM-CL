"""Verify and report fixed external knowledge probes across every saved stage."""
from captions import add_captions
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
METHODS=['seq_lora','migu_lora','olora','sapt_lora']
NAMES={'seq_lora':'SeqLoRA','migu_lora':'MIGU-LoRA','olora':'O-LoRA','sapt_lora':'SAPT-LoRA'}

def save_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def collect():
    run=Path(json.loads((ROOT/'summary_cv/latest.json').read_text())['run'])
    rows,transitions,check=[],[],[];baseline_predictions=None
    for method in METHODS:
        folder=run/'knowledge_probe'/method
        status=json.loads((folder/'status.json').read_text());assert status['status']=='completed'
        scores=json.loads((folder/'scores.json').read_text());assert len(scores)==10
        protocol=json.loads((folder/'protocol.json').read_text())
        assert hashlib.sha256((folder/'source.py').read_bytes()).hexdigest()==protocol['code_sha256']
        base=None;previous=None
        for stage in range(10):
            path=folder/f'stage_{stage:02d}.jsonl';records=[json.loads(line) for line in path.read_text().splitlines()]
            assert len(records)==198
            assert all(r['stage']==stage and r['method']==method for r in records)
            assert all(r['probe_sha256']==protocol['dataset']['selection_sha256'] for r in records)
            ids=[r['instance_id'] for r in records];assert len(set(ids))==198
            if base is None:base=records
            assert ids==[r['instance_id'] for r in base]
            def accuracy(records):return 100*sum(r['correct'] for r in records)/len(records)
            subject_scores={s:accuracy([r for r in records if r['subject']==s]) for s in protocol['dataset']['subjects']}
            values={'method':method,'stage':stage,'general_accuracy':accuracy([r for r in records if r['group']=='general']),
                    'medical_accuracy':accuracy([r for r in records if r['group']=='medical']),
                    'macro_accuracy':sum(subject_scores.values())/57}
            for key in ['general_accuracy','medical_accuracy','macro_accuracy']:
                assert math.isclose(values[key],scores[stage][key],abs_tol=1e-10)
            rows.append(values)
            for group in ['general','medical']:
                indices=[i for i,r in enumerate(records) if r['group']==group]
                lost=sum(base[i]['correct']==1 and records[i]['correct']==0 for i in indices)
                gained=sum(base[i]['correct']==0 and records[i]['correct']==1 for i in indices)
                base_correct=sum(base[i]['correct'] for i in indices)
                transitions.append({'method':method,'stage':stage,'group':group,'count':len(indices),
                    'lost_base_correct':lost,'gained_base_wrong':gained,'base_correct':base_correct,
                    'base_correct_retained_percent':100*(base_correct-lost)/base_correct,
                    'accuracy_change_vs_base':100*(gained-lost)/len(indices),
                    'lost_previous_correct':sum(previous[i]['correct']==1 and records[i]['correct']==0 for i in indices) if previous else None,
                    'gained_previous_wrong':sum(previous[i]['correct']==0 and records[i]['correct']==1 for i in indices) if previous else None})
            previous=records
        baseline=[r['prediction'] for r in base]
        if baseline_predictions is None:baseline_predictions=baseline
        else:assert baseline==baseline_predictions
        check.append({'method':method,'stages':10,'predictions':1980,'scores_recomputed':True,
            'same_base_predictions':True,'code_snapshot_verified':True})
    save_csv(ROOT/'summary_cv/knowledge_probe_results.csv',rows)
    save_csv(ROOT/'summary_cv/knowledge_probe_transitions.csv',transitions)
    (ROOT/'summary_cv/knowledge_probe_verification.json').write_text(json.dumps({'checks':check,'total_predictions':7920},indent=2)+'\n')
    lines=['# 外部知识题：逐阶段表现与保留','','使用已保存的基线及九个阶段 checkpoint，在不参与本次训练的固定 MMLU test 子集上补测。包含 51 个非医学学科各 2 题（102 题），6 个医学相关学科各 16 题（96 题），共 198 题。四方法、10 阶段均已完成，共核对 7,920 条选择题记录。','',
        '## 先看整体变化','','![固定外部知识题的逐阶段表现](../sample/cv_external_knowledge.png)','',
        '所有阶段使用完全相同的题目和提示。通用、医学两组分别算匹配准确率；57 学科宏平均按学科等权，避免医学题多导致其权重过大。','',
        '| 方法 | 通用：基座 → 最终 | 医学：基座 → 最终 | 57 学科宏平均：基座 → 最终 |','|---|---:|---:|---:|']
    for method in METHODS:
        base=next(r for r in rows if r['method']==method and r['stage']==0)
        final=next(r for r in rows if r['method']==method and r['stage']==9)
        lines.append('| '+NAMES[method]+' | '+' | '.join(f"{base[k]:.2f}% → {final[k]:.2f}% ({final[k]-base[k]:+.2f} 点)" for k in ['general_accuracy','medical_accuracy','macro_accuracy'])+' |')
    lines+=['','## 原来答对的题，后来还答得对吗','','| 方法 | 领域 | 基座答对 | 后来答错 | 原来答错、后来答对 | 保留基座正确答案的比例 |','|---|---|---:|---:|---:|---:|']
    for r in transitions:
        if r['stage']==9:
            lines.append(f"| {NAMES[r['method']]} | {r['group']} | {r['base_correct']} | {r['lost_base_correct']} | {r['gained_base_wrong']} | {r['base_correct_retained_percent']:.2f}% |")
    lines+=['','同样的准确率可以由不同的“新答对 / 原来答对后又答错”组合产生，因此保留这两类问题的计数。全部相邻阶段的题目转变见 [transitions.csv](knowledge_probe_transitions.csv)。','',
        '## 全部阶段','','| 方法 | 阶段 | 通用准确率 (%) | 医学准确率 (%) | 学科宏平均 (%) |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {NAMES[r['method']]} | {r['stage']} | {r['general_accuracy']:.2f} | {r['medical_accuracy']:.2f} | {r['macro_accuracy']:.2f} |")
    lines+=['','## 指标定义与适用范围','','设固定题目集合为 $Q_g$：','',
        r'$$',r'K_s^{(g)}=\frac{100}{|Q_g|}\sum_{q\in Q_g}\mathbf{1}\!\left[\widehat{y}_{s,q}=y_q\right].',r'$$','',
        r'$$',r'\mathrm{Retention}_s^{(g)}=100\cdot\frac{\sum_{q\in Q_g}\mathbf{1}[\widehat{y}_{0,q}=y_q\;\land\;\widehat{y}_{s,q}=y_q]}{\sum_{q\in Q_g}\mathbf{1}[\widehat{y}_{0,q}=y_q]}.',r'$$','',
        '这是固定小样本的外部学科知识表现，并非完整 MMLU 分数，也不是“模型所有知识”的测量。非医学学科每科只有两题，单题影响明显；本轮只有一个训练种子，不提供多种子方差或声称显著性。医学相关组包括 anatomy、clinical_knowledge、college_biology、college_medicine、medical_genetics、professional_medicine。','',
        '采用统一的零样本四选一提示，只比较 A/B/C/D 下一 token 的 logits。概率只在这四个候选字母内归一化；这里的指标与九任务的生成答案 ROUGE-L 属于不同评测设置。题目不用于训练、选 checkpoint 或调参；无法确定模型原始预训练是否已见过 MMLU。','',
        '## 来源、保存与重算','','来源：[MMLU 作者](https://github.com/hendrycks/test) / [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)。下载通过镜像，固定 commit，核对 LFS SHA256；[协议与哈希](knowledge_probe_protocol.json)、[固定题目](knowledge_probe_items.jsonl)、[核对记录](knowledge_probe_verification.json) 均随报告保留。','',
        '各阶段逐题 prompt、token、候选概率、预测字母与参考保存在本次运行的 `knowledge_probe/<method>/`。补测脚本只读取原始训练 checkpoint，未改写训练结果；首次加载零阶段的空适配器状态已作为单独情形处理。','',
        '```bash','.plot-env/bin/python exp/prepare_knowledge_probe.py',
        '# CUDA_VISIBLE_DEVICES 指定 GPU；断开终端后继续运行，并可复用已保存的批次',
        'CUDA_VISIBLE_DEVICES=0 nohup setsid .plot-env/bin/python -u exp/evaluate_knowledge_probe.py --method seq_lora > /tmp/knowledge_probe.log 2>&1 < /dev/null &',
        '.vision-env/bin/python summary_cv/collect_knowledge_probe.py','.plot-env/bin/python summary_cv/plot_process.py','```']
    (ROOT/'summary_cv/knowledge_probe.md').write_text(add_captions('\n'.join(lines)+'\n'))
    print('\n'.join(lines[8:15]))

if __name__=='__main__':collect()
