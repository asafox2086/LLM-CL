"""CPU-only saved-factor audit; no model loading, optimizer or GPU admission.

Separates first-step zero-B geometry from the old Qwen zero-LR confound.
Historical isolated/default-sequential comparisons are descriptive, not paired.
"""
import hashlib,json,math
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(a,b,factor):
    return math.sqrt(sum(float((a[k][factor].double()-b[k][factor].double()).square().sum()) for k in a))


def algebra_check():
    # Local RNG; synthetic algebra check is not an actual model experiment.
    g=torch.Generator().manual_seed(42)
    a=torch.randn(3,5,generator=g,dtype=torch.float64,requires_grad=True)
    b=torch.zeros(4,3,dtype=torch.float64,requires_grad=True)
    x=torch.randn(7,5,generator=g,dtype=torch.float64)
    y=torch.randn(7,4,generator=g,dtype=torch.float64)
    opt=torch.optim.AdamW([a,b],lr=1e-4,weight_decay=0)
    first=a.detach().clone();records=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True);loss=(4*x@a.T@b.T-y).square().mean();loss.backward()
        records.append({'step':step+1,'A_grad_l2':float(a.grad.norm()),'B_grad_l2':float(b.grad.norm())})
        opt.step()
        if step==0:assert torch.equal(a,first) and torch.count_nonzero(b)>0
    assert records[0]['A_grad_l2']==0 and records[1]['A_grad_l2']>0
    return records


def main():
    torch.set_num_threads(4)
    pointer=REPORT/'research_current_attempt.json';ref=json.loads(pointer.read_text())
    old=ROOT/ref['old_control_run'];sequential=ROOT/ref['run']
    initial_path=sequential/'initial_adapter.pt'
    initial=torch.load(initial_path,map_location='cpu',weights_only=False)['adapter']
    assert len(initial)==48 and all(torch.count_nonzero(x['B'])==0 for x in initial.values())
    sources={str(initial_path.relative_to(ROOT)):digest(initial_path),str(pointer.relative_to(ROOT)):digest(pointer)}
    def load_checked(path,expected=None,durable=False):
        sha=digest(path);assert expected is None or sha==expected
        if durable:
            meta=json.loads(path.with_suffix('.json').read_text());assert meta['sha256']==sha
        sources[str(path.relative_to(ROOT))]=sha
        return torch.load(path,map_location='cpu',weights_only=not durable)['adapter']
    events={}
    for name,run in [('isolated',old),('sequential',sequential)]:
        path=run/'events.jsonl';sources[str(path.relative_to(ROOT))]=digest(path)
        events[name]=[json.loads(line) for line in path.read_text().splitlines()]
    rows=[];previous=initial
    for tid in range(1,20):
        path=old/'adapters'/f'native_isolated_T{tid:02d}.pt'
        iso=load_checked(path,ref['completed_control_sha256'][str(path.relative_to(ROOT))])
        seq=load_checked(sequential/'durable_states'/f'stage_{tid:02d}.pt',durable=True)
        row={'task':tid}
        for kind,label,state,start in [('isolated',f'native_isolated_T{tid:02d}',iso,initial),('sequential',f'sequential_T{tid:02d}',seq,previous)]:
            logged=[e for e in events[kind] if e['label']==label]
            assert logged and all(e['learning_rate_used']>0 for e in logged)
            row[kind]={'updates':len(logged),'positive_lr_updates':len(logged),
                       'A_change_l2':distance(state,start,'A'),'B_change_l2':distance(state,start,'B'),
                       'A_equal_start':all(torch.equal(state[k]['A'],start[k]['A']) for k in state),
                       'B_equal_start':all(torch.equal(state[k]['B'],start[k]['B']) for k in state)}
        assert row['isolated']['updates']==row['sequential']['updates']
        if row['isolated']['updates']==1:
            assert row['isolated']['A_equal_start'] and not row['isolated']['B_equal_start']
        rows.append(row);previous=seq
    params={f:sum(x[f].numel() for x in initial.values()) for f in ['A','B']}
    one=[r['task'] for r in rows if r['isolated']['updates']==1]
    result={'rows':rows,'one_update_tasks':one,'parameter_counts':params,
            'synthetic_algebra_check':algebra_check(),'source_sha256':sources,
            'audit_sha256':digest(Path(__file__)),'sources_read_only':True,'GPU_used':False,
            'limitations':['OneVision positive-LR phenomenon, separate from Qwen zero-LR scheduler confound.',
                           'Unchanged A on step1 follows B=0, zero weight decay and fresh optimizer; it is expected, not a broken gradient.',
                           'Sequential B is inherited, so its first-stage-update A gradient need not vanish.',
                           'Historical isolated/sequential dropout streams differ; no causal performance comparison.',
                           'Parameter update evidence does not prove underfitting or knowledge sufficiency.',
                           'No data or training decisions were changed by this read-only audit; no accuracy inferred from factor motion.']}
    (REPORT/'factor_update_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 单次更新与LoRA因子：参数证据','',
           '只读CPU审计19个独立任务与19个已完成顺序阶段；没有追加训练或占用GPU。', '',
           f'9个单更新任务：{one}。所有任务均有正学习率；这不是旧Qwen队列的零学习率问题。', '',
           'LoRA为ΔW=4BA，初始B=0，故第一次反传∂L/∂A=4BᵀG=0；'
           '当前AdamW无权重衰减、每阶段新建优化器。第一次更新改变B，A保持初始化是预期行为。'
           '9个独立任务的A均逐元素未变、B均非零；合成双精度两步检查亦确认第二步A梯度可非零。', '',
           f'A共{params["A"]}、B共{params["B"]}参数。单步独立任务实际只更新其中B；不能说整个LoRA没有学习。', '',
           '| T | 更新数 | 独立A变化L2 | 独立B变化L2 | 顺序A阶段变化L2 | 顺序B阶段变化L2 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        i,s=r['isolated'],r['sequential']
        lines.append(f'| {r["task"]} | {i["updates"]} | {i["A_change_l2"]:.6f} | {i["B_change_l2"]:.6f} | {s["A_change_l2"]:.6f} | {s["B_change_l2"]:.6f} |')
    lines+=['','顺序训练继承非零B，因此即使该阶段只有一步也可能更新A。这是两种起点的优化差异，'
            '不能把顺序／旧LoRA复用的收益都解释为专业知识。不同训练流尚未严格配对，因子范数也不直接代表性能。', '',
            '下一步应在固定训练预算下比较随机A+零B与旧A+零B（初始ΔW都为零），并另报告更多更新的预算对照。'
            '这些GPU干预尚未完成，本审计不能证明训练不足造成了多少准确率损失。', '', '限制：','']
    lines+=['- '+x for x in result['limitations']]
    (REPORT/'FACTOR_UPDATE_EVIDENCE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'one_update_tasks':one,'one_update_sequential_A_changed':[r['task'] for r in rows if r['task'] in one and not r['sequential']['A_equal_start']], 'parameter_counts':params}))


if __name__=='__main__':main()
