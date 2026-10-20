"""Publish zero-update calibration and link the sealed CPU study; no models."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REL='summary/edge_deployment_20261019/cpu_update_space_20261020/v1'
D=ROOT/REL


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepend(path,text):
    old=path.read_text()
    assert text.splitlines()[0] not in old
    path.write_text(text+'\n\n'+old)


def main():
    z=D/'zero_training_control';c=json.loads((z/'COMPLETE.json').read_text());v=json.loads((z/'EVIDENCE.json').read_text())
    assert c['status']==v['status']=='CPU_ZERO_TRAINING_TWO_MATRIX_COMPLETE'
    assert sha(z/'EVIDENCE.json')==c['evidence_sha256']
    assert sha(ROOT/'analysis/cpu_osft_zero_training_20261020.py')==c['source_sha256']==v['source_sha256']
    assert v['actual_evidence_sha256']==sha(D/'EVIDENCE.json')
    assert not v['QA_access'] and not v['cuda_initialized'] and v['optimizer_updates']==0
    assert len(v['rows'])==2
    lines=['# 零训练控制：这两处CPU重分解漂移不足以解释实际变化','',
           '原数值条件、原作者核，只有初始分解及18次阶段重分解，无训练、模型前向或QA访问。','',
           '| 预定矩阵 | 零训练变化范数／实际变化范数 | 零训练与实际方向余弦 |',
           '| --- | ---: | ---: |']
    for x in v['rows']:
        assert len(x['zero_training_history'])==19
        assert [a['decompositions'] for a in x['zero_training_history']]==list(range(1,20))
        assert math.isclose(x['zero_to_actual_norm_ratio']**2,x['zero_energy']/x['actual_energy'],rel_tol=1e-12)
        assert math.isclose(math.fsum(s*s for s in x['final_zero_singular_values']),x['zero_energy'],rel_tol=1e-9,abs_tol=1e-12)
        lines.append(f"| {x['name']} | {100*x['zero_to_actual_norm_ratio']:.4f}% | {x['actual_vs_zero_cosine']:.6f} |")
    lines+=['','**结论：** 在这两处控制中，CPU零训练漂移不足实际变化范数的0.1%，不能单靠这个CPU重分解路径解释实际变化。',
            '', '**边界：** 原训练SVD在GPU、这里在CPU，不是原GPU数值轨迹精确重放；没有验证其余166矩阵，也没有证明实际变化都是有用知识或导致F1提升。不是新的精度／量化研究，不继续搜索dtype或算子。',
            '',f"控制实际CPU两线程，耗时{v['seconds']:.1f}秒。全部结果与源SHA见[EVIDENCE](EVIDENCE.json)。"]
    (z/'README.md').write_text('\n'.join(lines)+'\n')
    (z/'INDEPENDENT_CHECK.json').write_text(json.dumps({'status':'ZERO_TRAINING_INDEPENDENT_CHECK_PASS','evidence_sha256':sha(z/'EVIDENCE.json'),'source_sha256':sha(__file__),'rows':2,'CPU_only':True},indent=2)+'\n')
    text='\n## 已补零训练校准\n\n固定的layer0/q与layer23/down，在原数值条件下只做19次CPU作者分解／重建，零训练漂移仅为实际变化范数的0.0961%／0.0349%，不足以解释这两处实际变化。CPU与原GPU后端不同，不是全部矩阵或原训练数值轨迹的精确反事实，不能把差额当有用学习比例。[完整两层控制与边界](zero_training_control/README.md)。\n'
    p=D/'README.md';old=p.read_text();assert '## 已补零训练校准' not in old;p.write_text(old+text)
    note='**2026-10-20 CPU实际更新空间：** [216个正式最终矩阵的谱分析与两处零训练控制]('+REL+'/README.md)已完成：OSFT变化平方范数96.67%在q/v之外，每矩阵最佳rank8只保留8.20%总变化；共同q/v投到SeqFT末态切空间保留1.97%，低维各向同性参考1.84%。两处零训练CPU漂移不足实际变化范数0.1%。这些是权重空间事实，不是F1贡献或参数可删证明，不能把OSFT等同普通小LoRA；原四组GPU1小测与两机基线保持。'
    prepend(ROOT/'README.md',note)
    detail='**2026-10-20最新CPU参数空间证据（K59）：** [正式最终矩阵与零训练控制](../edge_deployment_20261019/cpu_update_space_20261020/v1/README.md)：OSFT的168个矩阵变化96.67%平方范数在q/v之外，最佳每层rank8只覆盖8.20%；SeqFT代数rank8更新覆盖100%。末态局部空间537600／71245824维，不等于整体轨迹可达空间。两处CPU零训练漂移仅实际范数0.0961%／0.0349%，只排除该CPU路径在这两层的充分解释。没有新F1或端侧收益，不按能量删参数；GPU四组小测仍须检验位置与学习效果。旧队列段落为历史，以AGENTS顶部和当前screen入口为准。'
    for name in ['RESEARCH_REPORT.md','CAUSAL_STATUS.md']:
        prepend(ROOT/'summary/onevision_diagnostics_20261016'/name,detail)
    k=ROOT/'summary/onevision_diagnostics_20261016/KEY_FINDINGS.md';old=k.read_text();assert '## K59：' not in old
    old+='\n## K59：正式OSFT的实际更新不等同于普通小LoRA；零训练控制校准解释\n\n'
    old+='**问题／实验：** 在相同原基座读取正式1%OSFT与SeqFT最终checkpoint，CPU两线程计算168＋48个实际更新矩阵完整谱，独立复算能量／截断残差及LoRA QR谱；不读QA或新推理。\n\n'
    old+='**结果：** OSFT更新96.67%平方范数在q/v外，最佳rank8保留8.20%、rank32保留15.05%；SeqFT代数rank8保留100%。OSFT末态保护块71245824维、Seq末态切空间537600维；OSFT q/v位移在Seq末态切空间保留1.97%，各向同性参考1.84%。这不是特殊冲突证据。两处预定CPU零训练19次重分解漂移只达实际范数0.0961%／0.0349%。\n\n'
    old+='**结论／行动：** 用普通rank8替换OSFT不是等价的权重空间；增加位置能否保住F1仍由已有四组小测检验。没有证明这些变化全有用、所有低秩都无效、哪些模块可删或已有端侧收益。两层CPU校准不是全部矩阵或原GPU数值路径的精确反事实。最终权重、学习率、投影、位置与训练轨迹差异仍需区分。\n\n'
    old+='[数据、全部边界与零训练控制](../edge_deployment_20261019/cpu_update_space_20261020/v1/README.md)。\n';k.write_text(old)
    head='# 2026-10-20：用户授权CPU探究；正式最终更新空间已完成实测\n\n- 新'+REL+'/README.md：正式OSFT168矩阵与SeqFT48矩阵，CPU两线程135秒，全谱及独立截断残差／QR核验通过。OSFT变化96.67%平方范数在q/v外，最佳每矩阵rank8保留8.20%；不是准确率贡献或可删性。两方法末态局部维数不同，不能把训练坐标数当有效空间。\n- 固定两处零训练CPU原作者19次重分解控制已完成：漂移只占实际变化范数0.0961%／0.0349%；原GPU后端不同，仅排除该CPU路径在两处的充分解释，不把高秩尾部全称有用学习。没有QA／新模型前向／优化器／GPU调用。测量脚本与原模型不改；报告v1路径类型错误单列，独立v2仅修文档hash，不重跑测量。\n- 新CPU分析是真实新证据，不解除GPU1守卫，不恢复旧CPU／无条件全测。原四组screen仍需真实F1小测，Ours和未测端侧指标保持空白，基线nohup持续。CPU标量／谱不等于F1或部署收益。'
    prepend(ROOT/'AGENTS.md',head)
    print(json.dumps({'published':True,'CPU_only':True,'no_model_or_GPU_work':True,'zero_rows':2}))


if __name__=='__main__':main()
