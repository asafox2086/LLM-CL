"""Independent stdlib aggregation of sealed CPU spectra; no model or QA."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/edge_deployment_20261019/cpu_update_space_20261020/v1'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    c = json.loads((REPORT/'COMPLETE.json').read_text())
    e = json.loads((REPORT/'EVIDENCE.json').read_text())
    assert c['status'] == 'CPU_FINAL_WEIGHT_SPACE_COMPLETE'
    assert c['evidence_sha256'] == sha(REPORT/'EVIDENCE.json')
    assert c['source_sha256'] == sha(ROOT/'analysis/cpu_update_space_20261020.py')
    assert e['CPU_only'] and not e['QA_access'] and not e['cuda_initialized']
    assert len(e['OSFT_layers']) == 168 and len(e['SeqFT_layers']) == 48
    assert len({x['name'] for x in e['OSFT_layers']}) == 168
    ranks = ['2','4','8','16','32','64','128','256']
    for rows in [e['OSFT_layers'], e['SeqFT_layers']]:
        for x in rows:
            s = x['singular_values']
            assert len(s) == min(x['shape'])
            assert all(v >= 0 and math.isfinite(v) for v in s)
            assert all(a >= b for a,b in zip(s,s[1:]))
            assert math.isclose(math.fsum(v*v for v in s), x['energy'], rel_tol=1e-9, abs_tol=1e-12)
            for k in ranks:
                assert math.isclose(math.fsum(v*v for v in s[:int(k)]), x['rank_energy'][k], rel_tol=1e-9, abs_tol=1e-12)
    total = math.fsum(x['energy'] for x in e['OSFT_layers'])
    assert math.isclose(total, e['OSFT']['energy'], rel_tol=1e-12)
    assert math.isclose(math.fsum(x['energy'] for x in e['OSFT_by_projection'].values()), total, rel_tol=1e-12)
    for k in ranks:
        expected = math.fsum(x['rank_energy'][k] for x in e['OSFT_layers'])/total
        assert math.isclose(expected, e['OSFT']['rank_retention'][k], rel_tol=1e-12)
    extra = math.fsum(v['total_energy_fraction'] for k,v in e['OSFT_by_projection'].items() if k not in ['q_proj','v_proj'])
    proposed8 = sum(8*sum(x['shape']) for x in e['OSFT_layers'])
    plain_dim = math.fsum(x['SeqFT_tangent_dimension'] for x in e['OSFT_layers'] if 'SeqFT_tangent_dimension' in x)
    assert plain_dim == e['OSFT_qv_to_SeqFT_final_tangent']['local_dimension']
    check = {'status': 'INDEPENDENT_SPECTRA_AGGREGATION_PASS', 'evidence_sha256': sha(REPORT/'EVIDENCE.json'),
             'report_source_sha256': sha(__file__), 'verified_spectra': 216, 'official_QA_read': False,
             'OSFT_extra_locations_energy_fraction': extra, 'ordinary_all_projection_rank8_coordinates': proposed8,
             'CPU_only': True, 'new_model_forward': False}
    (REPORT/'INDEPENDENT_CHECK.json').write_text(json.dumps(check,indent=2)+'\n')
    lines = ['# 1%正式最终参数：变化发生在哪里，小LoRA能表示多少？', '',
             'CPU读取已完成的OSFT和顺序LoRA最终checkpoint，对比同一原始基座；没有新训练、模型前向或测试答案访问。全部168个OSFT矩阵、48个LoRA矩阵及两处独立截断残差已核验。', '',
             '| 探究的问题 | 做了什么实验，结果是什么 | 结论 |', '| --- | --- | --- |',
             f"| OSFT只是把q/v训得更多吗？ | 将实际最终权重变化按7类位置拆开；q/v之外占总平方范数的{100*extra:.2f}%。 | OSFT实际变化并不限于q/v；能量不是准确率贡献，不能据此删q/v或指定最重要位置。 |",
             f"| 每层rank8能完整复现这些变化吗？ | 逐矩阵计算最佳rank8近似，保留OSFT总更新能量{100*e['OSFT']['rank_retention']['8']:.2f}%；顺序LoRA代数更新保留{100*e['SeqFT_algebraic_BA']['rank_retention']['8']:.2f}%。 | 这是可表示性的上界测量，不是训练出来的近似模型；未测Answer F1。 |",
             f"| 两种方法的可更新空间相同吗？ | 顺序LoRA训练坐标540672，末态局部权重维数{int(plain_dim)}；OSFT训练坐标113766144，末态保护块维数{e['OSFT_final_protected_local_dimension']}。 | 不能用训练坐标数代替有效权重空间；OSFT同时改变位置与空间，仍未隔离唯一性能原因。 |",
             f"| OSFT变化落在已有LoRA空间中吗？ | 在共同q/v位置，OSFT变化投到SeqFT末态切空间保留{100*e['OSFT_qv_to_SeqFT_final_tangent']['retained_energy_fraction']:.2f}%；按维数计算的各向同性参考{100*e['OSFT_qv_to_SeqFT_final_tangent']['isotropic_fraction']:.2f}%。 | 两种末态方向不同；末态切空间不是整个训练可达空间，不能说LoRA完全学不到或方向有害。 |", '',
             '## 各位置的实际变化', '',
             '| 位置 | OSFT更新平方范数占比 | 相对原权重范数 | 最佳rank8保留 | 最佳rank32保留 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for k,v in e['OSFT_by_projection'].items():
        lines.append(f"| {k} | {100*v['total_energy_fraction']:.2f}% | {100*v['relative_norm']:.4f}% | {100*v['rank_retention']['8']:.2f}% | {100*v['rank_retention']['32']:.2f}% |")
    lines += ['', '平方范数占比按绝对权重能量汇总，大矩阵可能占更多；因此同时报告相对原权重幅度。以上都不是功能重要性或梯度显著性。', '',
              '## 固定秩测量网格', '', '| 每矩阵保留秩 | OSFT更新能量保留 | 顺序LoRA代数更新保留 |', '| ---: | ---: | ---: |']
    for k in ranks:
        lines.append(f"| {k} | {100*e['OSFT']['rank_retention'][k]:.3f}% | {100*e['SeqFT_algebraic_BA']['rank_retention'][k]:.3f}% |")
    lines += ['', f'原q/v rank8有540672个训练坐标；七类位置都用rank8则是{proposed8}个。这是结构计数，不是显存或端侧模型大小。', '',
              '## 后续行动与边界', '',
              'GPU1原四组小测继续检验“普通LoRA增加／替换更新位置能否提高F1”；本分析不根据TEST选择位置、秩或停止点，不改变其冻结协议。若最佳低秩近似误差较大，不能把普通rank8视作与OSFT等价的更新空间；若近似好，也仍要验证少样本能否学习到。', '',
              'OSFT实际末态减基座包含训练、分解／重建及原数值条件的历史；没有无训练反事实，不能把高秩尾部全部称为学习所得。SeqFT使用scale×B×A代数系数，不能宣称与原混合精度前向逐位等价。两方法LR、参数化、约束和位置仍混杂。', '',
              '此次没有证明任何参数无用，没有新F1、显著提升或端侧优势；结果用于否定未经检验的空间等价假设，不能直接删参数或填主表。', '',
              f"实际CPU两线程，耗时{e['seconds']:.1f}秒，进程最大RSS约{e['CPU_max_RSS_KiB']/1024:.1f}MiB；这是离线谱分析占用，不是训练或部署内存。", '',
              '[全部逐矩阵谱与源SHA](EVIDENCE.json) · [独立复算](INDEPENDENT_CHECK.json) · [事先协议](PROTOCOL.md)']
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(check))


if __name__ == '__main__': main()
