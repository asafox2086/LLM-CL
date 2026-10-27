"""Read only future complete paired-option VERIFIED/OUTCOME scalar results.

No scoring/tokenization, prediction/QA files, checkpoints, models or network.
Writes a new READOUT.json/FINDINGS.md; never edits producer results or tables.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ('repeat', 'paired')
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
VERIFIER = '030cc59d5d1ba3be7ddec30fd2ac73b0c7cc248fc4162f403ee01fb92188a5d3'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())


def finite(value):
    assert isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return value


def close(left, right):
    # Arithmetic consistency only; never a quality/acceptance tolerance.
    assert math.isclose(finite(left), finite(right), rel_tol=1e-12, abs_tol=1e-10)


def concentration(items):
    deltas = [finite(r['delta_F1']) for r in items]
    gain = math.fsum(d for d in deltas if d > 0)
    loss = math.fsum(-d for d in deltas if d < 0)
    gross = gain+loss
    return dict(QA=len(items), higher=sum(d > 0 for d in deltas), lower=sum(d < 0 for d in deltas),
        tie=sum(d == 0 for d in deltas), gross_gain_F1_sum=gain, gross_loss_F1_sum=loss,
        net_F1_sum=math.fsum(deltas), gross_absolute_F1_sum=gross,
        gross_gain_macro_pp=gain/100, gross_loss_macro_pp=loss/100, net_macro_pp=math.fsum(deltas)/100,
        top1_share_of_gross_gain=max((d for d in deltas if d > 0), default=0)/gain if gain else None,
        top1_share_of_gross_loss=max((-d for d in deltas if d < 0), default=0)/loss if loss else None,
        absolute_contribution_HHI=math.fsum((abs(d)/gross)**2 for d in deltas) if gross else None,
        no_share_divided_by_net_gain=True,
        concentration_scope='Descriptive fixed QA differences; no significance, causal localization or new decision gate')


def paired_phase(quality, phase):
    left, right = quality['repeat'][phase], quality['paired'][phase]
    assert left['QA'] == right['QA'] == 100
    a, b = left['per_id'], right['per_id']
    assert len(a) == len(b) == 100
    assert [r['id_sha256'] for r in a] == [r['id_sha256'] for r in b]
    assert len({r['id_sha256'] for r in a}) == 100
    items = []
    for repeat, paired in zip(a, b):
        assert repeat['task'] == paired['task'] and repeat['domain'] == paired['domain']
        assert 0 <= finite(repeat['F1']) <= 100 and 0 <= finite(paired['F1']) <= 100
        delta = paired['F1']-repeat['F1']
        items.append(dict(id_sha256=repeat['id_sha256'], task=repeat['task'], domain=repeat['domain'],
            repeat_F1=repeat['F1'], paired_F1=paired['F1'], delta_F1=delta, macro_pp_contribution=delta/100,
            direction='higher' if delta > 0 else 'lower' if delta < 0 else 'tie'))
    domains = {}
    for domain in DOMAINS:
        rows = [r for r in items if r['domain'] == domain]
        assert len(rows) == left['domains'][domain]['QA'] == right['domains'][domain]['QA'] == 20
        delta = right['domains'][domain]['F1']-left['domains'][domain]['F1']
        close(math.fsum(r['delta_F1'] for r in rows)/20, delta)
        domains[domain] = dict(repeat_F1=left['domains'][domain]['F1'], paired_F1=right['domains'][domain]['F1'],
            paired_minus_repeat_F1=delta, contributions=concentration(rows))
    macro = right['domain_macro_F1']-left['domain_macro_F1']
    close(math.fsum(r['delta_F1'] for r in items)/100, macro)
    return dict(repeat_macro_F1=left['domain_macro_F1'], paired_macro_F1=right['domain_macro_F1'],
        paired_minus_repeat_macro_pp=macro, domains=domains, contributions=concentration(items),
        per_id_sorted_by_absolute_contribution=sorted(items, key=lambda r: (-abs(r['delta_F1']), r['id_sha256'])),
        weights='Each original QA contributes delta_F1/100 macro pp; five domains each have20 QA')


def strict_diagnostic(value):
    result, by_id = {}, {}
    for arm in ARMS:
        source = value[arm]; counts = source['counts']; rows = source['per_id']
        assert counts['QA'] == len(rows) == 12 and len({r['id_sha256'] for r in rows}) == 12
        assert source['cyclic_F1_calls'] == 0 and source['invalid_pairs_never_count_as_stable']
        assert counts['both_mappable']+counts['both_unmappable']+counts['only_one_mappable'] == 12
        assert counts['same_content']+counts['changed_content'] == counts['both_mappable']
        assert counts['stable_strict_correct']+counts['stable_strict_wrong'] == counts['same_content']
        assert counts['both_mappable'] == sum(r['both_mappable'] for r in rows)
        assert counts['same_content'] == sum(r['same_content'] for r in rows)
        for row in rows:
            assert not row['same_content'] or row['both_mappable']
        by_id[arm] = {r['id_sha256']:r for r in rows}
        result[arm] = dict(counts=counts, full_denominator=12,
            not_both_mappable=counts['both_unmappable']+counts['only_one_mappable'],
            stable_and_strict_correct=counts['stable_strict_correct'], stable_and_strict_wrong=counts['stable_strict_wrong'],
            per_id=rows, strict_mapping=source['strict_mapping'], scope=source['scope'])
    assert set(by_id['repeat']) == set(by_id['paired'])
    common = [i for i in by_id['repeat'] if all(by_id[a][i]['both_mappable'] for a in ARMS)]
    result['common_both_mappable'] = dict(QA=len(common),
        id_sha256=common, arms={a:dict(same_content=sum(by_id[a][i]['same_content'] for i in common),
            stable_strict_correct=sum(by_id[a][i]['same_content'] and by_id[a][i]['original_strict_correct'] for i in common),
            stable_strict_wrong=sum(by_id[a][i]['same_content'] and not by_id[a][i]['original_strict_correct'] for i in common)) for a in ARMS},
        descriptive_only=True)
    return result


def derive(outcome):
    quality = outcome['quality']
    assert set(quality) == set(ARMS)
    phases = {p:paired_phase(quality, p) for p in ('acquired', 'final')}
    for phase, result in phases.items():
        prior = outcome['comparisons']['paired_minus_repeat'][phase]
        close(result['paired_minus_repeat_macro_pp'], prior['domain_macro_F1'])
        for domain in DOMAINS: close(result['domains'][domain]['paired_minus_repeat_F1'], prior['domains'][domain])
    t16 = {}
    for arm in ARMS:
        before, acquired = quality[arm]['before']['tasks']['16'], quality[arm]['acquired']['tasks']['16']
        assert before['QA'] == acquired['QA'] == 20
        t16[arm] = dict(QA=20, before_F1=before['F1'], acquired_F1=acquired['F1'],
            acquired_minus_before=acquired['F1']-before['F1'])
    t16['paired_minus_repeat'] = dict(before_F1=t16['paired']['before_F1']-t16['repeat']['before_F1'],
        acquired_F1=t16['paired']['acquired_F1']-t16['repeat']['acquired_F1'],
        change_difference=t16['paired']['acquired_minus_before']-t16['repeat']['acquired_minus_before'])
    prefix = outcome['preintervention_prefix']
    assert [s['stage'] for s in prefix['stages']] == list(range(1,16))
    assert prefix['no_forced_prefix_equality_gate'] and prefix['additional_model_or_score_calls'] == 0
    resources = {a:{k:outcome['resources'][a][k] for k in (
        'training_seconds','eval_seconds','training_monitor_includes_212_stage_readouts','actual_final_layout',
        'checkpoint_file_bytes','checkpoint_sha256','fixed_window_max_peak_allocated','fixed_window_max_peak_reserved','scope')} for a in ARMS}
    return dict(phase_comparisons=phases,T16_acquisition=t16,preintervention_prefix=prefix,
        strict_option_diagnostic=strict_diagnostic(outcome['strict_option_readout']), resources=resources,
        interpretation_limits=[
            'Raw original Answer F1 and strict A-D mapping/correctness are distinct readouts.',
            'An original-F1 gain alone supports the paired-supervision package, not repaired content-position binding.',
            'More same-content selections alone can mean stable wrong answers; full12 and common-mappable denominators are both retained.',
            'Final-only gains do not establish improved new acquisition; before/acquired refer to task-specific model times.',
            'All positive/negative/tie QA differences are retained. Concentration shares use gross gain/loss, never net gain.',
            'No new threshold, significance test, acceptance gate or formal/Ours claim; this is explored DEV, one seed.'])


def markdown(value):
    rows = ['# 成对选项监督：完整结果读数', '',
        '仅派生已封600条原样F1与24条严格映射，不读取原始QA、预测文件或CP，不新增评分。', '',
        '| 域 | 刚习得 paired−repeat（pp） | 最终 paired−repeat（pp） |', '|---|---:|---:|']
    for d in DOMAINS:
        rows.append(f"| {d} | {value['phase_comparisons']['acquired']['domains'][d]['paired_minus_repeat_F1']:+.6f} | {value['phase_comparisons']['final']['domains'][d]['paired_minus_repeat_F1']:+.6f} |")
    rows += ['', '各域20题；每题F1差÷100是五域宏的pp贡献。所有逐题正负差在READOUT.json保留。']
    for phase in ('acquired','final'):
        c = value['phase_comparisons'][phase]['contributions']
        rows += [f"{phase}：宏净差{c['net_macro_pp']:+.6f} pp；gross gain {c['gross_gain_macro_pp']:.6f}、gross loss {c['gross_loss_macro_pp']:.6f} pp；{c['higher']}高/{c['lower']}低/{c['tie']}平。集中度只按gross计算，不用接近零的净差作分母。"]
    rows += ['', '| 臂 | 全分母 | 双可映射 | 稳定且严格正确 | 稳定且严格错误 | 未双可映射 |', '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        s = value['strict_option_diagnostic'][arm]
        rows.append(f"| {arm} | 12 | {s['counts']['both_mappable']} | {s['stable_and_strict_correct']} | {s['stable_and_strict_wrong']} | {s['not_both_mappable']} |")
    t16 = value['T16_acquisition']['paired_minus_repeat']; prefix = value['preintervention_prefix']
    rows += ['', f"T16前F1差{t16['before_F1']:+.6f} pp，刚习得差{t16['acquired_F1']:+.6f} pp，阶段变化差{t16['change_difference']:+.6f} pp。",
        f"前15阶段adapter摘要相同：{sum(s['adapter_digest_equal'] for s in prefix['stages'])}/15；S16前输出比较原样复制：{json.dumps(prefix['S16_before'],ensure_ascii=False)}。差异只报告，不新增失败门。", '',
        '原样F1提升本身只支持监督增强整包；绑定改善必须结合固定12题的映射覆盖、稳定且正确/稳定且错及共同可映射分母解释。稳定率提高不等于正确，原F1满分也不替代严格字母判定。仅最终提高不能称更会新习得。', '',
        '这些是已探索DEV的单seed描述；无显著性、等效性、正式胜强基线或Ours判定。完整权重与五窗GPU值只复制已核验结果，不将本派生称新资源测量。', '']
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args(); run = args.run.resolve(); output = args.output.resolve()
    assert args.run.is_absolute() and args.output.is_absolute() and output != run
    required = [run/'VERIFIED.json',run/'OUTCOME.json']
    if not all(p.is_file() for p in required):
        raise SystemExit('Not consumed: complete VERIFIED.json and OUTCOME.json are required; no partial results read.')
    assert not output.exists() and output.parent.is_dir(), 'New output directory only; never overwrite results'
    verified,outcome = map(read,required)
    assert verified['status'] == 'VERIFIED_ONCE' and verified['verifier_sha256'] == VERIFIER
    assert verified['all_four_phases_real_exit0'] and verified['scoring_calls'] == outcome['scored_existing_answers'] == 600
    assert verified['cyclic_F1_calls'] == outcome['cyclic_F1_calls'] == 0
    assert verified['strict_cyclic_mapping_readouts'] == outcome['strict_cyclic_mapping_readouts'] == 24
    assert verified['outcome_sha256'] == sha(required[1])
    assert outcome['status'] == 'COMPLETE_PAIRED_OPTION_EXPLORATORY_RESULT'
    assert outcome['model_generations'] == 634 and outcome['FIT_backwards'] == 2426
    assert outcome['optimizer_updates'] == 174 and outcome['supervised_forwards'] == 2492
    value = derive(outcome)
    value['provenance'] = dict(VERIFIED_sha256=sha(required[0]),OUTCOME_sha256=sha(required[1]),
        consumer_sha256=sha(Path(__file__)),new_models=0,new_generations=0,new_scores=0,
        only_input_files=['VERIFIED.json','OUTCOME.json'])
    text = markdown(value)
    output.mkdir(exist_ok=False)
    with (output/'READOUT.json').open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    with (output/'FINDINGS.md').open('x',encoding='utf8') as stream: stream.write(text)
    print(json.dumps(dict(complete=True,output=str(output),new_scores=0,new_models=0)))


if __name__ == '__main__': main()
