"""CPU-only, same-ID contextual comparison; never mixes this into formal ranking."""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer_f1 import answer_f1

DOMAINS = ['natural', 'medical', 'geography', 'finance', 'traffic']
BASE = 'summary/onevision_diagnostics_20261016'
INDEX = 'summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/VERIFIED_INDEX.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]


def aggregate(rs):
    tasks, domains = defaultdict(list), defaultdict(list)
    for r in rs:
        tasks[r['native_task']].append(r['F1'])
        domains[r['domain']].append(r['F1'])
    assert set(tasks) == set(range(1, 20)) and set(domains) == set(DOMAINS)
    return dict(domains={d: sum(domains[d])/len(domains[d]) for d in DOMAINS},
                AP=sum(sum(v)/len(v) for v in tasks.values())/19,
                domain_macro=sum(sum(v)/len(v) for v in domains.values())/5)


def cohort(root):
    ref = read(root/BASE/'warm_quality_20261018/predeclared_hard_case/REFERENCE_COMPARISON.json')
    native = read(root/BASE/'deterministic_native/complete.json')
    attempt = read(root/BASE/'warm_quality_20261018/pilot_attempt.json')
    out = dict(CPU_only=True, no_new_inference=True, cohort_selection='existing frozen155, no score selection',
               sources={}, research_rows={}, baseline_proofs=[])
    for name, run, label in [('native_SeqFT', native['run'], 'sequential_T19'),
                             ('A_plus_B', attempt['run'], 'quality_hard')]:
        rr = []
        for d in DOMAINS:
            p = root/run/f'evaluations/{label}/{d}_image.jsonl'
            assert sha(p) == ref['input_sha256'][str(p.relative_to(root))]
            out['sources'][str(p.relative_to(root))] = sha(p)
            for r in rows(p):
                if 'prediction' in r:
                    f = answer_f1(r['prediction'], r['references'])
                    assert f == r['scores']['token_f1']
                    rr.append(dict(id=r['id'], native_task=r['native_task'], domain=d,
                                   references=r['references'], F1=f))
        assert len(rr) == 155 and len({r['id'] for r in rr}) == 155
        out['research_rows'][name] = rr
    assert [(r['id'], r['references']) for r in out['research_rows']['native_SeqFT']] == [
        (r['id'], r['references']) for r in out['research_rows']['A_plus_B']]
    index = read(root/INDEX)
    for item in index['methods']:
        if item['backbone'] != 'onevision':
            continue
        p = root/item['proof_file']
        proof = read(p)
        assert proof['final_QA'] == 21454 and proof['rescored_cells'] == 74
        assert proof['positive_finite_training_stages'] == 19
        out['baseline_proofs'].append(dict(path=item['proof_file'], sha256=sha(p), proof=proof))
    assert out['baseline_proofs']
    return out


def collect(root, bundle):
    wanted = {r['id']: r for r in bundle['research_rows']['A_plus_B']}
    results = []
    for item in bundle['baseline_proofs']:
        proof = item['proof']; run = root/proof['candidate']['run']
        assert sha(run/'config.json') == proof['config_sha256']
        # This validates the exact metric implementation on both hosts.
        assert sha(root/'analysis/answer_f1.py') == proof['source_sha256']['analysis/answer_f1.py']
        selected, full, inputs = [], [], {}
        for t in range(1, 20):
            rel = f'evaluations/stage_19/T{t:02}_test_predictions.jsonl'
            p = run/rel
            assert sha(p) == proof['prediction_sha256'][rel]
            inputs[str(p.relative_to(root))] = sha(p)
            for r in rows(p):
                f = answer_f1(r['prediction'], r['references'])
                assert f == r['scores']['token_f1']
                # Domain is determined by frozen task, not predicted/selected from answers.
                d = 'natural' if t == 1 else 'medical' if t <= 11 else 'geography' if t <= 15 else 'finance' if t == 16 else 'traffic'
                full.append(dict(native_task=t, domain=d, F1=f))
                if r['id'] in wanted:
                    w = wanted[r['id']]
                    assert w['references'] == r['references'] and w['native_task'] == t and w['domain'] == d
                    selected.append(dict(id=r['id'], native_task=t, domain=d, F1=f))
        assert len(full) == 21454 and len(selected) == 155 and {r['id'] for r in selected} == set(wanted)
        full_summary = aggregate(full)
        assert abs(full_summary['AP']-proof['aggregate']['AP']) < 1e-9
        assert all(abs(full_summary['domains'][d]-proof['domains'][d]['F1']) < 1e-9 for d in DOMAINS)
        results.append(dict(method=proof['candidate']['method'], cohort=aggregate(selected), rows=selected,
                            full=full_summary, proof_file=item['path'], proof_sha256=item['sha256'],
                            prediction_sha256=inputs))
    return dict(CPU_only=True, GPU_operations=0, optimizer_updates=0, new_predictions=0,
                baseline_subset_QA=155, results=results)


def merge(bundle, baseline):
    result = dict(CPU_only=True, GPU_operations=0, new_predictions=0,
                  evaluation_QA=155, same_IDS_gold_metric=True, formal_ranking=False,
                  limitations=['Research nativeAnyRes/BF16 differs from formalV2 image cap401408/FP16.',
                               'Same IDs alone do not establish matched training/inference protocol.',
                               '155 diagnostic scores cannot be compared to full21454 scores.',
                               'No TEST selection or hyperparameter changes; no full FWT/BWT from this subset.'],
                  research={k: aggregate(v) for k,v in bundle['research_rows'].items()},
                  baseline=baseline, input_sources=bundle['sources'],
                  report_source_sha256=sha(Path(__file__)))
    lines = ['# 已完成基线与A＋B：同155题的描述性比较', '',
             'CPU从已封存预测重算；原ID、金标及Answer F1一致，未新增推理、训练或测试选参。自然10／医学73／遥感32／金融10／交通30题，AP为19原生任务等权。', '',
             '**设置有差异：研究组为原生AnyRes/BF16，正式V2为公共401408像素/FP16。下表用于先看同题表现，不能作为公平胜出的正式主表。**', '',
             '| 方法 | 设置 | 自然 | 医学 | 遥感 | 金融 | 交通 | AP |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    table = [(r['method'], '正式V2', r['cohort']) for r in baseline['results']]
    table += [(k, '研究原生', v) for k,v in result['research'].items()]
    for name, setting, s in table:
        lines.append('| '+name+' | '+setting+' | '+' | '.join(f'{s["domains"][d]:.3f}' for d in DOMAINS)+f' | {s["AP"]:.3f} |')
    lines += ['', 'A＋B相对同设置原生SeqFT的诊断AP差为+0.803点；已有域宏差95%区间[-0.760,2.872]含0，尚未证实稳定或大幅优势。跨设置差值不作方法因果归因。', '',
              '该155题上A＋B的AP点估计高于两项正式基线，但设置不同；而且PaRSP在完整测试AP40.390略高于正式SeqFT40.286，在155题中却低于SeqFT，说明诊断子集不能代替完整测试排名。', '',
              '正式完整21454题主表保留原核验结果；A＋B该格继续留空，须同正式协议完整评测后再排序。此子集不补全FWT/BWT。', '',
              '逐题子集分数、全部最终预测SHA与原完成证明见 `EVIDENCE.json`；没有使用中间阶段或未完成方法充当最终结果。']
    return result, '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cohort','collect','merge'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--cohort',type=Path);p.add_argument('--baseline',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='cohort': value=cohort(a.root)
    elif a.mode=='collect': value=collect(a.root,read(a.cohort))
    else:
        value, md=merge(read(a.cohort),read(a.baseline))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.with_suffix('.md').write_text(md,encoding='utf8')
    a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(mode=a.mode, output=str(a.output), CPU_only=True)))


if __name__=='__main__': main()
