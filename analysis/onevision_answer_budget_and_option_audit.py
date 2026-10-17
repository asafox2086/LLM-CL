"""CPU audit of answer budgets and strict option-text/letter mismatch.

Auxiliary evaluation diagnosis only: never changes predictions or main scores.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    return re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', text.lower())


def options(context):
    matches = list(re.finditer(r'(?m)^([A-Z])\s*:\s*', context))
    if len(matches) < 2 or context[:matches[0].start()].strip():
        return None
    letters = [m.group(1) for m in matches]
    if letters != [chr(ord('A')+i) for i in range(len(letters))]:
        return None
    result = {m.group(1):context[m.end():matches[i+1].start() if i+1<len(matches) else len(context)].strip()
              for i,m in enumerate(matches)}
    values = [tuple(norm(t)) for t in result.values()]
    if any(not v for v in values) or len(set(values)) != len(values):
        return None
    return result


def stats(values):
    ordered = sorted(values)
    return {'QA':len(values), 'min':ordered[0], 'median':ordered[(len(values)-1)//2],
            'p95':ordered[math.ceil(.95*len(values))-1], 'max':ordered[-1],
            'greater_than_512':sum(n>512 for n in values)}


def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    sources = {}

    def read(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    model = ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
        path = model/name
        sources[str(path.relative_to(ROOT))] = sha(path)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest = read(path)
    assert sha(path) == '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    lengths = defaultdict(list)
    data = {}
    too_long = []
    for task in manifest['tasks']:
        for split, info in task['files'].items():
            path = ROOT/info['path']
            assert sha(path) == info['sha256']
            sources[str(path.relative_to(ROOT))] = info['sha256']
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == info['count']
            for row in rows:
                n = len(tokenizer(row['answer'],add_special_tokens=False)['input_ids'])
                lengths[task['domain'],split].append(n)
                data[row['id']] = dict(row, _audit_answer_tokens=n, _audit_task=task['task'])
                if n>512:
                    too_long.append({'id':row['id'],'domain':task['domain'],'split':split,
                                     'answer_tokens':n,'teacher_forced_scored_prefix':512,
                                     'configured_generation_budget':task['max_new_tokens']})
    assert sum(len(v) for (d,s),v in lengths.items() if s=='train') == 1313
    assert sum(len(v) for (d,s),v in lengths.items() if s=='test') == 21454
    assert all(x['split']=='test' and x['domain']=='traffic' for x in too_long)
    native = read(REPORT/'deterministic_native/complete.json')
    assert native['complete']
    run = ROOT/native['run']
    diagnostic = []
    for domain in ['natural','medical','geography','finance','traffic']:
        path = run/'evaluations/sequential_T19'/f'{domain}_image.jsonl'
        assert sha(path) == native['artifact_sha256'][str(path.relative_to(run))]
        sources[str(path.relative_to(ROOT))] = sha(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows)==100
        for row in rows:
            source = data[row['id']]
            n = source['_audit_answer_tokens']
            assert row['references']==source['answers']
            assert row['target_truncated']==(n>512)
            assert row['target_tokens']==min(n,512)+1
        generated = [row for row in rows if 'prediction' in row]
        reencoded = [len(tokenizer(row['prediction'],add_special_tokens=False)['input_ids']) for row in generated]
        diagnostic.append({'domain':domain,'NLL_QA':100,'NLL_target_truncated':sum(row['target_truncated'] for row in rows),
                           'generated_QA':len(generated),'EM_correct':sum(row['scores']['exact_match']==100 for row in generated),
                           'token_F1':sum(row['scores']['token_f1'] for row in generated)/len(generated),
                           'rougeL':sum(row['scores']['rougeL'] for row in generated)/len(generated),
                           'prediction_retokenized_lengths':reencoded,
                           'prediction_retokenized_length_reaches_budget':sum(n>=manifest['tasks'][row['native_task']-1]['max_new_tokens'] for row,n in zip(generated,reencoded)),
                           'stop_reason':'Original runner did not retain generated token IDs/EOS reason. Retokenized length is descriptive, not exact proof of stop cause.'})
    prior = read(REPORT/'FULL_COMPLETED_DOMAINS_AUDIT.json')
    full_run = ROOT/'exp/onevision_capacity_20261016/cap1_full_confirm/20261017_091853'
    path = full_run/'evaluations/final_full_input_router/finance_image.jsonl'
    assert sha(path)==prior['source_sha256'][str(path.relative_to(ROOT))]
    sources[str(path.relative_to(ROOT))]=sha(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = [json.loads(line) for line in (ROOT/manifest['tasks'][15]['files']['test']['path']).read_text().splitlines()]
    assert [r['id'] for r in rows] == [r['id'] for r in expected]
    counts=Counter();certificates=[];wrong_or_partial=[]
    for row in rows:
        source=data[row['id']]
        assert row['references']==source['answers']
        raw_exact=any(norm(row['prediction'])==norm(gold) for gold in source['answers'])
        assert row['scores']['exact_match']==100*raw_exact
        counts['raw_EM_correct']+=raw_exact
        parsed=options(source['context'])
        if parsed is None:
            counts['options_not_strictly_parseable']+=1
            continue
        assert source['context']==source['metadata']['options']
        counts['strict_options_parseable']+=1
        matches=[letter for letter,text in parsed.items() if norm(text)==norm(row['prediction'])]
        assert len(matches)<=1
        if not matches:
            continue
        counts['exact_unique_option_text_outputs']+=1
        matched=matches[0]
        if matched==source['answer'] and not raw_exact:
            certificates.append({'id':row['id'],'prediction':row['prediction'],
                                 'gold_letter':source['answer'],'matched_option_letter':matched,
                                 'option_text':parsed[matched], 'raw_EM':0,
                                 'condition':'Unchanged output equals unique gold-selected option text under the existing primary normalizer.'})
        else:
            wrong_or_partial.append({'id':row['id'],'matched_letter':matched,'gold':source['answer']})
    counts['certified_content_correct_but_letter_EM_wrong']=len(certificates)
    counts['matched_wrong_or_partial_or_already_correct']=len(wrong_or_partial)
    assert counts['exact_unique_option_text_outputs']==len(certificates)+len(wrong_or_partial)
    out={'CPU_only':True,'answer_budget_statistics':{d:{s:stats(lengths[d,s]) for s in ['train','test']} for d in ['natural','medical','geography','finance','traffic']},
         'original_test_answers_longer_than_loss_and_generation_budget':too_long,
         'deterministic_diagnostic':diagnostic,
         'full_finance':{'QA':2233,'counts':dict(counts),'raw_EM_percent':100*counts['raw_EM_correct']/2233,
                         'certified_format_penalty_pp':100*len(certificates)/2233,
                         'certificates':certificates,'other_exact_option_text_outputs':wrong_or_partial},
         'sources':sources,'audit_source_sha256':sha(Path(__file__)),
         'limitations':[
             'Retrospective auxiliary audit of previously inspected output; primary predictions/scores remain immutable. No new model accuracy or method gain.',
             'Strict option-text equivalence certifies these particular outputs, not general semantic correctness or all finance errors. Multiple-answer partial selection remains wrong.',
             'No test-based prompt/router/capacity tuning; parser uses input context only, then verifies its result against original gold.',
             'Canonical reference lengths within budget reject mandatory reference-length truncation, not all generation/termination problems.',
             'Raw generated IDs/EOS flags were not saved; output re-tokenization cannot establish exact stopping cause.',
             'Three long TEST references remain in full evaluation unchanged. Their teacher-forced NLL is prefix-limited; generation scores retain original full references.',
             'Neither textual overlap nor option correctness alone proves image use, expert knowledge sufficiency, or removal of continual-learning interference.',
             'Only completed full finance2233 and diagnostic500/155 are analyzed. Full traffic/21454 completion is not asserted.']}
    assert not torch.cuda.is_initialized()
    for name,digest in sources.items(): assert sha(ROOT/name)==digest
    (REPORT/'ANSWER_BUDGET_AND_OPTION_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    traffic=next(d for d in diagnostic if d['domain']=='traffic')
    lines=['# 原答案预算与选项格式：只读评价诊断','',
           '原始预测、主评分、原QA、训练和在跑GPU源码均不修改。此处是事后辅助诊断，不是新模型成绩。','',
           '## 1. 是否因为答案超过预算才得零分？','',
           '官方本地OneVision tokenizer逐条核对冻结1313训练与21454测试答案。1%全部训练答案≤512 tokens；交通205条训练最长418。500题诊断的NLL截断均为0；交通100题全部在512预算内。','',
           f"交通固定30生成题原EM {traffic['EM_correct']}/30，Token F1 {traffic['token_F1']:.3f}，ROUGE-L {traffic['rougeL']:.3f}。其中{traffic['prediction_retokenized_length_reaches_budget']}/30输出重新编码长度达到512；这是长度描述，不是原生成EOS/停止原因。不能把多数零EM归为标准答案必然超预算，也不能将词重合当全部内容正确。",'',
           '完整TEST仅3条交通答案超过512 tokens，最大560；仍保留在原测试中。原teacher-forced NLL只计前512加EOS，不能称其完整答案NLL；生成评分仍对完整原参考。三条ID与精确长度见JSON。','',
           '## 2. 金融字母答错，是否必然是不知道答案？','',
           f"已结束的金融完整2233题，原EM正确{counts['raw_EM_correct']}题（{100*counts['raw_EM_correct']/2233:.3f}%）。在{counts['strict_options_parseable']}条严格可解析且无归一化重复的输入选项中，有{len(certificates)}条原EM=0的输出，精确等于正确字母指向的唯一选项内容；例如输出数值、货币或公司名而不是A/B/C。",'',
           f"这{len(certificates)}条构成**明确的选项内容／字母格式差异证据**，相当于全金融{100*len(certificates)/2233:.3f}个百分点的原EM计分差额。不是LoRA训练提升，不据此替换主表，也不声称余下错题都是格式。全部{counts['exact_unique_option_text_outputs']}个唯一选项内容输出均核验，其余{len(wrong_or_partial)}条错误／部分选择／已正确也保留，禁止只展示有利例子。",'',
           '| 原ID | 原输出 | 原金标字母 | 精确匹配的选项字母 | 原EM |',
           '| --- | --- | --- | --- | ---: |']
    for c in certificates[:4]:
        lines.append(f"| {c['id']} | {c['prediction'].replace('|','/')} | {c['gold_letter']} | {c['matched_option_letter']} | 0 |")
    lines+=['','## 证明与未证明','',
            '已排除“所有金融零EM都代表选项内容错误”的解释；也排除“交通诊断标准答案整体超出解码预算”的解释。没有证明专业知识充分、图像被使用，或跨域干扰已解决。下一步匹配实验仍需同时报告原生成分数、答案概率与逐题转移。','',
            '未实验修改提示、增加解码长度或重训，因而“改进效果”仅限诊断解释清楚，无模型性能改进声明。','',
            '源文件SHA、3条长答案ID、全部格式证书与反例见[JSON](ANSWER_BUDGET_AND_OPTION_AUDIT.json)。']
    (REPORT/'ANSWER_BUDGET_AND_OPTION_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'finance':dict(counts),'long_test_answers':too_long,'traffic':traffic},ensure_ascii=False))


if __name__=='__main__':
    main()
