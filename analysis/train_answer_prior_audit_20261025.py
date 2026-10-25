"""One explicit, stdlib-only audit of selected frozen TRAIN files; never TEST.

LLMCL_ROOT is mandatory. Optional LLMCL_PRIOR_AUDIT_REPORT selects a new output
directory. No data or model code is imported; importing this module does no I/O.
"""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata


POOLS = {
    'pct01': (
        'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json',
        '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777', 1313),
    'pct10': (
        'data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json',
        '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7', 13105),
}
EXPERIMENT = 'five_domain_native_cl_20261016_sparse'
REPORT_REL = 'summary/train_answer_prior_20261025'
FINANCE_TYPES = {'single_choice', 'multiple_choice', 'numerical'}
MEDICAL_TYPES = {'Modality', 'Position', 'Organ', 'Size', 'Abnormality',
                 'Quantity', 'Plane', 'Shape', 'Color', 'KG'}
YES_NO = {'Yes', 'No', 'yes', 'no', 'YES', 'NO', '是', '否'}
DESCRIPTIVE_FLAGS = {'question_collection_rubric_present'}
COLLECTION_RUBRICS = (
    r'\b(?:which|what)\s+(?:of\s+)?(?:the\s+)?following\b',
    r'\b(?:select|choose|identify)\s+(?:all\s+)?(?:of\s+)?(?:the\s+)?following\b',
    r'下列(?:哪(?:一|些)?(?:项|个|种)?|各(?:项|个|种)?)?',
    r'以下哪(?:一|些)?(?:项|个|种)?',
)
STRUCTURAL_FLAGS = {
    'question_letter_reference': (
        r'\b(?:option|choice|answer|statement)s?\s*(?:\([A-Z]\)|[A-Z]\b)',
        r'\b[A-Z]\s*(?:and|or|through|to|,)\s*[A-Z]\b',
        r'(?:选项|答案|陈述)\s*[A-Z]', r'\([A-Z]\)'),
    'option_letter_reference': (
        r'\b(?:option|choice|answer|statement)s?\s*(?:\([A-Z]\)|[A-Z]\b)',
        r'\b[A-Z]\s*(?:and|or|through|to|,)\s*[A-Z]\b',
        r'(?:选项|答案|陈述)\s*[A-Z]', r'\([A-Z]\)'),
    'all_or_none_of': (
        r'\b(?:all|none|both|neither|either)\s+(?:of\s+)?(?:the\s+)?'
        r'(?:above|below|following|preceding|these|options|choices|statements)\b',
        r'\b(?:all|none)\s+of\b', r'^\s*(?:all|none|both|neither)\s*[.!]?\s*$',
        r'以上|以下|上述|下述|均不|全不|都不|皆不|均是|都是|全部正确'),
    'order_or_position_semantics': (
        r'\b(?:above|below|preceding|former|latter|first|second|third|'
        r'fourth|last|ordered|ordering|order|sequence|respectively|ascending|descending)\b',
        r'前者|后者|上面|下面|第[一二三四五六七八九十]|顺序|次序|依次|分别|升序|降序'),
}


def check(condition, code):
    if not condition:
        raise ValueError(code)


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf8')


def write_json(path, value):
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def rooted(root, relative):
    p = Path(relative)
    check(not p.is_absolute() and '..' not in p.parts, 'nonrelative_source_path')
    return root / p


def meta_key(value, allowed):
    """Never print an unexpected metadata value, which might contain free text."""
    if isinstance(value, str) and value in allowed:
        return value
    return 'UNKNOWN_SHA256:' + digest_bytes(canonical(value))


def safe_answer(answer):
    # Exact case and encoding are retained; only these short original labels
    # are displayable. All frequency keys always use the original string SHA.
    if answer in YES_NO or re.fullmatch(r'[A-Z]', answer):
        return answer
    return None


def parse_options(context):
    """Only the source-confirmed line-leading A: / B: syntax is supported."""
    flags = set()
    if not isinstance(context, str):
        return None, {'options_context_type_unknown'}
    if not context.strip():
        return None, {'options_absent'}
    matches = list(re.finditer(r'(?m)^([A-Z]):[ \t]*', context))
    if len(matches) < 2 or context[:matches[0].start()].strip():
        return None, {'options_unrecognized_syntax'}
    labels = [m.group(1) for m in matches]
    if len(labels) != len(set(labels)):
        flags.add('options_duplicate_labels')
    if labels != [chr(65 + i) for i in range(len(labels))]:
        flags.add('options_nonconsecutive_labels')
    contents = [context[m.end():matches[i + 1].start()
                        if i + 1 < len(matches) else len(context)].strip()
                for i, m in enumerate(matches)]
    if any(not x for x in contents):
        flags.add('options_empty_content')
    normalized = [' '.join(unicodedata.normalize('NFKC', x).casefold().split())
                  for x in contents]
    if len(set(normalized)) != len(normalized):
        flags.add('options_duplicate_content')
    # An unfamiliar marker inside a purported option is not silently swallowed.
    possible_markers = re.finditer(
        r'(?m)^[ \t]*(?:\([A-Za-z]\)|[A-Za-z][ \t]*[:.)：])', context)
    strict_starts = {m.start() for m in matches}
    if any(m.start() not in strict_starts for m in possible_markers):
        flags.add('options_mixed_or_unknown_marker_syntax')
    return (list(zip(labels, contents)) if not flags else None), flags


def semantic_flags(question, options):
    flags = set()
    clean_question = question
    if re.search(r'\bfollowing\b', question, re.IGNORECASE) or any(
            re.search(p, question, re.IGNORECASE) for p in COLLECTION_RUBRICS):
        flags.add('question_collection_rubric_present')
    for pattern in COLLECTION_RUBRICS:
        clean_question = re.sub(pattern, ' ', clean_question, flags=re.IGNORECASE)
    for name, patterns in STRUCTURAL_FLAGS.items():
        if name == 'question_letter_reference':
            texts = [question]
        elif name == 'option_letter_reference':
            texts = [v for _, v in options]
        else:
            texts = [clean_question] + [v for _, v in options]
        if any(re.search(pattern, text, re.IGNORECASE) for text in texts for pattern in patterns):
            flags.add(name)
    # "Following" in the question is a descriptive rubric observation. Within
    # an option it can refer to another option, so retain the conservative risk.
    if any(re.search(r'\bfollowing\b', v, re.IGNORECASE) for _, v in options):
        flags.add('order_or_position_semantics')
    return flags


def describe(row, task):
    check(isinstance(row, dict), 'row_not_object')
    for key in ('id', 'dataset', 'answer'):
        check(isinstance(row.get(key), str), 'row_field_type:' + key)
    refs, metadata = row.get('answers'), row.get('metadata')
    check(isinstance(refs, list) and bool(refs)
          and all(isinstance(x, str) for x in refs), 'invalid_references')
    check(bool(row['answer'].strip()) and row['answer'] == refs[0], 'invalid_primary_target')
    check(isinstance(metadata, dict) and bool(row['id']), 'invalid_id_or_metadata')
    flags = set()
    if not isinstance(row.get('question'), str):
        flags.add('question_field_type_unknown')
    if not isinstance(row.get('context'), str):
        flags.add('context_field_type_unknown')
    if any(not x.strip() for x in refs):
        flags.add('empty_reference_values')
    if len(refs) != 1:
        flags.add('multiple_reference_values')
    if len(set(refs)) != len(refs):
        flags.add('duplicate_reference_values')
    domain = task['domain']
    check(row['dataset'] == ('FinMME' if domain == 'finance' else 'SLAKE'),
          'unexpected_dataset')
    info = dict(id_sha256=digest_bytes(row['id'].encode()), task=task['task'], domain=domain,
                answer_sha256=digest_bytes(row['answer'].encode()),
                safe_original_answer=safe_answer(row['answer']),
                references_sha256=digest_bytes(canonical(refs)),
                reference_count=len(refs), reference_set_sha256=digest_bytes(canonical(sorted(set(refs)))),
                question_type='', answer_type='not_applicable', q_lang='not_applicable',
                options_count=None, original_option_labels=None,
                original_choice_members=None, choice_set=None,
                no_predefined_permutation_obstacle=False)
    if domain == 'medical':
        info['question_type'] = meta_key(metadata.get('content_type'), MEDICAL_TYPES)
        info['answer_type'] = meta_key(metadata.get('answer_type'), {'OPEN', 'CLOSED'})
        info['q_lang'] = meta_key(metadata.get('q_lang'), {'en', 'zh'})
        if any(str(info[k]).startswith('UNKNOWN') for k in ('question_type', 'answer_type', 'q_lang')):
            flags.add('unrecognized_medical_metadata')
        if metadata.get('content_type') != task.get('native_label'):
            flags.add('medical_content_type_task_mismatch')
        if row['context'] or metadata.get('options') not in (None, ''):
            flags.add('unexpected_medical_options_or_context')
        if info['answer_type'] == 'CLOSED' and row['answer'] not in YES_NO:
            flags.add('closed_target_not_yes_no')
        # No artificial MCQ or option ordering is inferred for medical rows.
    else:
        kind = metadata.get('question_type')
        info['question_type'] = meta_key(kind, FINANCE_TYPES)
        if not isinstance(kind, str) or kind not in FINANCE_TYPES:
            flags.add('unrecognized_finance_question_type')
        if not isinstance(metadata.get('options'), str):
            flags.add('options_metadata_type_unknown')
        elif metadata['options'] != row['context']:
            flags.add('context_metadata_options_mismatch')
        target = row['answer']
        if kind in ('single_choice', 'multiple_choice'):
            valid = (kind == 'single_choice' and bool(re.fullmatch(r'[A-Z]', target))) or (
                kind == 'multiple_choice' and bool(re.fullmatch(r'[A-Z]{2,26}', target)))
            if not valid:
                flags.add('unrecognized_choice_target_encoding')
            elif len(target) != len(set(target)):
                flags.add('choice_target_repeated_label')
            else:
                # Preserve original contiguous-letter target even if input
                # options cannot be parsed; membership validation is separate.
                info['original_choice_members'] = list(target)
                info['choice_set'] = sorted(target)
                info['safe_original_answer'] = target
        options, parser_flags = parse_options(row['context'])
        flags.update(parser_flags)
        if options is not None:
            labels = [x[0] for x in options]
            info['options_count'] = len(labels)
            info['original_option_labels'] = labels
            if isinstance(row.get('question'), str):
                flags.update(semantic_flags(row['question'], options))
            if info['choice_set'] is not None and not set(info['choice_set']).issubset(labels):
                flags.add('choice_target_outside_option_labels')
            if kind in ('single_choice', 'multiple_choice') and not (flags - DESCRIPTIVE_FLAGS):
                info['no_predefined_permutation_obstacle'] = True
        if kind == 'numerical':
            flags.add('numerical_not_choice_permutation_candidate')
    info['flags'] = sorted(flags)
    return info


def aggregate(items):
    n = len(items)
    answers = Counter(x['answer_sha256'] for x in items)
    refs = Counter(x['references_sha256'] for x in items)
    refsets = Counter(x['reference_set_sha256'] for x in items)
    displays = {x['answer_sha256']: x['safe_original_answer'] for x in items}
    flags = Counter(f for x in items for f in x['flags'])
    choices = Counter(''.join(x['choice_set']) for x in items if x['choice_set'] is not None)
    options = Counter(str(x['options_count']) for x in items)
    qualified = sum(x['no_predefined_permutation_obstacle'] for x in items)
    return dict(QA=n, unique_original_answers=len(answers),
                largest_original_answer_count=max(answers.values(), default=0),
                largest_original_answer_fraction=max(answers.values(), default=0) / n if n else None,
                answer_frequencies=[dict(answer_sha256=k, safe_original_answer=displays[k], count=v)
                                    for k, v in sorted(answers.items())],
                original_reference_list_frequencies=dict(sorted(refs.items())),
                original_reference_set_frequencies=dict(sorted(refsets.items())),
                reference_counts=dict(sorted(Counter(str(x['reference_count']) for x in items).items())),
                parsed_choice_set_frequencies=dict(sorted(choices.items())),
                original_options_count_frequencies=dict(sorted(options.items())),
                flag_counts=dict(sorted(flags.items())),
                any_flag_QA=sum(bool(x['flags']) for x in items),
                any_obstacle_flag_QA=sum(bool(set(x['flags']) - DESCRIPTIVE_FLAGS) for x in items),
                no_predefined_permutation_obstacle_QA=qualified,
                no_predefined_permutation_obstacle_fraction=qualified / n if n else None)


def load_pool(root, condition, opened):
    rel, expected, total = POOLS[condition]
    raw = rooted(root, rel).read_bytes()
    check(digest_bytes(raw) == expected, 'manifest_sha:' + condition)
    opened[rel] = expected
    manifest = json.loads(raw)
    check(manifest['experiment_id'] == EXPERIMENT and manifest['condition'] == condition
          and manifest['seed'] == 42 and manifest['formal_ready']
          and not manifest['assets_pending'], 'manifest_identity:' + condition)
    tasks = manifest['tasks']
    check([x['task'] for x in tasks] == list(range(1, 20)), 'manifest_tasks')
    check(sum(x['files']['train']['count'] for x in tasks) == total, 'manifest_train_total')
    selected = [x for x in tasks if x['domain'] in ('medical', 'finance')]
    check([x['task'] for x in selected] == list(range(2, 12)) + [16], 'selected_task_mapping')
    check(tasks[15]['files']['train']['count'] == (89 if condition == 'pct01' else 887),
          'formal_finance_count')
    rows, content, fileproofs = [], {}, []
    for task in selected:
        info = task['files']['train']  # Never open or hash files.test.
        rel = info['path']
        check(Path(rel).name == 'T%02d_train.jsonl' % task['task'], 'not_selected_train_path')
        blob = rooted(root, rel).read_bytes()
        check(digest_bytes(blob) == info['sha256'], 'train_sha:' + condition)
        opened[rel] = info['sha256']
        lines = blob.splitlines()
        check(len(lines) == info['count'] and all(lines), 'train_count_or_blank_line')
        task_ids = []
        for line in lines:
            source = json.loads(line)
            entry = describe(source, task)
            key = entry['id_sha256']
            check(key not in content, 'duplicate_selected_train_id')
            content[key] = digest_bytes(canonical(source))
            rows.append(entry)
            task_ids.append(key)
        fileproofs.append(dict(task=task['task'], domain=task['domain'], path=rel,
                               count=len(lines), sha256=info['sha256'],
                               ordered_id_hashes_sha256=digest_bytes(canonical(task_ids))))
    by_domain, by_task, strata = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        by_domain[row['domain']].append(row)
        by_task['T%02d' % row['task']].append(row)
        # Keep question type, language, open/closed and option count explicit.
        key = '|'.join([row['domain'], 'T%02d' % row['task'], row['question_type'],
                        row['answer_type'], row['q_lang'], str(row['options_count'])])
        strata[key].append(row)
    return dict(manifest_sha256=expected, declared_whole_pool_train_QA=total,
                audited_train_QA=len(rows), files=fileproofs,
                by_domain={k: aggregate(v) for k, v in sorted(by_domain.items())},
                by_task={k: aggregate(v) for k, v in sorted(by_task.items())},
                by_stratum={k: aggregate(v) for k, v in sorted(strata.items())},
                rows=rows), content


def findings(result):
    lines = ['# 1%与10% TRAIN答案先验／选项结构：描述结果', '',
             '只读正式两池的医学／金融TRAIN；没有读取TEST文件、模型、预测或评分。', '',
             '| 池 | 域 | 审计QA | 原答案种数 | 原字符串最大占比 | 未发现预定排列障碍QA |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for pool, value in result['conditions'].items():
        for domain, s in value['by_domain'].items():
            qualifier = str(s['no_predefined_permutation_obstacle_QA']) if domain == 'finance' else '不适用'
            lines.append('| %s | %s | %d | %d | %.6f | %s |' % (
                pool, domain, s['QA'], s['unique_original_answers'],
                s['largest_original_answer_fraction'], qualifier))
    lines += ['', '完整分母、原答案SHA频数、多选原集合、数值分层、解析失败／多参考与保守风险flags见[AUDIT.json](AUDIT.json)。', '',
              '本次计数不判断偏斜是否充分造成选择塌缩；均衡也不能排除预训练token先验或条件位置偏好。',
              '1%与10%并列只是监督组成比较，不能据此判断10%刚习得功能是否改善。后者须等待各task刚习得表现及金融选择集中度的独立合法证明。',
              '“未发现预定排列障碍”不是语义可逆认证；医学未制造选项。没有自动模型、prior correction或prompt修改准入。', '']
    return '\n'.join(lines)


def main():
    root = Path(os.environ['LLMCL_ROOT']).resolve()
    check(root.is_dir(), 'missing_root')
    report = Path(os.environ.get('LLMCL_PRIOR_AUDIT_REPORT', str(root / REPORT_REL)))
    report.mkdir(parents=True, exist_ok=True)
    check(not any((report / name).exists() for name in
                  ('STARTED.json', 'AUDIT.json', 'COMPLETE.json', 'FAILURE.json', 'FINDINGS.md')),
          'audit_output_already_exists')
    started = time.time()
    source_sha = digest_bytes(Path(__file__).read_bytes())
    protocol_sha = digest_bytes((root / REPORT_REL / 'PROTOCOL.md').read_bytes())
    write_json(report / 'STARTED.json', dict(unix=started, pid=os.getpid(),
               source_sha256=source_sha, protocol_sha256=protocol_sha, conditions=list(POOLS),
               declared_whole_train_pools=[1313, 13105], only_train_tasks=list(range(2, 12)) + [16]))
    opened = {}
    try:
        conditions, objects = {}, {}
        for pool in POOLS:
            conditions[pool], objects[pool] = load_pool(root, pool, opened)
        check(set(objects['pct01']).issubset(objects['pct10']), 'selected_train_not_nested')
        check(all(objects['pct01'][k] == objects['pct10'][k] for k in objects['pct01']),
              'nested_selected_train_content_changed')
        result = dict(status='COMPLETE_DESCRIPTIVE_TRAIN_ONLY',
                      source_sha256=source_sha, protocol_sha256=protocol_sha,
                      conditions=conditions, selected_train_nested_exact=True,
                      descriptive_non_obstacle_flags=sorted(DESCRIPTIVE_FLAGS),
                      opened_data_files=opened, test_files_read=0, other_domain_train_files_read=0,
                      model_loads=0, decoder_forwards=0, generations=0, new_scores=0, updates=0,
                      seconds=time.time() - started,
                      interpretation='Counts describe supervision, not a cause or model outcome; '
                      '1%/10% acquisition requires separate function evidence.')
        write_json(report / 'AUDIT.json', result)
        with (report / 'FINDINGS.md').open('x', encoding='utf8', newline='\n') as stream:
            stream.write(findings(result))
        write_json(report / 'COMPLETE.json', dict(source_sha256=source_sha,
                   protocol_sha256=protocol_sha, seconds=time.time() - started,
                   audit_sha256=digest_bytes((report / 'AUDIT.json').read_bytes()),
                   findings_sha256=digest_bytes((report / 'FINDINGS.md').read_bytes()),
                   actual_opened_data_file_count=len(opened), test_files_read=0,
                   model_loads=0, generations=0, new_scores=0))
        print(json.dumps(dict(status='COMPLETE', seconds=time.time() - started,
                              audited_QA={k: v['audited_train_QA'] for k, v in conditions.items()},
                              new_scores=0, test_files_read=0)))
    except BaseException as exc:
        write_json(report / 'FAILURE.json', dict(seconds=time.time() - started,
                   error_type=type(exc).__name__, error_code=str(exc),
                   source_sha256=source_sha, opened_data_files=opened,
                   test_files_read=0, model_loads=0, new_scores=0))
        raise


if __name__ == '__main__':
    main()
