"""Read-only CPU: frozen knowledge coverage, saved update geometry/resources.

Coverage is descriptive after test inspection; never selects/replaces facts.
Literal relation/value overlap is NOT semantic answerability or clinical proof.
"""
import collections,hashlib,json,math,sys,unicodedata
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'exp/onevision_diagnostics_20261016'))
from geometry import inner
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def normalize(text):return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())


def coverage(rows,raw,facts):
    relations={normalize(f['relation']) for f in facts};values={normalize(f['value']) for f in facts}
    subjects={normalize(f['subject']) for f in facts};counts=collections.Counter();relation_counts=collections.Counter()
    for row in rows:
        source=raw[str(row['source_id'])]
        assert source['question'].strip()==row['question'].strip() and source['answer'].strip()==row['answer'].strip()
        relation=normalize(source['triple'][1]);relation_counts[source['triple'][1]]+=1
        counts['author_relation_exactly_in_pool']+=relation in relations
        counts['author_relation_and_fixed_tail_exactly_in_same_fact']+=any(
            relation==normalize(f['relation']) and normalize(source['triple'][2])==normalize(f['value']) for f in facts)
        counts['triple_head_is_vhead']+=source['triple'][0]=='vhead'
        counts['gold_answer_equals_a_fact_value']+=any(normalize(a) in values for a in row['answers'])
        counts['gold_answer_equals_a_fact_subject']+=any(normalize(a) in subjects for a in row['answers'])
        question=normalize(row['question'])
        counts['question_contains_fact_subject']+=any(s and s in question for s in subjects)
    return {'QA':len(rows),'counts':dict(counts),'author_relation_counts':dict(relation_counts)}


def norm(state):return 4*math.sqrt(max(0.,sum(inner(a,a) for a in state.values())))
def cosine(a,b):
    na=norm(a);nb=norm(b)
    return 16*sum(inner(a[k],b[k]) for k in a)/(na*nb) if na and nb else None


def scores(records):
    return {'QA':len(records),'EM_correct':sum(x['scores']['exact_match']==100 for x in records),
            **{metric:sum(x['scores'][metric] for x in records)/len(records) for metric in ['exact_match','token_f1','rougeL']},
            'mean_nll':sum(x['nll'] for x in records)/len(records)}


def cpu_checks():
    generator=torch.Generator().manual_seed(42)
    a={'A':torch.randn(2,3,generator=generator,dtype=torch.float64),'B':torch.randn(4,2,generator=generator,dtype=torch.float64)}
    b={'A':torch.randn(1,3,generator=generator,dtype=torch.float64),'B':torch.randn(4,1,generator=generator,dtype=torch.float64)}
    da,db=4*a['B']@a['A'],4*b['B']@b['A']
    assert math.isclose(norm({'k':a}),float(da.norm()),rel_tol=1e-12)
    assert math.isclose(cosine({'k':a},{'k':b}),float((da*db).sum()/da.norm()/db.norm()),rel_tol=1e-12)
    delta={'A':torch.cat([a['A'],b['A']],0),'B':torch.cat([a['B'],-b['B']],1)}
    assert torch.allclose(4*delta['B']@delta['A'],da-db)
    facts=[{'subject':'Lung','relation':'belong to','value':'Respiratory System'}]
    raw={'1':{'question':'Which organ?','answer':'Lung','triple':['vhead','belong to','Respiratory System']},
         '2':{'question':'Is the lung in that system?','answer':'Yes','triple':['vhead','belong to','Respiratory System']}}
    rows=[dict(source_id=k,question=v['question'],answer=v['answer'],answers=[v['answer']]) for k,v in raw.items()]
    c=coverage(rows,raw,facts)['counts']
    assert c['author_relation_and_fixed_tail_exactly_in_same_fact']==2 and c['gold_answer_equals_a_fact_value']==0 and c['gold_answer_equals_a_fact_subject']==1
    return 'passed: unequal-rank dense geometry, difference factors, yes/no and reverse-subject literal-overlap interpretation'


def main():
    torch.set_num_threads(4);checks=cpu_checks();sources={}
    def read(path):sources[str(path.relative_to(ROOT))]=digest(path);return json.loads(path.read_text())
    pool=read(ROOT/'data/OneVisionMechanisms/20261016/knowledge16.json');assert pool['control_version']==2
    raw={split:{str(x['qid']):x for x in read(ROOT/f'data/SLAKE/raw/{split}.json')} for split in ['train','test']}
    cohorts={};qa_by_split={}
    for split in ['train','test']:
        p=ROOT/f'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/T11_{split}.jsonl'
        sources[str(p.relative_to(ROOT))]=digest(p);rows=[json.loads(line) for line in p.read_text().splitlines()]
        qa_by_split[split]=rows
        cohorts[split]={kind:coverage(rows,raw[split],pool[kind]) for kind in ['correct','unrelated','permuted']}
    # Fixed groups from raw author triples and the frozen correct pool, identical
    # across all outcomes; post-hoc reporting only, never selection/training.
    relation_groups={}
    for qid,row in raw['test'].items():
        relation,tail=map(normalize,row['triple'][1:])
        same=any(relation==normalize(f['relation']) for f in pool['correct'])
        both=any(relation==normalize(f['relation']) and tail==normalize(f['value']) for f in pool['correct'])
        relation_groups[qid]='same_relation_fixed_tail' if both else ('relation_only' if same else 'relation_absent')
    attempt=read(ROOT/'summary/onevision_mechanisms_20261016/knowledge_attempt.json');run=ROOT/attempt['run']
    assert digest(ROOT/'data/OneVisionMechanisms/20261016/knowledge16.json')==attempt['source_sha256']['knowledge16.json']
    shared_ref=read(run/'shared_adapter_reference.json');shared_path=ROOT/shared_ref['path'];assert digest(shared_path)==shared_ref['sha256']
    shared=torch.load(shared_path,map_location='cpu',weights_only=True)['adapter'];sources[str(shared_path.relative_to(ROOT))]=digest(shared_path)
    telemetry_path=run/'resources.jsonl';telemetry=[];text=telemetry_path.read_text().splitlines()
    for i,line in enumerate(text):
        try:telemetry.append(json.loads(line))
        except json.JSONDecodeError:assert i==len(text)-1 # read-only concurrent tail
    states={};updates={};resource={}
    for path in sorted((run/'adapters').glob('*.pt')):
        if not path.with_suffix('.json').exists():continue
        meta=read(path.with_suffix('.json'));assert digest(path)==meta['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)['adapter'];assert len(state)==48
        assert all(torch.isfinite(t).all() for layer in state.values() for t in layer.values())
        case=path.stem;sources[str(path.relative_to(ROOT))]=digest(path)
        # Residual files are new functions; rank8 files are continued shared states.
        additional=state if case.startswith('residual_') else {k:{'B':torch.cat([state[k]['B'],-shared[k]['B']],1),
                    'A':torch.cat([state[k]['A'],shared[k]['A']],0)} for k in state}
        states[case]=additional
        training=read(run/'auxiliary_training'/f'{case}.json');events=training['events']
        assert len(events)==8 and training['additional_external_QA_exposures']==64
        assert all(e['lr_used']==1e-4 and math.isfinite(e['gradient_l2_preclip']) and e['gradient_l2_preclip']>0 for e in events)
        updates[case]={'new_deltaW_norm':norm(additional),'new_over_shared_deltaW_norm':norm(additional)/norm(shared),
                       'additional_update_vs_shared_cosine':cosine(additional,shared),'trainable_parameters':events[0]['trainable_parameters'],
                       'updates':len(events),'auxiliary_QA_exposures':64,
                       'epoch_mean_training_loss':[sum(e['loss'] for e in events if e['epoch']==epoch)/2 for epoch in range(4)],
                       'gradient_norm_min':min(e['gradient_l2_preclip'] for e in events),'gradient_norm_max':max(e['gradient_l2_preclip'] for e in events)}
        before=[x for x in telemetry if x['phase']==case+'_training_before'];after=[x for x in telemetry if x['phase']==case+'_training_after']
        if before and after:
            a,b=before[-1],after[-1];assert a['pid']==b['pid'] and a['unix']<=b['unix']
            keys=['VmRSS_bytes','Pss_bytes','cpu_uss_bytes','gpu_allocated_bytes','gpu_reserved_bytes','gpu_process_nvml_bytes']
            window=[x for x in telemetry if x['pid']==a['pid'] and a['unix']<=x['unix']<=b['unix']]
            resource[case]={'before':a,'after':b,'growth_bytes':{k:b[k]-a[k] if a.get(k) is not None and b.get(k) is not None else None for k in keys},
                            'observed_window_peak_bytes':{k:max(x[k] for x in window if x.get(k) is not None) if any(x.get(k) is not None for x in window) else None for k in keys},
                            'note':'Same long-lived GPU process; CUDA allocator peak counters are cumulative, not per-case reset. Sampled window peaks can miss brief allocations.'}
    frozen_protocol=read(ROOT/'exp/onevision_diagnostics_20261016/frozen_protocol.json')
    expected_ids={'medical':frozen_protocol['diagnostic_test_ids']['medical'],
                  'medical_KG':[x['id'] for x in qa_by_split['test']]}
    native_groups=[];literal_group_scores=[]
    for folder in sorted((run/'evaluations').iterdir()):
        marker=folder/'durable_complete.json'
        if not marker.exists():continue
        proof=read(marker)
        for group in ['medical','medical_KG']:
            path=folder/(group+'_image.jsonl');assert digest(path)==proof['files'][path.name]
            sources[str(path.relative_to(ROOT))]=digest(path);records=[json.loads(line) for line in path.read_text().splitlines()]
            assert [x['id'] for x in records]==expected_ids[group] and all('scores' in x for x in records)
            for dimension in ['answer_type','q_lang']:
                values=sorted({raw['test'][x['id'].split(':')[-1]][dimension] for x in records})
                for value in values:
                    selected=[x for x in records if raw['test'][x['id'].split(':')[-1]][dimension]==value]
                    native_groups.append({'case':folder.name,'group':group,'dimension':dimension,'value':value,**scores(selected)})
            if group=='medical_KG':
                for value in ['relation_absent','relation_only','same_relation_fixed_tail']:
                    selected=[x for x in records if relation_groups[x['id'].split(':')[-1]]==value]
                    literal_group_scores.append({'case':folder.name,'group':value,**scores(selected)})
    result={'run':attempt['run'],'audit_sha256':digest(Path(__file__)),'CPU_checks':checks,'source_sha256':sources,'literal_coverage':cohorts,
            'fact_relation_counts':{kind:dict(collections.Counter(f['relation'] for f in pool[kind])) for kind in ['correct','unrelated','permuted']},
            'additional_update_geometry':updates,'additional_deltaW_cosine':{a:{b:cosine(x,y) for b,y in states.items()} for a,x in states.items()},
            'training_memory':resource,'native_reporting_groups':native_groups,'fixed_literal_relation_group_scores':literal_group_scores,'GPU_used_by_audit':False,
            'limitations':['Post-hoc literal overlap, not semantic coverage or clinical validation; no replacement facts selected.',
                           'Yes/no questions and questions asking for the subject need not have answers equal to fact values.',
                           'Relation aliases/synonyms and visual binding are not resolved by exact matching.',
                           'Checkpoint admission proves completed training, not completed evaluation; scores live in separate durable-case report.',
                           'High deltaW cosine may reflect common factor initialization, language, templates or optimizer; not semantic redundancy proof.',
                           'Epoch losses average two different minibatches; not per-fact retention or generalization.',
                           'Only aggregate gradient norms were saved by this original knowledge runner, not all8 raw gradient vectors.',
                           'Current knowledge pool remains immutable; additional causal knowledge coverage/encoding tests are still needed.']}
    (REPORT/'knowledge_mechanism_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 知识匹配、参数与训练资源审计','',
           '本CPU审计不训练、不占GPU、不改知识池。correct仅表示作者原始事实、按TRAIN问题词汇相关选取，不保证覆盖当前QA。', '',
           '| 数据 | 事实池 | QA | 同关系 | 同关系且固定tail匹配同一事实 | gold等于事实value | gold等于事实subject |',
           '|---|---|---:|---:|---:|---:|---:|']
    for split,groups in cohorts.items():
        for kind,c in groups.items():
            n=c['counts'];lines.append(f'| {split} | {kind} | {c["QA"]} | {n["author_relation_exactly_in_pool"]} | {n["author_relation_and_fixed_tail_exactly_in_same_fact"]} | {n["gold_answer_equals_a_fact_value"]} | {n["gold_answer_equals_a_fact_subject"]} |')
    lines+=['','这是字面统计，不是可回答率。是非题的gold可能是Yes/No，反向提问可能回答subject，功能同义词也会漏计。'
            '因此不能以gold-value零匹配宣布知识零覆盖；需结合作者原三元组、视觉实体绑定和事实编码核验解释。'
            'TRAIN的需求关系与事实池的关系分布不同，原16事实负结果不足以否定专业知识假设。', '',
            '## 训练完成快照（不代表评测全部完成）','',
            '| 干预 | 可训练参数 | 更新/暴露 | 第1/4轮平均训练loss | 新增ΔW范数 | 与旧共享ΔW余弦 |',
            '|---|---:|---:|---:|---:|---:|']
    for case,u in updates.items():
        losses=u['epoch_mean_training_loss'];lines.append(f'| {case} | {u["trainable_parameters"]} | 8/64 | {losses[0]:.3f}/{losses[-1]:.3f} | {u["new_deltaW_norm"]:.4f} | {u["additional_update_vs_shared_cosine"]:.4f} |')
    lines+=['','实际ΔW=4BA，rank8继续训练使用相对旧共享模块的差分。各知识版本间的完整余弦矩阵、逐例训练前后CPU/GPU具体字节及窗口采样峰值见JSON。'
            '原知识runner仅记录梯度范数；不声称保存了每步原始梯度。新基底／预算实验会另存每步原始梯度。', '']
    lines+=['## 完整KG任务按固定字面关系分组','',
                '三个分组相同且互斥：同关系及固定tail31题、仅同关系40题、关系不在池中196题。分组来自原三元组与冻结事实池，'
                '是事后诊断，不能称独立确认。相同关系/tail也不保证视觉head实体有对应事实。', '',
                '| 已完成干预 | 固定分组 | QA | EM正确数 | Token F1 |', '|---|---|---:|---:|---:|']
    shown={'context_shared_none','context_shared_correct','context_shared_unrelated','context_shared_permuted',
           'residual_rank1_correct','residual_rank1_unrelated','residual_rank1_permuted'}
    for row in literal_group_scores:
        if row['case'] in shown:lines.append(f'| {row["case"]} | {row["group"]} | {row["QA"]} | {row["EM_correct"]} | {row["token_f1"]:.3f} |')
    lines+=['','## 原生答案类型的生成表现','',
            'OPEN/CLOSED来自作者原标注，不重写问题或筛除样本。语言分组与其他干预完整记录在JSON。', '',
            '| 已完成干预 | 数据组 | 原答案类型 | QA | EM正确数 | Token F1 |','|---|---|---|---:|---:|---:|']
    for row in native_groups:
        if row['case'] in shown and row['dimension']=='answer_type':
            lines.append(f'| {row["case"]} | {row["group"]} | {row["value"]} | {row["QA"]} | {row["EM_correct"]} | {row["token_f1"]:.3f} |')
    lines+=['','## 训练占用前后（MiB，非模型文件大小）','',
            '| 干预 | CPU USS前→后 | GPU allocated前→后 | GPU reserved前→后 |', '|---|---:|---:|---:|']
    for case,resource_row in resource.items():
        def memory(key):return f'{resource_row["before"][key]/2**20:.2f}→{resource_row["after"][key]/2**20:.2f}'
        lines.append(f'| {case} | {memory("cpu_uss_bytes")} | {memory("gpu_allocated_bytes")} | {memory("gpu_reserved_bytes")} |')
    lines+=['','同一长进程已包含前面上下文推理缓存；具体前后值可读，但不能把CUDA缓存变化当专家参数增长。'
            '新基底／预算及容量实验分别使用新进程来降低跨配置污染。','', '限制：','']
    lines+=['- '+x for x in result['limitations']]
    (REPORT/'KNOWLEDGE_MECHANISM_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'trained_snapshots':list(states),'TRAIN_coverage':cohorts['train']['correct'],'TEST_coverage':cohorts['test']['correct']}))


if __name__=='__main__':main()
