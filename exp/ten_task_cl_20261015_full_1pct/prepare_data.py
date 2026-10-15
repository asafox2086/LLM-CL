"""Original task composition -> full train/test and a frozen nested 1% train view."""
import collections
import csv
import hashlib
import io
import json
import math
import os
import random
import zipfile
from pathlib import Path

import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEST=ROOT/'data/TenTaskCL'/HERE.name


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def image_bytes(dataset,value):
    if isinstance(value,dict):
        blob=value.get('bytes')
        if blob is None:blob=Path(value['path']).read_bytes()
    elif isinstance(value,bytes):blob=value
    else:raise TypeError('Unsupported native image representation')
    digest=hashlib.sha256(blob).hexdigest()
    p=ROOT/'data'/dataset/'native_images'/digest[:2]/(digest+'.img')
    if not p.exists():
        from PIL import Image
        with Image.open(io.BytesIO(blob)) as im:im.verify()
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)
    return str(p.relative_to(ROOT))


def row(dataset,identifier,question,answer,images=None,context='',source_split=None,metadata=None):
    answers=answer if isinstance(answer,list) else [answer]
    assert answers and all(isinstance(x,str) for x in answers)
    return {'id':dataset+':'+str(identifier),'dataset':dataset,'source_id':str(identifier),
            'question':question,'answer':answers[0],'answers':answers,'images':images or [],
            'context':context,'source_split':source_split,
            'metadata':metadata or {}}


def native_parquet(dataset,role):
    d=ROOT/'data'/dataset/'raw'
    prefix='train' if role=='train' else 'test' if dataset=='VQA_RAD' else 'validation'
    names=sorted(d.rglob(prefix+'-*.parquet'))
    assert names,(dataset,prefix)
    for p in names:
        source_offset=0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=32):
            for index,x in enumerate(batch.to_pylist()):
                image=image_bytes(dataset,x['image'])
                if dataset=='VQAv2':
                    answers=[a['answer'] for a in x['answers']]
                    primary=x['multiple_choice_answer']
                    r=row(dataset,x['question_id'],x['question'],[primary]+answers,[image],source_split=prefix,
                          metadata={'question_type':x['question_type'],'answer_type':x['answer_type'],'original_image_id':x['image_id']})
                elif dataset=='TextVQA':
                    r=row(dataset,x['question_id'],x['question'],x['answers'],[image],source_split=prefix,
                          metadata={'original_image_id':x['image_id'],'question_type':'text_reading'})
                else:
                    # Preserve even repeated source records with distinct original row IDs.
                    identifier=f'{p.name}:{source_offset+index}'
                    r=row(dataset,identifier,x['question'],x['answer'],[image],source_split=prefix)
                yield r
            source_offset+=batch.num_rows


def native_gqa(role):
    d=ROOT/'data/GQA'
    index=d/'native_image_index.json'
    if index.exists():images=json.loads(index.read_text())
    else:
        images={}
        for p in sorted((d/'raw').glob('*_images/*.parquet')):
            for b in pq.ParquetFile(p).iter_batches(batch_size=32):
                for x in b.to_pylist():images[str(x['id'])]=image_bytes('GQA',x['image'])
        write_json(index,images)
    split='train' if role=='train' else 'testdev'
    for p in sorted((d/'raw'/f'{split}_balanced_instructions').glob('*.parquet')):
        for b in pq.ParquetFile(p).iter_batches(batch_size=256):
            for x in b.to_pylist():
                assert x.get('answer') is not None
                yield row('GQA',x['id'],x['question'],x['answer'],[images[str(x['imageId'])]],source_split=split,
                          metadata={'question_type':x['types']['structural'],'semantic_type':x['types']['semantic'],
                                    'original_image_id':x['imageId'],'configuration':'official balanced'})


def native_slake(role):
    d=ROOT/'data/SLAKE'
    images=zipfile.ZipFile(d/'raw/imgs.zip')
    names={p:p for p in images.namelist()}
    for x in json.loads((d/'raw'/f'{role}.json').read_text()):
        suffix=x['img_name']
        members=[p for p in names if p.endswith('/'+suffix) or p==suffix]
        assert len(members)==1,(suffix,members)
        path=image_bytes('SLAKE',images.read(members[0]))
        yield row('SLAKE',x['qid'],x['question'],x['answer'],[path],source_split=role,
                  metadata={k:x[k] for k in ['q_lang','answer_type','content_type','img_id']})


def native_squad(role):
    p=ROOT/'data/SQuAD11/raw'/('train-v1.1.json' if role=='train' else 'dev-v1.1.json')
    for article in json.loads(p.read_text())['data']:
        for passage in article['paragraphs']:
            for x in passage['qas']:
                yield row('SQuAD11',x['id'],x['question'],[a['text'] for a in x['answers']],context=passage['context'],
                          source_split='train' if role=='train' else 'dev')


def native_finqa(role):
    with zipfile.ZipFile(ROOT/'data/FinQA/raw/source.zip') as z:
        candidates=[x for x in z.namelist() if x.endswith('/dataset/'+role+'.json')]
        assert len(candidates)==1
        for x in json.loads(z.read(candidates[0])):
            qa=x['qa'];answer=qa['answer'];answer=str(answer) if not isinstance(answer,str) else answer
            table=x.get('table_ori',x.get('table',[]))
            context='\n'.join(x['pre_text'])+'\n'+'\n'.join('\t'.join(map(str,a)) for a in table)+'\n'+'\n'.join(x['post_text'])
            yield row('FinQA',x['id'],qa['question'],answer,context=context,source_split=role,
                      metadata={'program_reference':qa.get('program'),'native_exe_ans':qa.get('exe_ans')})


def research_holdout(rows,group):
    buckets=collections.defaultdict(list)
    for x in rows:buckets[group(x)].append(x)
    keys=sorted(buckets,key=lambda k:hashlib.sha256(('42:'+k).encode()).digest())
    test_keys=set(keys[:max(1,math.ceil(.2*len(keys)))])
    return {role:[x for k in keys if (k not in test_keys)==(role=='train') for x in buckets[k]] for role in ['train','test']}


def native_medquad():
    # Source collection has no official split. Split all answer-bearing QA by original document.
    import xml.etree.ElementTree as ET
    d=ROOT/'data/MedQuAD/raw/source.zip'
    rows=[]
    missing=0
    with zipfile.ZipFile(d) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.xml'):continue
            document=ET.fromstring(z.read(name))
            for i,pair in enumerate(document.findall('.//QAPair')):
                q=pair.find('Question');a=pair.find('Answer')
                if q is None or a is None or a.text is None:
                    missing+=1;continue  # Source removed copyrighted answers; never generate replacements.
                rows.append(row('MedQuAD',name+':'+str(i),q.text or '',a.text,
                     source_split='research_document_holdout_80_20',
                     metadata={'source_file':name,'source_url':document.attrib.get('url'),
                               'question_type':q.attrib.get('qtype','unspecified')}))
    write_json(DEST/'medquad_unavailable_labels.json',{'source_missing_answer_count':missing,
               'reason':'No public gold answer; cannot train or score these original records',
               'question_type_filtering':False,'answer_bearing_rows':len(rows)})
    return research_holdout(rows,lambda x:x.get('metadata',{}).get('source_url') or x.get('metadata',{}).get('source_file') or x['id'])


def native_geoquestions():
    d=ROOT/'data/GeoQuestions1089/raw'
    qs=json.loads((d/'GeoQuestions1089.json').read_text());answers=json.loads((d/'GeoQuestions1089_answers.json').read_text())
    rows=[]
    unavailable=[]
    for key,x in qs.items():
        native=answers[key]
        if native is None:
            unavailable.append(key);continue
        answer=native if isinstance(native,str) else json.dumps(native,ensure_ascii=False)
        rows.append(row('GeoQuestions1089',key,x['Question'],answer,
             source_split='research_query_group_holdout_80_20',metadata={'question_type':str(x.get('Category')),
             'original_query_group':str(x.get('Query',x.get('SPARQL',key))),
             'native_answer_object':native if not isinstance(native,str) else None,
             'answer_format':'original CSV string or lossless serialization of original boolean result object'}))
    write_json(DEST/'geoquestions_unavailable_labels.json',{'source_rows':len(qs),
               'public_answer_rows':len(rows),'missing_answer_ids':unavailable,'question_type_filtering':False})
    return research_holdout(rows,lambda x:x['metadata']['original_query_group'])


def native_geochat():
    d=ROOT/'data/GeoChat_Instruct';raw=json.loads((d/'raw/GeoChat_Instruct.json').read_text());rows=[]
    for index,x in enumerate(raw):
        history=[];pairs=x['conversations']
        assert len(pairs)%2==0
        for turn in range(0,len(pairs),2):
            human,assistant=pairs[turn:turn+2]
            assert human['from']=='human' and assistant['from'] in ['gpt','assistant']
            question=human['value'].replace('<image>','').strip()
            image=ROOT/'data/GeoChat_Instruct/raw/images'/Path(x['image']).name
            tag=question.split(']',1)[0]+']' if question.startswith('[') else 'unmarked'
            rows.append(row('GeoChat',f'{index}:{turn//2}',question,assistant['value'],[str(image.relative_to(ROOT))],
                 context='\n'.join(history),source_split='research_image_group_holdout_80_20',
                 metadata={'question_type':tag,'original_human_value':human['value'],'image_group':x['image']}))
            history += ['Human: '+question,'Assistant: '+assistant['value']]
    return research_holdout(rows,lambda x:x['metadata']['image_group'])


def native_mme():
    d=ROOT/'data/MME_Finance';rows=[]
    with zipfile.ZipFile(d/'raw/MMfin.zip') as z:
        for x in csv.DictReader((d/'raw/MMfin.tsv').open(),delimiter='\t'):
            p=d/'native_images'/x['image_path'];assert '..' not in p.relative_to(d/'native_images').parts
            if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read('MMfin/'+x['image_path']))
            rows.append(row('MME-Finance',x['index'],x['question'],x['answer'],[str(p.relative_to(ROOT))],
                 context=x['background'],source_split='research_chart_group_holdout_80_20',
                 metadata={'question_type':x['task_category'],'image_type':x['image_type'],'image_style':x['image_style'],
                           'chart_group':str(Path(x['image_path']).parent)}))
    return research_holdout(rows,lambda x:x['metadata']['chart_group'])


def native_llavamed(role):
    d=ROOT/'data/LLaVA_Med'
    if role=='test':
        for line in (d/'raw/llava_med_eval_qa50_qa.jsonl').open():
            x=json.loads(line)
            yield row('LLaVA-Med',f"test:{x['question_id']}",x['text'].replace('<image>','').strip(),x['gpt4_answer'],
                      [str((d/'images'/Path(x['image']).name).relative_to(ROOT))],source_split='official eval qa50',
                      metadata={'annotation_origin':'author GPT-4 reference','question_type':str(x.get('category','conversation'))})
        return
    for x in json.loads((d/'raw/llava_med_instruct_60k.json').read_text()):
        history=[];pairs=x['conversations'];assert len(pairs)%2==0
        for i in range(0,len(pairs),2):
            q,a=pairs[i:i+2];assert q['from']=='human' and a['from'] in ['gpt','assistant']
            question=q['value'].replace('<image>','').strip()
            yield row('LLaVA-Med',f"train:{x['id']}:{i//2}",question,a['value'],
                 [str((d/'images'/Path(x['image']).name).relative_to(ROOT))],context='\n'.join(history),source_split='official instruct60k',
                 metadata={'question_type':'conversation','original_human_value':q['value'],'annotation_origin':'author GPT-4'})
            history += ['Human: '+question,'Assistant: '+a['value']]


def native_aqua(role,knowledge):
    d=ROOT/'data/AQUA'
    for index,x in enumerate(json.loads((d/'raw'/f'{role}.json').read_text())):
        if bool(x['need_external_knowledge']) != knowledge:continue
        # Native author modality flag defines the two tasks, not an inferred question-type filter.
        yield row('AQUA',f'{role}:{index}',x['question'],x['answer'],
             [] if knowledge else [str((d/'selected_images'/x['image']).relative_to(ROOT))],source_split=role,
             metadata={'need_external_knowledge':knowledge,'original_image':x['image'],
                       'question_type':'knowledge' if knowledge else 'visual','conditioning':'question-only knowledge view' if knowledge else 'original painting'})


def subset_one_percent(rows):
    strata=collections.defaultdict(list)
    for x in rows:
        m=x.get('metadata',{});strata[(x['dataset'],str(m.get('question_type',m.get('content_type','unspecified'))))].append(x)
    # Exact ceil(1% N) per source dataset, allocated proportionally across native task types.
    selected=[]
    datasets=sorted({k[0] for k in strata})
    for dataset in datasets:
        keys=sorted(k for k in strata if k[0]==dataset);n=sum(len(strata[k]) for k in keys);target=math.ceil(.01*n)
        quotas={k:math.floor(target*len(strata[k])/n) for k in keys}
        priority=sorted(keys,key=lambda k:(-(target*len(strata[k])/n-quotas[k]),k))
        for k in priority[:target-sum(quotas.values())]:quotas[k]+=1
        for k in keys:
            ordered=sorted(strata[k],key=lambda x:hashlib.sha256(('42:'+x['id']).encode()).digest())
            selected.extend(ordered[:quotas[k]])
    return selected


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    tasks=[(1,'natural','text','Answer the question using the supplied passage. Give the answer directly.'),
           (2,'natural','image','Answer the question using the image. Preserve any requested output format.'),
           (3,'medical','text','Answer the medical question accurately and directly.'),
           (4,'medical','image','Answer the medical question using the image and supplied conversation history.'),
           (5,'geography','text','Answer the geographic question. Use the original result-table format.'),
           (6,'geography','image','Follow the original remote-sensing image instruction and its requested format.'),
           (7,'finance','text','Answer the financial question using the supplied text and table.'),
           (8,'finance','image','Answer the original financial question using the image and supplied background.'),
           (9,'art','text','Answer the art history question directly.'),
           (10,'art','image','Answer the art question using the painting.')]
    manifests={c:{'experiment_id':HERE.name,'condition':c,'seed':42,'split_version':HERE.name+'_native_v1',
                'formal_ready':False,'assets_pending':[],'validation_enabled':False,
                'all_original_question_types':True,'counts_per_task':'variable; see each task.files',
                'test_definition':'full labeled public evaluation split; official hidden test without labels is not scored',
                'tasks':[]} for c in ['full','pct01']}
    research={}
    def cached(tid,factory,role):
        if tid not in research:research[tid]=factory()
        return research[tid][role]
    sources={1:lambda role:native_squad(role),2:lambda role:(x for n in ['VQAv2','GQA','TextVQA'] for x in (native_gqa(role) if n=='GQA' else native_parquet(n,role))),
             3:lambda role:cached(3,native_medquad,role),
             4:lambda role:(x for n in ['VQA_RAD','SLAKE','LLaVA-Med'] for x in (native_parquet(n,role) if n=='VQA_RAD' else native_slake(role) if n=='SLAKE' else native_llavamed(role))),
             5:lambda role:cached(5,native_geoquestions,role),
             6:lambda role:cached(6,native_geochat,role),7:lambda role:native_finqa(role),
             8:lambda role:cached(8,native_mme,role),9:lambda role:native_aqua(role,True),10:lambda role:native_aqua(role,False)}
    counts=[]
    for tid,domain,modality,instruction in tasks:
        split_rows={role:list(sources[tid](role)) for role in ['train','test']}
        assert split_rows['train'] and split_rows['test']
        for role,rows in split_rows.items():
            assert len({x['id'] for x in rows})==len(rows),(tid,role,'duplicate source QA IDs')
        assert not {x['id'] for x in split_rows['train']} & {x['id'] for x in split_rows['test']}
        sampled=subset_one_percent(split_rows['train'])
        for condition in manifests:
            task={'task':tid,'task_id':f'T{tid:02d}','domain':domain,'modality':modality,'instruction':instruction,'files':{}}
            for role in ['train','test']:
                rows=sampled if condition=='pct01' and role=='train' else split_rows[role]
                p=DEST/condition/f'T{tid:02d}_{role}.jsonl'
                p.parent.mkdir(parents=True,exist_ok=True)
                # Two protocols share the same test file bytes, not independently sampled tests.
                with p.open('w',encoding='utf-8') as f:
                    for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
                missing=sorted({name for x in rows for name in x.get('images',[]) if not (ROOT/name).is_file()})
                if missing:manifests[condition]['assets_pending'].append({'task':tid,'role':role,'missing_image_count':len(missing),'examples':missing[:3]})
                task['files'][role]={'path':str(p.relative_to(ROOT)),'count':len(rows),'sha256':sha(p),
                    'dataset_counts':dict(collections.Counter(x['dataset'] for x in rows)),
                    'original_type_counts':dict(collections.Counter(str(x.get('metadata',{}).get('question_type',x.get('metadata',{}).get('content_type','unspecified'))) for x in rows))}
            manifests[condition]['tasks'].append(task)
            ordered=sorted(sampled if condition=='pct01' else split_rows['train'],
                key=lambda x:hashlib.sha256(f"42:{tid}:{x['id']}".encode()).digest())
            task['training_order_sha256']=hashlib.sha256(''.join(x['id']+'\n' for x in ordered).encode()).hexdigest()
            task['training_order_rule']='SHA256(42:stage:original_QA_ID), ascending digest'
        assert manifests['full']['tasks'][-1]['files']['test']['sha256']==manifests['pct01']['tasks'][-1]['files']['test']['sha256']
        counts.append({'task':tid,'domain':domain,'modality':modality,'full_train':len(split_rows['train']),'pct01_train':len(sampled),'full_test':len(split_rows['test'])})
        print(json.dumps({'prepared':counts[-1]}),flush=True)
        research.pop(tid,None)
    policy=json.loads((HERE/'execution_policy.json').read_text())
    for condition,m in manifests.items():
        m['formal_ready']=not m['assets_pending']
        path=DEST/condition/'manifest.json';write_json(path,m)
        policy['conditions'][condition].update(manifest=str(path.relative_to(ROOT)),manifest_sha256=sha(path),ready=m['formal_ready'])
        if m['formal_ready']:
            for p in (DEST/condition).glob('*'):p.chmod(0o444)
    policy['training_enabled']=all(m['formal_ready'] for m in manifests.values())
    write_json(HERE/'execution_policy.json',policy)
    write_json(DEST/'task_counts.json',counts)
    write_json(DEST/'preparation_status.json',{'status':'ready' if policy['training_enabled'] else 'assets_pending','conditions':{c:m['assets_pending'] for c,m in manifests.items()},'counts':counts})


if __name__=='__main__':main()
