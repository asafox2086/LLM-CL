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
DEST=ROOT/'data/FiveDomainCL'/HERE.name


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


def native_rsvqa(role):
    d=ROOT/'data/RSVQA_LR'
    prefix='LR_split_'+role+'_'
    questions=json.loads((d/'raw'/(prefix+'questions.json')).read_text())['questions']
    answers={x['id']:x for x in json.loads((d/'raw'/(prefix+'answers.json')).read_text())['answers'] if x.get('active',True)}
    images={x['id']:x for x in json.loads((d/'raw'/(prefix+'images.json')).read_text())['images'] if x.get('active',True)}
    for x in questions:
        # Inactive entries are the author's placeholders for other source splits.
        # Respect this official split mask; never infer a question-type filter.
        if not x.get('active',True):continue
        native_image=images[x['img_id']]
        # The release contains RGB TIFFs. The original image bytes remain untouched.
        # Qwen's regular image preprocessing handles RGB conversion at model input.
        image=d/'images'/(str(x['img_id'])+'.tif')
        if not image.is_file():
            alternate=d/'images'/native_image['original_name']
            if alternate.is_file():image=alternate
        refs=[str(answers[a]['answer']) for a in x['answers_ids']]
        assert refs and all(answers[a]['question_id']==x['id'] for a in x['answers_ids'])
        yield row('RSVQA-LR',x['id'],x['question'],refs,[str(image.relative_to(ROOT))],source_split=role,
                  metadata={'question_type':x['type'],'original_image_id':x['img_id'],
                            'original_image_name':native_image['original_name'],'configuration':'original LR; all QA types; raw numeric answers'})


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


def native_finmme():
    """Author release is an evaluation pool named train, not an official split.

    Hold out complete image-byte groups. Do not expose gold-derived captions or
    related sentences to the model; the released questions/options and image
    are its inputs. Empty gold records remain in a separate source audit.
    """
    rows=[];unlabeled=[]
    for p in sorted((ROOT/'data/FinMME/raw/data').glob('*.parquet')):
        for batch in pq.ParquetFile(p).iter_batches(batch_size=32):
            for x in batch.to_pylist():
                image=image_bytes('FinMME',x['image'])
                group=Path(image).stem
                metadata={k:x[k] for k in ['question_type','unit','tolerance','verified_caption','related_sentences','options']}
                if metadata['tolerance'] is not None and not math.isfinite(metadata['tolerance']):metadata['tolerance']=None
                metadata.update(image_group_sha256=group,source_repository='luojunyu/FinMME',
                                gold_context_not_in_model_input=['verified_caption','related_sentences'])
                item=row('FinMME',x['id'],x['question_text'],x['answer'] or '',[image],
                         context=x['options'] or '',source_split='research_image_group_holdout_80_20',metadata=metadata)
                if x['answer'] is None or x['answer']=='':unlabeled.append(item)
                else:rows.append(item)
    assert len(rows)+len(unlabeled)==11099
    write_json(DEST/'unlabeled_source_records.json',{'dataset':'FinMME','records':unlabeled,
               'reason':'Author answer is empty; retained for audit, never fabricated or scored/trained as gold',
               'source_rows':11099,'public_labeled_rows':len(rows)})
    result=research_holdout(rows,lambda x:x['metadata']['image_group_sha256'])
    assert not {x['metadata']['image_group_sha256'] for x in result['train']} & {x['metadata']['image_group_sha256'] for x in result['test']}
    return result


def native_coda(role):
    """Use author English LLaVA Train/Test, including original regional images."""
    split='Train' if role=='train' else 'Test'
    files=sorted((ROOT/'data/CODA_LM/raw/English').glob(split+'-*.parquet'))
    assert len(files)==(32 if role=='train' else 4),(split,'Incomplete official shard set',len(files))
    for p in files:
        offset=0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=16):
            for i,x in enumerate(batch.to_pylist()):
                turns=x['conversations'];assert len(turns)==2 and turns[0]['from']=='human' and turns[1]['from']=='gpt'
                question=turns[0]['value'];answer=turns[1]['value']
                assert isinstance(answer,str) and answer.strip(),(split,x['id'],'No public gold')
                image=image_bytes('CODA_LM',x['image'])
                yield row('CODA-LM',f"{split}:{x['id']}",question,answer,[image],source_split=split,
                          metadata={'original_id':x['id'],'configuration':'author English LLaVA format',
                                    'question_type':x['id'].split('_')[1],
                                    'original_human_value':question,'original_assistant_value':answer,
                                    'source_shard':p.name,'source_row':offset+i,
                                    'regional_images':'Author release retains its original red-box prompt images'})
            offset+=batch.num_rows


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    tasks=[(1,'natural','image','Answer the question using the image. Give the answer directly.'),
           (2,'medical','image','Answer the medical question using the image.'),
           (3,'geography','image','Answer the remote-sensing question using the image. Preserve the original numeric answer format.'),
           (4,'finance','image','Answer the financial question using the image and original background.'),
           (5,'art','image','Answer the art question using the painting.'),
           (6,'traffic','image','Answer the driving question using the road image.')]
    manifests={c:{'experiment_id':HERE.name,'condition':c,'seed':42,'split_version':HERE.name+'_six_domain_v1',
                'formal_ready':False,'assets_pending':[],'validation_enabled':False,'dataset_revision':'six_domain_finmme_coda_20261016',
                'all_original_question_types':True,'counts_per_task':'variable; see each task.files','max_train_qa_per_task':99999,
                'test_definition':'full labeled public evaluation split; official hidden test without labels is not scored',
                'tasks':[]} for c in ['full','pct01']}
    research={}
    def cached(tid,factory,role):
        if tid not in research:research[tid]=factory()
        return research[tid][role]
    def aqua_all(role):
        for knowledge in [True,False]:
            for x in native_aqua(role,knowledge):
                x['images']=[str(Path('data/AQUA/selected_images')/x['metadata']['original_image'])]
                x['metadata']['conditioning']='original painting; all native knowledge and visual QA'
                yield x
    sources={1:lambda role:native_parquet('TextVQA',role),2:lambda role:native_slake(role),
             3:lambda role:native_rsvqa(role),4:lambda role:cached(4,native_finmme,role),5:aqua_all,6:native_coda}
    counts=[]
    for tid,domain,modality,instruction in tasks:
        split_rows={role:list(sources[tid](role)) for role in ['train','test']}
        expected={1:(34602,5000),2:(9835,2094),3:(57223,10004),4:(8864,2233),5:(69812,4912),6:(20495,2123)}
        assert tuple(len(split_rows[r]) for r in ['train','test'])==expected[tid],(tid,'Native count mismatch')
        assert split_rows['train'] and split_rows['test']
        assert len(split_rows['train'])<100000,(tid,'Compact protocol: reject hundred-thousand-QA task')
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
    if policy['training_enabled']:
        from build_dataset_report import main as refresh_report
        refresh_report()


if __name__=='__main__':main()
