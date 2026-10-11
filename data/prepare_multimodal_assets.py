import argparse,concurrent.futures,hashlib,io,json,re,time,zipfile
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT/'data'/'.multimodal_asset_cache'; WORK.mkdir(exist_ok=True)
VERSION='20261011-open-vqa-v1'
BOOL={'yes','no','true','false','y','n','是','否','是的','不是','有','没有','对','不对'}
EMPTY={'','none','null','n/a','na','nan','not available','not provided','unknown','not specified'}

def canon(s):return re.sub(r'[^\w\s]','',str(s).lower()).strip()
def rows(path,columns=None):
    f=pq.ParquetFile(path)
    for batch in f.iter_batches(batch_size=16,columns=columns):
        yield from batch.to_pylist()
def texts(value):
    if value is None:return []
    if isinstance(value,str):return [value.strip()] if value.strip() else []
    if isinstance(value,dict):return texts(value.get('answer',value.get('text',[])))
    return [s for v in value for s in texts(v)]
def split_name(path):
    name=path.name.split('-')[0]
    return {'val':'validation'}.get(name,name)

class Exporter:
    def __init__(self,base,spec):
        self.base=base;self.spec=spec;self.out=base/'open_ended';self.out.mkdir(exist_ok=True)
        self.streams={};self.counts=Counter();self.raw_counts=Counter();self.reject=Counter();self.cache={};self.image_records={};self.example=None
    def image(self,obj,key=None):
        if key is not None and key in self.cache:return self.cache[key]
        if isinstance(obj,dict):data=obj.get('bytes');path=obj.get('path')
        elif isinstance(obj,(bytes,bytearray)):data=obj;path=None
        else:data=None;path=str(obj)
        if not data and path:data=Path(path).read_bytes()
        assert data,'No image bytes'
        sha=hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            fmt=im.format;w,h=im.size;im.verify()
        suffix={'JPEG':'jpg','PNG':'png','WEBP':'webp','TIFF':'tiff'}.get(fmt,fmt.lower())
        rel=f'images/{sha[:2]}/{sha}.{suffix}';target=self.base/rel
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        self.image_records[rel]={'file':rel,'sha256':sha,'bytes':len(data),'width':w,'height':h,'format':fmt}
        if key is not None:self.cache[key]=rel
        return rel
    def emit(self,split,q,answers,images,uid,meta=None):
        answers=texts(answers);q=str(q or '').strip();self.raw_counts[split]+=1
        if not q or not answers:self.reject['missing_question_or_answer']+=1;return
        answers=[a for a in answers if canon(a) not in EMPTY]
        if not answers:self.reject['missing_answer']+=1;return
        if any(canon(a) in BOOL for a in answers):self.reject['boolean_yes_no']+=1;return
        if re.search(r'(?:^|\n)\s*(?:[A-D][.)]|\([A-D]\))\s+',q):self.reject['multiple_choice_prompt']+=1;return
        if (meta or {}).get('answer_type','').lower() in {'yes/no','closed'}:self.reject['upstream_closed_type']+=1;return
        if (meta or {}).get('structural_type','').lower() in {'choose','verify'}:self.reject['gqa_choice_or_verification']+=1;return
        if not images:self.reject['missing_image']+=1;return
        answer=Counter(answers).most_common(1)[0][0]
        record={'id':uid,'dataset':self.spec['folder'],'domain':self.spec['domain'],'source_split':split,'images':images,'question':q,'answer':answer,'answers':list(dict.fromkeys(answers)),'metadata':meta or {}}
        assert all((self.base/p).is_file() for p in images)
        if split not in self.streams:self.streams[split]=(self.out/(split+'.jsonl.partial')).open('w',encoding='utf-8')
        self.streams[split].write(json.dumps(record,ensure_ascii=False)+'\n');self.counts[split]+=1
        if self.example is None:self.example=record
    def finish(self,manifest):
        files=[]
        for split,stream in self.streams.items():
            stream.close();part=self.out/(split+'.jsonl.partial');dest=self.out/(split+'.jsonl');part.replace(dest)
            h=hashlib.sha256(dest.read_bytes()).hexdigest()
            files.append({'file':str(dest.relative_to(self.base)),'rows':self.counts[split],'bytes':dest.stat().st_size,'sha256':h})
        assert self.counts and self.image_records,self.spec['folder']
        index=self.out/'images_manifest.jsonl'
        with index.open('w') as f:
            for x in self.image_records.values():f.write(json.dumps(x)+'\n')
        result={'version':VERSION,'source_revision':manifest['revision'],'dataset':self.spec['folder'],'domain':self.spec['domain'],'raw_question_rows_seen':dict(self.raw_counts),'open_ended_counts':dict(self.counts),'excluded':dict(self.reject),'image_count':len(self.image_records),'image_bytes':sum(x['bytes'] for x in self.image_records.values()),'files':files,'images_manifest':str(index.relative_to(self.base)),'image_verification':'Every exported image: SHA-256 and Pillow format/size/verify; every JSONL image reference exists.','filter':'No boolean/yes-no answers, no upstream CLOSED tags, no option-letter prompts, no GQA choose/verify questions. Short textual and numeric answers remain generation targets.','example':self.example}
        (self.base/'prepared_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        card=manifest['source_url'];role='已收集论文使用的数据' if self.spec['role']=='selected_papers_dataset' else '补充领域数据，未声称在这 29 篇中使用'
        md=[f'# {self.spec["folder"]}', '',f'领域：{self.spec["domain"]}。任务：{self.spec["task"]}。', '',f'关系：{role}。对应方法：'+('、'.join(self.spec['paper_methods']) or '无（领域扩展）')+'。', '',f'下载来源：[{manifest["repo"]}]({card})；原始发布来源：{self.spec["upstream"]}。',f'固定 revision：`{manifest["revision"]}`；下载字节数：{manifest["bytes"]:,}；上游许可声明：`{manifest.get("license")}`。完整逐文件哈希见 `asset_manifest.json`。', '',f'范围：{self.spec["coverage"]}。', '', '## 使用', '', '`raw/` 保留所选上游完整文件；`images/` 为实际图像，`open_ended/*.jsonl` 为训练用开放式问答。JSONL 图像路径相对于本数据集文件夹。原始文件中混合的封闭问题仍保留用于溯源，训练请使用 `open_ended/`。', '', '| 划分 | 开放式问答条数 |','|---|---|']
        md.extend(f'| {s} | {n:,} |' for s,n in sorted(self.counts.items()))
        md.extend(['',f'导出图像 {len(self.image_records):,} 张。过滤统计：`{dict(self.reject)}`。', '', result['filter'], '', '上游划分默认保留；法律合同按 CIK 公司分组确定性 80/10/10 划分，避免同公司合同跨训练／评估。这里提供的是数据资源，未执行模型训练，不能由下载成功推断模型效果。'])
        if self.spec['folder']=='SEC_Contracts_VQA':md.extend(['', '**法律数据质量说明**：上游将 SEC 合同转换为真实合同页图像，字段标签由 GPT-4o 自动抽取，未经本次人工逐条审校。这里把非布尔、非空字段转换为固定英文问题；保留整份合同的多页图像，不杜撰单页证据定位。适合领域适配候选数据，不能作为人工标注的权威法律推理金标准。'])
        if self.spec['folder']=='GQA':md.extend(['', '公开 test balanced instructions 无答案，因此只保留在 raw/，不生成带伪答案的训练 JSONL；testdev 使用公开答案。'])
        if self.spec['folder']=='VQA_RAD':md.extend(['', '使用的是镜像中公开的格式化／清洗版本，不等同于原始 VQA-RAD 论文中的全部问题条数。实际条数以本地 manifest 为准。'])
        (self.base/'README.md').write_text('\n'.join(md)+'\n')
        print('PREPARED',self.spec['folder'],dict(self.counts),'images',len(self.image_records),'excluded',dict(self.reject),flush=True)
        return result

def slake(exp):
    raw=exp.base/'raw'
    with zipfile.ZipFile(raw/'imgs.zip') as z:
        names=z.namelist()
        for split in ['train','validation','test']:
            records=json.loads((raw/(split+'.json')).read_text())
            for i,r in enumerate(records):
                if str(r.get('answer_type','')).upper()!='OPEN':exp.raw_counts[split]+=1;exp.reject['upstream_closed_type']+=1;continue
                img=r.get('img_name',r.get('image',''));matches=[n for n in names if n.endswith('/'+img) or n==img]
                assert len(matches)==1,(img,matches)
                rel=exp.image(z.read(matches[0]),img)
                exp.emit(split,r['question'],r['answer'],[rel],f'{split}-{r.get("qid",i)}',{'q_lang':r.get('q_lang'),'answer_type':'OPEN'})

def gqa(exp):
    image_map={}
    for f in sorted((exp.base/'raw').glob('*_images/*.parquet')):
        for r in rows(f):image_map[str(r['id'])]=exp.image(r['image'],r['id'])
        print('GQA_IMAGES',f.name,len(image_map),flush=True)
    for f in sorted((exp.base/'raw').glob('*_instructions/*.parquet')):
        split=split_name(f)
        for i,r in enumerate(rows(f)):
            image=image_map.get(str(r['imageId']))
            exp.emit(split,r['question'],r.get('answer'),[image] if image else [],f'{split}-{r["id"]}',{'upstream_image_id':r['imageId'],'structural_type':r.get('types',{}).get('structural','')})
        print('GQA_QUESTIONS',split,dict(exp.counts),flush=True)

LEGAL_QUESTIONS={
 'agreement_date':'What is the agreement signing date of this contract?',
 'effective_date':'What is the effective date of this contract?',
 'expiration_date':'What is the expiration date of this contract?',
 'party_address':'What is the address of the contracting party?',
 'party_name':'What is the name of the contracting party?',
 'counterparty_address':'What is the address of the counterparty?',
 'counterparty_name':'What is the name of the counterparty?',
 'counterparty_signer_name':'Who signs this contract on behalf of the counterparty?',
 'counterparty_signer_title':'What is the title of the counterparty signer?',
 'governing_law':'Which jurisdiction governs this contract?',
 'venue':'Where will legal proceedings under this contract take place?',
 'payment_frequency':'What is the payment frequency required by this contract?',
 'payment_term':'What is the payment term specified in this contract?',
 'renewal_term':'What is the renewal term of this contract?',
 'agreement_term':'What is the duration of this agreement?',
 'termination_notice_period':'What notice period is required to terminate this contract?',
 'opt_out_length':'How much notice is required to opt out of renewal?',
 'contract_value':'What is the total fixed contract value, including currency?',
}
def legal(exp):
    for f in sorted((exp.base/'raw/data').glob('*.parquet')):
        columns=['cik','filename','images','file_url']+list(LEGAL_QUESTIONS)
        for i,r in enumerate(rows(f,columns)):
            group=str(r['cik']);bucket=int(hashlib.sha256(('legal-cik-v1-'+group).encode()).hexdigest()[:8],16)%100
            split='train' if bucket<80 else ('validation' if bucket<90 else 'test')
            doc=str(r['file_url']);imgs=r['images'] or []
            if not imgs:exp.reject['missing_image']+=1;continue
            refs=[exp.image(img,doc+'-'+str(j)) for j,img in enumerate(imgs)]
            for field,q in LEGAL_QUESTIONS.items():
                exp.emit(split,q,r[field],refs,f'{f.stem}-{i}-{field}',{'cik':r['cik'],'document_id':doc,'field':field,'original_url':r['file_url'],'label_origin':'upstream GPT-4o field extraction; deterministic question template','split_group':'CIK'})

def generic(exp):
    for f in sorted((exp.base/'raw').rglob('*.parquet')):
        split=split_name(f)
        for i,r in enumerate(rows(f)):
            q=r.get('question',r.get('query'))
            answers=r.get('answers',r.get('answer',r.get('label')))
            meta={'answer_type':r.get('answer_type',''),'upstream_question_id':r.get('question_id',r.get('questionId')),'upstream_image_id':r.get('image_id',r.get('docId'))}
            if not q or not texts(answers) or any(canon(x) in BOOL for x in texts(answers)) or str(meta['answer_type']).lower() in ['yes/no','closed']:
                exp.emit(split,q,answers,[],f'{f.stem}-{i}',meta);continue
            key=r.get('image_id',r.get('docId'))
            if exp.spec['folder']=='DocVQA':
                key=(r.get('ucsf_document_id',r.get('docId')),r.get('ucsf_document_page_no',''))
            if key is not None:key=(split,str(key))
            image=exp.image(r['image'],key)
            exp.emit(split,q,answers,[image],f'{f.stem}-{i}',meta)
        print('EXPORT_PROGRESS',exp.spec['folder'],f.name,dict(exp.counts),flush=True)

def prepare(spec):
    base=ROOT/'data'/spec['folder'];manifest=json.loads((base/'asset_manifest.json').read_text());old=base/'prepared_manifest.json'
    if old.exists():
        info=json.loads(old.read_text())
        if info['version']==VERSION and info['source_revision']==manifest['revision']:return info
    exp=Exporter(base,spec)
    if spec['folder']=='SLAKE':slake(exp)
    elif spec['folder']=='GQA':gqa(exp)
    elif spec['folder']=='SEC_Contracts_VQA':legal(exp)
    else:generic(exp)
    return exp.finish(manifest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    specs=json.loads((ROOT/'data/multimodal_dataset_specs.json').read_text());submitted=set();done={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={}
        while len(done)<len(specs):
            for spec in specs:
                name=spec['folder']
                if name not in submitted and (ROOT/'data'/name/'asset_manifest.json').exists():
                    futures[pool.submit(prepare,spec)]=name;submitted.add(name)
            for future,name in list(futures.items()):
                if future.done():done[name]=future.result();del futures[future]
            if not args.watch and len(done)<len(specs) and not futures:raise RuntimeError('Some dataset downloads are not complete')
            if len(done)<len(specs):time.sleep(10)
    (WORK/'prepared_datasets.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n')
    print('ALL_DATASETS_PREPARED',len(done),flush=True)
