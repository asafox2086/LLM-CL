"""Freeze 16 author-KG facts using TRAIN questions only; never read test gold."""
import collections,csv,hashlib,io,json,math,re,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'exp/onevision_diagnostics_20261016'))
sys.path.insert(0,str(ROOT/'exp/five_domain_native_cl_20261016_sparse'))
from runtime_utils import write
OUT=ROOT/'data/OneVisionMechanisms/20261016'
STOP=set('what is are the of in this image which where how does do a an and to it shown organ body has that on for'.split())

def tokens(text):
    en=[x for x in re.findall('[a-z]+',text.lower()) if x not in STOP]
    zh=re.findall('[\u3400-\u9fff]+',text)
    return en+[s[i:i+2] for s in zh for i in range(len(s)-1)]

def frozen_write(path,obj):
    if path.exists():assert json.loads(path.read_text())==obj,'Frozen study changed'
    else:write(path,obj);path.chmod(0o444)

def prepare():
    source=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    manifest=json.loads(source.read_text());train=[];train_hashes={}
    for t in manifest['tasks']:
        if t['domain']!='medical':continue
        info=t['files']['train'];path=ROOT/info['path'];actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==info['sha256']
        train_hashes[info['path']]=actual;train.extend(json.loads(l) for l in path.read_text().splitlines())
    # The selection corpus accesses ONLY train questions, never answers, image metadata or test files.
    query=collections.Counter(token for row in train for token in tokens(row['question']))
    archive=ROOT/'data/SLAKE/raw/KG.zip';sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert sha=='c7637f47772ce8ce028737aa3068496adc5fe774afb290719c5affd7c074af32'
    facts=[]
    with zipfile.ZipFile(archive) as z:
        for name in ['KG/en_organ.csv','KG/en_organ_rel.csv','KG/organ.csv','KG/organ_rel.csv']:
            delimiter='#' if '/en_' in name else ','
            for index,parts in enumerate(csv.reader(io.StringIO(z.read(name).decode('utf-8-sig')),delimiter=delimiter)):
                if index==0 or len(parts)<3:continue
                subject,relation=parts[:2];value=delimiter.join(parts[2:]);assert subject and relation and value
                facts.append({'id':f'{name}:{index+1}','subject':subject,'relation':relation,'value':value,
                    'language':'en' if '/en_' in name else 'zh','source':name,'line':index+1})
    df=collections.Counter(t for f in facts for t in set(tokens(' '.join([f['subject'],f['relation'],f['value']]))))
    def relevance(f):
        ts=collections.Counter(tokens(' '.join([f['subject'],f['relation'],f['value']])))
        return sum(math.log1p(query[t])*math.log((1+len(facts))/(1+df[t]))*min(n,2) for t,n in ts.items() if t in query)/math.sqrt(1+sum(ts.values()))
    selected=[];unrelated=[]
    for lang in ['en','zh']:
        pool=sorted([f for f in facts if f['language']==lang],key=lambda f:(-relevance(f),hashlib.sha256(('42:'+f['id']).encode()).hexdigest()))
        chosen=pool[:8];selected.extend(chosen);used={f['id'] for f in chosen}
        # Low training-relevance pool; greedy match rendered character length, never target test correctness.
        low=pool[len(pool)//2:]
        for f in chosen:
            target=len(f['subject']+f['relation']+f['value'])
            candidates=[v for v in low if v['id'] not in used]
            v=min(candidates,key=lambda x:(abs(len(x['subject']+x['relation']+x['value'])-target),relevance(x),x['id']))
            unrelated.append(v);used.add(v['id'])
    permuted=[]
    for lang in ['en','zh']:
        group=[f for f in selected if f['language']==lang]
        # Relation mismatch is explicitly synthetic; originals remain intact.
        values=[f['value'] for f in group]
        shift=next((k for k in range(1,len(group)) if all(values[i]!=values[(i+k)%len(group)] for i in range(len(group)))),1)
        for i,f in enumerate(group):
            value=values[(i+shift)%len(group)]
            # A majority duplicate value makes a complete derangement impossible.
            # Use a deterministic DIFFERENT value; record this as mismatch, not a token-identical permutation.
            if value==f['value']:value=next(v for v in values if v!=f['value'])
            permuted.append(dict(f,value=value,synthetic_relation_mismatch=True))
    output={'source_url':'https://www.med-vqa.com/slake/','resource_type':'author-supplied benchmark knowledge graph; NOT independent clinical evidence',
        'source_revision':'a9083ce6c34ac3ffb17671a605962924d8a8f9e9','control_version':2,'KG_zip_sha256':sha,
        'selection':'8EN+8ZH; training-question lexical relevance only; no test records/gold accessed',
        'train_files_sha256':train_hashes,'train_question_ids':[r['id'] for r in train],'knowledge_fact_count':16,
        'correct':selected,'unrelated':unrelated,'permuted':permuted,
        'notes':['Anatomy organ/relations only; no treatment advice.','Wrong relations are offline negative controls only, not factual claims.',
                 'All mismatch values differ from originals; repeated-valued groups use a different existing value, not a strict length-preserving permutation.',
                 'Context lengths are recorded at inference; unrelated facts are approximately character-length matched, not claimed token-identical.',
                 'Positive results establish benefit of supplied benchmark knowledge, not general clinical competence.']}
    OUT.mkdir(parents=True,exist_ok=True);frozen_write(OUT/'knowledge16.json',output)
    return output

if __name__=='__main__':
    x=prepare();print(json.dumps({'facts':len(x['correct']),'source_sha256':x['KG_zip_sha256'],'selected_ids':[f['id'] for f in x['correct']]}))
