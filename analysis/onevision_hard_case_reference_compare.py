"""Sealed hard-cell versus same155 native sequential reference, CPU only."""
import hashlib,json,random
from collections import defaultdict
from pathlib import Path
from answer_f1 import answer_f1
from onevision_acquisition_budget_bridge import interval

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'summary/onevision_diagnostics_20261016'
REPORT=BASE/'warm_quality_20261018/predeclared_hard_case'
DOMAINS=['natural','medical','geography','finance','traffic']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    target=REPORT/'REFERENCE_COMPARISON.json'
    assert not target.exists(),'Do not repeat an unchanged comparison as progress'
    inputs={}
    def read(p):
        inputs[str(p.relative_to(ROOT))]=sha(p)
        return json.loads(p.read_text())
    case=read(REPORT/'VERIFIED_EVIDENCE.json')
    assert case['condition_complete'] and case['TEST_view_selection'] is False
    assert case['report_source_sha256']==sha(ROOT/'analysis/onevision_predeclared_hard_case_report.py')
    snapshots=read(REPORT/'INPUT_SNAPSHOT_MAP.json')['snapshots']
    for rel,expected in case['input_sha256'].items():
        p=ROOT/rel
        if sha(p)!=expected:
            item=snapshots[rel];p=ROOT/item['archived'];assert item['sha256']==expected
        assert sha(p)==expected,rel
        inputs[str(p.relative_to(ROOT))]=expected
    native_path=BASE/'deterministic_native/complete.json';native=read(native_path)
    assert native['complete']
    for rel,expected in native['source_sha256'].items():assert sha(ROOT/rel)==expected
    attempt=read(BASE/'warm_quality_20261018/pilot_attempt.json');current=ROOT/attempt['run']
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest=read(manifest_path);assert sha(manifest_path)==attempt['spec']['parent_spec']['manifest_sha256']
    images={}
    for task in manifest['tasks']:
        p=ROOT/task['files']['test']['path'];assert sha(p)==task['files']['test']['sha256']
        for s in p.read_text().splitlines():
            r=json.loads(s);images[r['id']]=r['images']
    domains=[];scores={v:{t:[] for t in range(1,20)} for v in ['reference','combined']};blocks={}
    for domain in DOMAINS:
        rel=f'evaluations/sequential_T19/{domain}_image.jsonl';p=ROOT/native['run']/rel
        assert sha(p)==native['artifact_sha256'][rel];inputs[str(p.relative_to(ROOT))]=sha(p)
        before=[json.loads(s) for s in p.read_text().splitlines()]
        p=current/f'evaluations/quality_hard/{domain}_image.jsonl';inputs[str(p.relative_to(ROOT))]=sha(p)
        after=[json.loads(s) for s in p.read_text().splitlines()]
        assert [r['id'] for r in before]==[r['id'] for r in after]
        values=[];groups=defaultdict(list);native_values=[];combined_values=[]
        for a,b in zip(before,after):
            assert a['references']==b['references'] and a['native_task']==b['native_task']
            assert ('prediction' in a)==('prediction' in b)
            if 'prediction' not in a:continue
            x=answer_f1(a['prediction'],a['references']);y=answer_f1(b['prediction'],b['references'])
            assert x==a['scores']['token_f1'] and y==b['scores']['token_f1']
            scores['reference'][a['native_task']].append(x);scores['combined'][b['native_task']].append(y)
            values.append(y-x);native_values.append(x);combined_values.append(y)
            groups[tuple(images[a['id']])].append(y-x)
        blocks[domain]=list(groups.values())
        domains.append({'domain':domain,'QA':len(values),'reference_F1':sum(native_values)/len(values),
                        'combined_F1':sum(combined_values)/len(values),
                        'paired_difference':interval(values,[images[r['id']] for r in before if 'prediction' in r])})
    assert sum(d['QA'] for d in domains)==155 and all(scores[v][t] for v in scores for t in range(1,20))
    rng=random.Random(42);draws=[]
    for _ in range(10000):
        effects=[]
        for domain in DOMAINS:
            cluster=blocks[domain];selected=[cluster[rng.randrange(len(cluster))] for _ in cluster]
            effects.append(sum(map(sum,selected))/sum(map(len,selected)))
        draws.append(sum(effects)/5)
    draws.sort()
    aggregate={v:{'AP':sum(sum(x)/len(x) for x in ts.values())/19,
                  'domain_macro':sum(d['reference_F1' if v=='reference' else 'combined_F1'] for d in domains)/5}
               for v,ts in scores.items()}
    result={'CPU_only':True,'GPU_operations':0,'optimizer_updates':0,'scope':'Same500NLL/155generated diagnostic only',
            'TEST_selection':False,'domains':domains,'aggregate':aggregate,
            'AP_difference':aggregate['combined']['AP']-aggregate['reference']['AP'],
            'domain_macro_difference':aggregate['combined']['domain_macro']-aggregate['reference']['domain_macro'],
            'domain_stratified_image_bootstrap_95CI':[draws[249],draws[9749]],'replicates':10000,'seed':42,
            'multiple_comparison_corrected':False,'input_sha256':inputs,'report_source_sha256':sha(Path(__file__)),
            'limits':['Diagnostic point advantage is not large or established on complete21454.',
                      'Fixed-weight image intervals do not establish multi-training-seed superiority.',
                      'NativeAnyRes research is separate from formalV2; extra pool/router/compute costs differ.',
                      'Do not compare diagnostic44.165 against full-test41.361 as if the cohorts matched.']}
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['aggregate','AP_difference','domain_macro_difference','domain_stratified_image_bootstrap_95CI']}))

if __name__=='__main__':main()
