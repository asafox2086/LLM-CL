"""Read completed native F1 outputs once; no model calls or rescoring."""
from pathlib import Path
import json,hashlib,math,statistics
R=Path('/data2/liyapeng_grp/program/LLMCL')
O=R/'exp/expert_mass_control_20261104/runs/diagnostic_v1'
S=R/'summary/expert_mass_control_20261104'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def completed(p):
    assert not (p/'FAILURE.json').exists()
    assert sha(p/'RESULT.json')==read(p/'COMPLETE.json')['result_sha256']
    return read(p/'RESULT.json')
def index(result):
    values={}
    for v in result['records']:
        assert math.isfinite(v['F1']) and 0<=v['F1']<=100
        a=values.setdefault(v['arm'],{})
        assert v['id'] not in a;a[v['id']]=v
    for a,rows in values.items():
        assert len(rows)==40
        assert {d:sum(x['domain']==d for x in rows.values()) for d in result['domain_F1'][a]}=={d:8 for d in result['domain_F1'][a]}
        assert abs(math.fsum(x['F1'] for x in rows.values())/40-result['macro_F1'][a])<1e-10
    return values
def pair(a,b):
    assert a.keys()==b.keys()
    ds=[a[k]['F1']-b[k]['F1'] for k in sorted(a)]
    return dict(mean_delta=math.fsum(ds)/40,better=sum(x>0 for x in ds),worse=sum(x<0 for x in ds),tie=sum(x==0 for x in ds),answer_same=sum(a[k]['prediction']==b[k]['prediction'] for k in a))
def main():
    if not (O/'COMPLETE.json').exists() or not (S/'EXIT.json').exists():
        print('WAITING: final RESULT and real worker EXIT required');return
    ex=read(S/'EXIT.json');assert ex['returncode']==0 and not Path('/proc',str(ex['pid'])).exists()
    dest=S/'PAIRED_ANALYSIS.json';assert not dest.exists(),'Do not replay consumer'
    result=completed(O)
    assert read(O/'STARTED.json')['source_sha256']==sha(R/'analysis/expert_mass_control_20261104.py')
    assert len(read(O/'CONTROLS_COMPLETE.json'))==5
    values=index(result)
    subset_path=R/'exp/expert_subset_f1_20261104/runs/diagnostic_v1'
    marginal_path=R/'exp/expert_marginal_f1_20261104/runs/diagnostic_v1'
    subset=index(completed(subset_path));marginal=index(completed(marginal_path))
    expected={'all_scaled_eighth'}|{'only_mass_matched_'+str(k) for k in range(8)}
    assert values.keys()==expected
    source_proofs=[]
    for arm,rows in values.items():
        paths=list((O/arm).rglob('*_train_predictions.jsonl'));assert len(paths)==40
        seen=set()
        for p in paths:
            v=read(p);q=v['id'];assert q not in seen;seen.add(q)
            assert v['scores']['token_f1']==rows[q]['F1'] and v['prediction']==rows[q]['prediction']
            scores=read(p.with_suffix('.scores.json'));assert scores['count']==1 and scores['token_f1']==v['scores']['token_f1']
            source_proofs.append(dict(path=str(p),sha256=sha(p)))
        assert seen==rows.keys()==subset['fixed_all'].keys()==marginal['experts_off'].keys()
    slots=[]
    for k in range(8):
        a=values['only_mass_matched_'+str(k)];b=marginal['only_'+str(k)]
        slots.append(dict(slot=k,matched_vs_original_single=pair(a,b),matched_vs_full_fixed=pair(a,subset['fixed_all']),matched_macro=math.fsum(x['F1'] for x in a.values())/40))
    report=dict(scope='40 TRAIN questions; coefficient-sum control, not output-norm matching',
        mass_result_sha256=sha(O/'RESULT.json'),exit_sha256=sha(S/'EXIT.json'),
        subset_result_sha256=sha(subset_path/'RESULT.json'),marginal_result_sha256=sha(marginal_path/'RESULT.json'),
        scaled_all_vs_full=pair(values['all_scaled_eighth'],subset['fixed_all']),
        scaled_all_vs_off=pair(values['all_scaled_eighth'],marginal['experts_off']),slots=slots,
        matched_macro_range=[min(x['matched_macro'] for x in slots),max(x['matched_macro'] for x in slots)],
        matched_macro_median=statistics.median(x['matched_macro'] for x in slots),
        original_scores_reused=True,rescored=False,source_proofs=source_proofs,
        limits=['training diagnostic sample','same-number slots across layers not semantically aligned','no equivalence margin preregistered','does not identify training cause or establish full-TEST mediation'])
    with dest.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='source_proofs'}))
if __name__=='__main__':main()
