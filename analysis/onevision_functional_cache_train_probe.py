"""Chronological functional-cache screening on existing TRAIN-probe NLL.

Real saved sequential checkpoints, no TEST files or new model forwards.
Only already-arrived anchors and past/current retained states are queried.
This is a 20-TRAIN screening study, not a deployed router or F1 result.
"""
import hashlib,itertools,json,math,os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'summary/onevision_diagnostics_20261016'
REPORT=BASE/'functional_state_cache_20261018'
PENALTY=.02  # preregistered nats/QA per extra retained state, not TEST fitted
KMAX=3

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def pick(stage,available,arrival,matrix,kmax,queried):
    assert stage in available and all(1<=s<=stage for s in available)
    ids=[i for i,t in arrival.items() if t<=stage]
    assert ids
    costs=[]
    for count in range(1,min(kmax,len(available))+1):
        for subset in itertools.combinations(sorted(available),count):
            if stage not in subset:continue
            assert all(arrival[i]<=stage for i in ids)
            for s in subset:
                for i in ids:queried.setdefault((s,i),stage)
            nll=sum(min(matrix[s][i] for s in subset) for i in ids)/len(ids)
            costs.append((nll+PENALTY*(len(subset)-1),len(subset),tuple(-s for s in sorted(subset,reverse=True)),subset,nll))
    best=min(costs)
    return {'stage':stage,'arrived_TRAIN_ids':ids,'candidate_states':sorted(available),
            'selected_states':list(best[3]),'retained_count':best[1],
            'best_available_state_per_TRAIN_mean_NLL':best[4],
            'regularized_objective':best[0],
            'current_state_only_TRAIN_NLL':sum(matrix[stage][i] for i in ids)/len(ids),
            'decision_uses_future_QA_or_states':False}

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)
    assert not (REPORT/'TRAIN_SCREEN.json').exists(),'Do not repeat a completed screen'
    protocol=REPORT/'PROTOCOL.md';assert protocol.exists()
    source={str(protocol.relative_to(ROOT)):sha(protocol)}
    p=BASE/'deterministic_native/complete.json';proof=json.loads(p.read_text());assert proof['complete']
    source[str(p.relative_to(ROOT))]=sha(p)
    for rel,expected in proof['source_sha256'].items():assert sha(ROOT/rel)==expected
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest=json.loads(manifest_path.read_text());source[str(manifest_path.relative_to(ROOT))]=sha(manifest_path)
    arrival_all={}
    for task in manifest['tasks']:
        info=task['files']['train'];p=ROOT/info['path'];assert sha(p)==info['sha256'];source[info['path']]=sha(p)
        for s in p.read_text().splitlines():arrival_all[json.loads(s)['id']]=task['task']
    assert len(arrival_all)==1313
    matrix={};probe_ids=None
    for stage in range(1,20):
        values={}
        for domain in ['natural','medical','geography','finance','traffic']:
            rel=f'raw_probe_gradients/sequential_T{stage:02d}/{domain}.pt';p=ROOT/proof['run']/rel
            assert sha(p)==proof['artifact_sha256'][rel];source[str(p.relative_to(ROOT))]=sha(p)
            data=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
            assert data['train_only'] and data['eval_mode_no_dropout']
            assert len(data['ids'])==len(data['nll'])==4
            for ident,loss in zip(data['ids'],data['nll']):
                assert ident in arrival_all and ident not in values and math.isfinite(loss)
                values[ident]=float(loss)
            del data
        if probe_ids is None:probe_ids=set(values)
        assert set(values)==probe_ids and len(values)==20
        matrix[stage]=values
    arrival={i:arrival_all[i] for i in sorted(probe_ids)}
    traces={};queries={}
    for kmax in [1,KMAX]:
        previous=[];trace=[];queried={}
        for stage in range(1,20):
            candidates=set(previous)|{stage}
            event=pick(stage,candidates,arrival,matrix,kmax,queried)
            previous=event['selected_states'];trace.append(event)
        traces[str(kmax)]=trace
        queries[str(kmax)]={'distinct_state_TRAIN_pairs_needed_if_run_online':len(queried),
                           'QA_arrivals_and_past_states_respected':all(arrival[i]<=t and s<=t for (s,i),t in queried.items()),
                           'first_queries':[{'state':s,'TRAIN_id':i,'decision_stage':t} for (s,i),t in sorted(queried.items())],
                           'note':'Pair cost is charged when QA/state first coexist; no TEST oracle is used.'}
    result={'CPU_only':True,'new_GPU_forwards_or_optimizer_updates':0,'TEST_opened':False,
            'parent_TRAIN_QA':1313,'screen_TRAIN_anchors':20,'scope':'Pre-existing four-per-domain TRAIN probes only; NOT full memory policy or F1',
            'penalty_nats_per_QA_per_extra_state':PENALTY,'Kmax':[1,KMAX],
            'count_is_not_TEST_optimal_LoRA_number':True,'decision_rule':'current mandatory; previous retained + current candidates; exact subset objective with complexity cost',
            'arrival_stage':arrival,'actual_saved_TRAIN_NLL':matrix,'traces':traces,'virtual_model_pair_cost':queries,
            'input_sha256':source,'source_sha256':sha(Path(__file__)),
            'limits':['Original probes are domain-stratified diagnostics, not a representative/full TRAIN memory.',
                      'Old saved future-anchor losses are never exposed to a decision before that anchor arrives.',
                      'Lowest NLL across states is a TRAIN coverage objective, not an actual learned TEST router.',
                      'No TEST F1 or automatic globally optimal capacity is inferred; GPU prototype remains unimplemented.',
                      'Snapshot storage, temporary candidates, optimizer, feature extraction and routing costs still require measurement.']}
    (REPORT/'TRAIN_SCREEN.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'screen_only_TRAIN':True,'K3_counts':[r['retained_count'] for r in traces['3']],
                      'K3_states':[r['selected_states'] for r in traces['3']],
                      'final_K1_NLL':traces['1'][-1]['best_available_state_per_TRAIN_mean_NLL'],
                      'final_K3_NLL':traces['3'][-1]['best_available_state_per_TRAIN_mean_NLL'],'virtual_pair_cost':queries}))

if __name__=='__main__':main()
