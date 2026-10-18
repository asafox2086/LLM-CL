"""Verify complete TRAIN-only feedback and real CPU updates during GPU testing.

Does not read moving TEST predictions or treat a running build as completed.
"""
import hashlib,json,math
from pathlib import Path
import torch
import onevision_answer_compatibility_core as core
from answer_f1 import answer_f1

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_diagnostics_20261016/answer_compatibility_20261018/v1'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    assert sha(p)==read(p.with_suffix('.json'))['sha256']
    return torch.load(p,map_location='cpu',weights_only=True)


def main():
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True)
    attempt=read(REPORT/'build_attempt.json');run=ROOT/attempt['run']
    assert run.resolve().is_relative_to((ROOT/'exp/onevision_answer_compatibility_20261018/v1').resolve())
    spec=read(REPORT/'CPU_PLAN_CHECK.json')['spec'];assert attempt['spec']==read(run/'spec.json')==spec
    for n,s in spec['source_sha256'].items():assert sha(ROOT/n)==s
    saved=load(run/'feedback_state.pt');raw=load(run/'router_raw_gradients.pt')
    event=read(run/'router_TRAIN.json')
    destination=REPORT/'TRAIN_FIT_VERIFIED.json'
    signature={n:sha(run/n) for n in ['feedback_state.pt','router_raw_gradients.pt','router_TRAIN.json']}
    if destination.exists():
        old=read(destination)
        if old['TRAIN_artifact_sha256']==signature and old['auditor_sha256']==sha(Path(__file__)):return
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest_path)==spec['parent_spec']['manifest_sha256']
    train={}
    for task in read(manifest_path)['tasks']:
        f=task['files']['train'];p=ROOT/f['path'];assert sha(p)==f['sha256']
        train.update({r['id']:r for r in [json.loads(s) for s in p.read_text().splitlines()]})
    assert len(train)==1313 and len(saved['TRAIN_ids'])==20
    assert saved['TRAIN_ids']==event['TRAIN_ids']==raw['TRAIN_ids']
    source=ROOT/'summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1/build_complete.json'
    proof=read(source);assert sha(source)==spec['parent_complete_sha256']
    p=ROOT/proof['run']/'final_state.pt';assert sha(p)==spec['fixed_pool_sha256']
    state=torch.load(p,map_location='cpu',weights_only=True);keys=sorted(state['pool'])
    assert keys==[12,18,19] and state['memory_ids']==saved['TRAIN_ids']
    scores=[]
    for k in keys:
        folder=run/f'evaluations/TRAIN_teacher_state{k}'
        p=folder/'TRAIN_memory_image.jsonl'
        assert sha(p)==read(folder/'durable_complete.json')['files'][p.name]
        assert sha(p)==saved['teacher_sha256'][str(k)]==event['teacher_sha256'][str(k)]
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        assert [r['id'] for r in rows]==saved['TRAIN_ids']
        for r in rows:
            assert r['references']==train[r['id']]['answers']
            assert answer_f1(r['prediction'],r['references'])==r['scores']['token_f1']
        scores.append([r['scores']['token_f1']/100 for r in rows])
    matrix=torch.tensor(scores,dtype=torch.float32);assert torch.equal(matrix,saved['scores'])
    groups=[tuple(train[i]['images']) for i in saved['TRAIN_ids']]
    theta,events,gradients=core.fit(state['memory_features'],matrix,groups)
    assert torch.equal(theta,saved['theta']) and events==event['events']
    assert len(gradients)==len(raw['gradients'])==40 and all(torch.equal(x,y) for x,y in zip(gradients,raw['gradients']))
    assert all(math.isfinite(e['loss']) and e['LR']==.03 for e in events)
    prototypes=torch.stack([state['prototypes'][k] for k in keys]);routes=[]
    for i,feature in enumerate(state['memory_features']):
        before,_=core.infer(feature,state['memory_features'],matrix,torch.zeros_like(theta),keys,prototypes)
        after,_=core.infer(feature,state['memory_features'],matrix,theta,keys,prototypes)
        routes.append(dict(TRAIN_id=saved['TRAIN_ids'][i],fixed=before,learned=after))
    result=dict(CPU_only=True,TRAIN_stage_complete_not_GPU_build_complete=True,
        auditor_sha256=sha(Path(__file__)),TRAIN_artifact_sha256=signature,
        source_pool_sha256=spec['fixed_pool_sha256'],teacher_predictions=60,
        informative_TRAIN_QA=int((matrix.max(dim=0).values>matrix.min(dim=0).values).sum()),
        actual_metric_parameters=theta.numel(),actual_nonzero_gradient_steps=sum(int(bool(torch.count_nonzero(g))) for g in gradients),
        actual_changed_dimensions=int(torch.count_nonzero(theta)),theta_l2=float(theta.norm()),
        first_event=events[0],last_event=events[-1],
        all40_raw_gradients_and_parameters_exactly_replayed=True,TRAIN_routes=routes,
        TRAIN_route_changes=sum(r['fixed']!=r['learned'] for r in routes),
        TEST_predictions_or_answers_parsed=False,new_GPU_or_model_training=False,
        limits=['Head supervision includes all20 TRAIN scores; image-excluded neighbors are not independent validation.',
                'TRAIN hard routing above includes own anchor and is resubstitution, not unseen performance.',
                'Parameter/gradient changes do not establish full TEST route changes or Answer F1 gain.',
                'CPU audit replay is validation overhead, not another proposed method condition.'])
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['teacher_predictions','informative_TRAIN_QA',
        'actual_nonzero_gradient_steps','actual_changed_dimensions','TRAIN_route_changes',
        'all40_raw_gradients_and_parameters_exactly_replayed']}))


if __name__=='__main__':main()
