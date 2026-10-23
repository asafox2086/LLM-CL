"""Once-only integrity and scalar consumer; no model/F1/bootstrap."""
import json, math
from core import *

def main():
    assert not (REPORT/'VERIFIED.json').exists(),'Never rerun a sealed diagnostic verifier'
    ready=json.loads((REPORT/'READY.json').read_text());c=json.loads((REPORT/'COMPLETE.json').read_text())
    assert c['complete'] and c['ready_sha256']==sha(REPORT/'READY.json')
    assert 0<c['model_RUN_seconds']<=BUDGET and c['full_graph_reload']
    for key in ('sources','dependencies','evidence'):
        for n,h in ready[key].items():assert sha(ROOT/n)==h,n
    for n,h in c['file_sha256'].items():assert sha(ROOT/n)==h,n
    before=json.loads((WORK/'before.json').read_text());after=json.loads((WORK/'after.json').read_text())
    ids=[v['id'] for v in ready['data']['diagnostic']]
    assert [v['id'] for v in before]==[v['id'] for v in after]==ids and len(ids)==65
    for a,b in zip(before,after):
        assert a['input_tensors']==b['input_tensors'] and a['raw_pixels']==b['raw_pixels'] and a['parent_NLL']==b['parent_NLL']
        assert all(math.isfinite(r[k]) for r in (a,b) for k in ('parent_NLL','student_NLL','feature_error'))
    es=[json.loads(p.read_text()) for p in sorted((WORK/'updates').glob('*.json'))]
    assert len(es)==17 and [e['step'] for e in es]==list(range(1,18))
    assert [i for e in es for i in e['QA_ids']]==[v['id'] for v in ready['data']['calibration']]
    assert all(len(e['input_details'])==len(e['QA_ids']) for e in es)
    details=[r['details'] for r in before+after]+[d for e in es for d in e['input_details']]
    cost=c['teacher_feature_cost'];assert cost['visual_requests']==len(details)==390
    assert cost['actual_extractor_calls']==sum(not d['visual_cache_hit'] for d in details)
    assert cost['cache_hits']==sum(bool(d['visual_cache_hit']) for d in details)
    assert all(e['optimizer']['parameter_IDs_match'] and e['optimizer']['moment_devices']==['cpu'] for e in es)
    layout=json.loads((WORK/'STUDENT_LAYOUT.json').read_text())
    coordinates=sum(math.prod(p['shape']) for p in layout.values())
    assert coordinates==sum(p['coordinates'] for p in layout.values())==ready['student_training_coordinates']==15239504
    assert all(e['optimizer']['coordinates']==coordinates and e['optimizer']['gradient_bytes']==coordinates*4 and
               e['optimizer']['moment_bytes']==coordinates*8 for e in es)
    assert all(math.isfinite(e['raw_gradient_norm']) and e['raw_gradient_norm']>=0 and
               all(math.isfinite(lr) and lr>0 for lr in e['learning_rate_used']) for e in es)
    assert c['original_parent_digest_before']==c['original_parent_digest_after']
    assert c['result']==outcome(before,after,c['result']['actual_final_layout'])
    cp=WORK/c['checkpoint']['path'];assert sha(cp)==c['checkpoint']['sha256']
    assert c['extra_KD_QA']==260 and c['diagnostic_NLL_forwards']==195 and c['training_updates']==17
    assert c['language_updates']==c['DEV_QA']==c['TEST_QA']==0
    write(REPORT/'VERIFIED.json',dict(complete=True,result=c['result'],complete_sha256=sha(REPORT/'COMPLETE.json'),
        checkpoint_sha256=c['checkpoint']['sha256'],model_RUN_seconds=c['model_RUN_seconds'],
        no_F1_or_bootstrap=True,not_full_quality_confirmation=True,automatic_continual_training=False))

if __name__=='__main__':main()
