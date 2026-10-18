"""CPU checks of actual old/new branch dispatch, without loading a backbone."""
import hashlib,json,os
from pathlib import Path
import torch
import onevision_answer_compatibility_trial as worker


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)
    x=object.__new__(worker.FeedbackExperiment);x.feedback=None
    original=worker.parent.CacheExperiment.activate_route
    try:
        worker.parent.CacheExperiment.activate_route=lambda obj,row:setattr(obj,'last_route',{'old_branch':True})
        x.activate_route({'question':'fixture','images':['x'],'answers':['poison'],'_task':999})
        assert x.last_route=={'old_branch':True}
    finally:worker.parent.CacheExperiment.activate_route=original
    state=dict(pool={12:{'fixture':12},18:{'fixture':18},19:{'fixture':19}},
        memory_features=torch.tensor([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]),
        prototypes={12:torch.tensor([1.,0.]),18:torch.tensor([0.,1.]),19:torch.tensor([-1.,0.])})
    scores=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,1.],[0.,0.,1.,0.]])
    calls=[];x.feature=lambda row:torch.tensor([1.,0.]);x.restore=lambda value:calls.append(value)
    for view in worker.VIEWS:
        x.set_feedback(state,scores,torch.zeros(2),view)
        x.activate_route({'question':'fixture','images':['x'],'answers':['poison'],'_task':999})
        first=dict(x.last_route);before=len(calls)
        x.activate_route({'question':'fixture','images':['x'],'answers':['different'],'_task':1})
        assert x.last_route==first and len(calls)==before and first['route_state']==12
        assert first['route_mode']==view
    assert not torch.cuda.is_initialized()
    report=worker.REPORT;report.mkdir(parents=True,exist_ok=True)
    result=dict(CPU_only=True,actual_old_branch_forwarded=True,both_new_routes_active=True,
        gold_and_task_field_changes_do_not_change_route=True,same_state_not_reloaded=True,
        fixture_only_no_backbone_or_real_QA=True,CUDA_initialized=False,GPU_started=False,
        source_sha256={str(p.relative_to(worker.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),Path(worker.__file__)]})
    (report/'CPU_DISPATCH_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
