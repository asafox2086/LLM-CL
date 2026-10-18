"""Small TRAIN-feedback metric; no backbone, QA loader or TEST selection.

TRAIN scores must be original Answer F1 divided by100, not NLL or domain IDs.
The caller supplies actual TRAIN image groups for leave-one-image-out fitting.
"""
import torch
import torch.nn.functional as F

STEPS=40
LR=.03
KERNEL_TEMPERATURE=.1
EXPERT_TEMPERATURE=.1
LOG_WEIGHT_LIMIT=2.
L2=.01


def image_mask(image_groups):
    assert all(isinstance(g,(set,frozenset,tuple)) and len(g)>0 for g in image_groups)
    return torch.tensor([[not(set(a)&set(b)) for b in image_groups] for a in image_groups],dtype=torch.bool)


def weights(query,memory,theta,allowed=None):
    assert query.device.type==memory.device.type==theta.device.type=='cpu'
    assert query.ndim==memory.ndim==2 and query.shape[1]==memory.shape[1]==theta.numel()
    scale=torch.exp(theta/2)
    q=F.normalize(query.float()*scale,dim=1)
    m=F.normalize(memory.float()*scale,dim=1)
    logits=q@m.T/KERNEL_TEMPERATURE
    if allowed is not None:
        assert allowed.shape==logits.shape and allowed.dtype==torch.bool
        assert allowed.any(dim=1).all()
        logits=logits.masked_fill(~allowed,float('-inf'))
    return F.softmax(logits,dim=1)


def objective(memory,scores,theta,allowed):
    assert scores.ndim==2 and scores.shape[1]==len(memory)
    assert torch.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()
    estimated=weights(memory,memory,theta,allowed)@scores.T
    probabilities=F.softmax(estimated/EXPERT_TEMPERATURE,dim=1)
    achieved=(probabilities*scores.T).sum(dim=1)
    regret=scores.max(dim=0).values-achieved
    return regret.mean()+L2*theta.square().mean(),regret.detach()


def fit(memory,scores,image_groups):
    assert len(memory)==len(image_groups)>=2 and memory.ndim==2
    assert torch.isfinite(memory).all()
    allowed=image_mask(image_groups)
    assert allowed.any(dim=1).all(),'Need another TRAIN image; no identity leakage fallback'
    theta=torch.nn.Parameter(torch.zeros(memory.shape[1]))
    optimizer=torch.optim.AdamW([theta],lr=LR,weight_decay=0)
    records=[];raw_gradients=[]
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss,regret=objective(memory,scores,theta,allowed)
        assert torch.isfinite(loss)
        loss.backward();assert theta.grad is not None and torch.isfinite(theta.grad).all()
        raw_gradients.append(theta.grad.detach().clone())
        norm=float(theta.grad.norm());before=theta.detach().clone()
        optimizer.step()
        with torch.no_grad():theta.clamp_(-LOG_WEIGHT_LIMIT,LOG_WEIGHT_LIMIT)
        records.append(dict(step=step+1,loss=float(loss.detach()),mean_TRAIN_LOO_soft_regret=float(regret.mean()),
            raw_gradient_l2=norm,actual_parameter_delta_l2=float((theta.detach()-before).norm()),LR=LR))
    return theta.detach().clone(),records,raw_gradients


def infer(feature,memory,scores,theta,keys,prototypes):
    assert keys==sorted(set(keys)) and scores.shape==(len(keys),len(memory))
    assert prototypes.shape==(len(keys),feature.numel())
    kernel=weights(feature.reshape(1,-1),memory,theta)[0]
    expected=scores@kernel
    tied=expected==expected.max()
    # Exact quality ties use input geometry, not arbitrary recency or TEST gold.
    scale=torch.exp(theta/2)
    geometric=F.normalize(prototypes*scale,dim=1)@F.normalize(feature*scale,dim=0)
    geometric=geometric.masked_fill(~tied,float('-inf'))
    chosen=int(geometric.argmax())
    return keys[chosen],dict(estimated_TRAIN_Answer_F1=(100*expected).tolist(),
        exact_score_tie_count=int(tied.sum()),kernel_max=float(kernel.max()))


def checks():
    torch.set_num_threads(1)
    memory=F.normalize(torch.tensor([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]]),dim=1)
    scores=torch.tensor([[1.,1.,0.,0.],[0.,0.,1.,1.]])
    allowed=~torch.eye(4,dtype=torch.bool)
    before=float(objective(memory,scores,torch.zeros(2),allowed)[0])
    theta,events,grads=fit(memory,scores,[(f'image{i}',) for i in range(4)])
    after=float(objective(memory,scores,theta,allowed)[0])
    assert after<before and theta[0]>theta[1]
    assert any(g.norm()>0 for g in grads) and not torch.equal(theta,torch.zeros_like(theta))
    # Image-group exclusion, not just diagonal exclusion.
    group_allowed=torch.tensor([[a!=b for b in ['same','same','other','other']]
                               for a in ['same','same','other','other']])
    k=weights(memory,memory,theta,group_allowed)
    assert torch.count_nonzero(k[~group_allowed])==0
    assert torch.allclose(k.sum(dim=1),torch.ones(4))
    overlap=image_mask([('a','b'),('b','c'),('d',),('e',)])
    assert not overlap[0,1] and overlap[0,2]
    # Entirely uninformative TRAIN feedback must be reported as inactive.
    zero_theta,zero_events,zero_grads=fit(memory,torch.zeros_like(scores),[(str(i),) for i in range(4)])
    assert torch.equal(zero_theta,torch.zeros_like(zero_theta))
    assert all(not torch.count_nonzero(g) for g in zero_grads)
    chosen,_=infer(memory[0],memory,torch.zeros_like(scores),torch.zeros(2),[12,18],
        torch.tensor([[-1.,0.],[1.,0.]]))
    assert chosen==18
    assert infer(memory[0],memory,scores,theta,[12,18],torch.eye(2))[0]==12
    try:fit(memory,scores,[('same',)]*4)
    except AssertionError:pass
    else:raise AssertionError('Same-image identity leakage negative control not detected')
    return dict(CPU_only=True,checks=8,synthetic_loss_before=before,synthetic_loss_after=after,
        actual_backbone_or_TRAIN_feedback_used=False,GPU_started=False,TEST_performance_evidence=False)


if __name__=='__main__':
    import json
    print(json.dumps(checks()))
