"""Weights-only common-right-subspace feasibility, not a model F1 experiment.

Rank8 is the single prespecified candidate; rank24 is a numerical full-union
control. Fit actual deltaW products, not unidentifiable A-factor coordinates.
Never imported by a GPU worker; importing this module does not change devices.
"""
import hashlib,json,math,time
from pathlib import Path
import psutil
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1'
REPORT=ROOT/'summary/onevision_diagnostics_20261016/shared_basis_probe_20261018/v1'
FAMILY=ROOT/'exp/onevision_shared_basis_probe_20261018/v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def norm(a,b):
    _,r=torch.linalg.qr(a.double().T,mode='reduced')
    return float((b.double()@r.T).norm())


def project(pairs,rank):
    hs=[]
    for p in pairs:
        a,b=p['A'].double(),p['B'].double();e,q=torch.linalg.eigh(b.T@b)
        assert float(e.min())>=-1e-9*max(float(e.max()),1e-20)
        hs.append(e.clamp_min(0).sqrt().unsqueeze(1)*q.T@a)
    _,s,vh=torch.linalg.svd(torch.cat(hs),full_matrices=False)
    assert rank<=len(s)
    common=vh[:rank].clone()
    for row in common:
        if row[int(row.abs().argmax())]<0:row.neg_()
    # Store the actual proposed tensors in the same FP32 as original adapters.
    a=common.float().contiguous()
    bs=[(p['B'].double()@p['A'].double()@a.double().T).float().contiguous() for p in pairs]
    return a,bs,float(s[:rank].square().sum()/s.square().sum()) if s.square().sum()>0 else 1.


def relative_error(original,a,b):
    aa=torch.cat([original['A'].double(),a.double()])
    bb=torch.cat([original['B'].double(),-b.double()],dim=1)
    n=norm(original['A'],original['B']);error=norm(aa,bb)
    return dict(original_deltaW_norm=4*n,deltaW_error_norm=4*error,
                relative_error=error/n if n else None)


def difference_error(x,y,a,bx,by):
    original_a=torch.cat([y['A'].double(),x['A'].double()])
    original_b=torch.cat([y['B'].double(),-x['B'].double()],dim=1)
    n=norm(original_a,original_b)
    aa=torch.cat([original_a,a.double()])
    bb=torch.cat([original_b,-(by.double()-bx.double())],dim=1)
    e=norm(aa,bb)
    return dict(original_deltaW_difference_norm=4*n,error_norm=4*e,relative_error=e/n if n else None)


def checks():
    # Same row space with different coordinates must remain representable.
    a=torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.]])
    b=torch.tensor([[1.,2.],[-1.,3.]])
    g=torch.tensor([[2.,1.],[0.,.5]])
    pairs=[dict(A=a,B=b),dict(A=g@a,B=(b@torch.linalg.inv(g)))]
    shared,bs,_=project(pairs,2)
    assert all(relative_error(p,shared,c)['relative_error']<5e-6 for p,c in zip(pairs,bs))
    # Orthogonal functions cannot be compressed to the same rank without loss.
    independent=[dict(A=torch.eye(4)[:2],B=torch.eye(2)),dict(A=torch.eye(4)[2:],B=torch.eye(2))]
    shared,bs,_=project(independent,2)
    assert sum(relative_error(p,shared,c)['deltaW_error_norm'] for p,c in zip(independent,bs))>1.
    shared,bs,_=project(independent,4)
    assert all(relative_error(p,shared,c)['relative_error']<5e-6 for p,c in zip(independent,bs))
    # Stable low-rank norm must agree with dense product on this fixture.
    assert abs(norm(a,b)-float((b.double()@a.double()).norm()))<1e-12
    return dict(gauge_control=True,lossy_orthogonal_negative_control=True,
        full_union_control=True,stable_norm_dense_control=True)


def snapshot(label):
    m=psutil.Process().memory_full_info()
    return dict(label=label,unix=time.time(),CPU_RSS=m.rss,CPU_PSS=getattr(m,'pss',None),CPU_USS=getattr(m,'uss',None))


def main():
    torch.set_num_threads(1);proof_path=SOURCE/'build_complete.json';proof=read(proof_path)
    assert proof['complete'];run=ROOT/proof['run']
    assert run.resolve().is_relative_to((ROOT/'exp/onevision_state_cache_20261018/v1').resolve())
    for name,s in proof['artifact_sha256'].items():
        p=run/name;assert p.resolve().is_relative_to(run.resolve()) and sha(p)==s
    REPORT.mkdir(parents=True,exist_ok=True);FAMILY.mkdir(parents=True,exist_ok=True)
    out=REPORT/'EVIDENCE.json'
    if out.exists():
        old=read(out)
        if old['source_complete_sha256']==sha(proof_path) and old['analyzer_sha256']==sha(Path(__file__)):return
    mechanism=checks();samples=[snapshot('before_source_load')]
    state=torch.load(run/'final_state.pt',map_location='cpu',weights_only=True);pool=state['pool']
    keys=sorted(pool);assert keys==[12,18,19]
    layers=list(pool[keys[0]]);assert len(layers)==48 and all(list(pool[k])==layers for k in keys)
    original_count=sum(t.numel() for p in pool.values() for pair in p.values() for t in pair.values())
    assert original_count==1622016
    samples.append(snapshot('original_pool_loaded'))
    results={}
    for rank in [8,24]:
        compressed={};profiles=[];differences={f'{a}_to_{b}':[] for i,a in enumerate(keys) for b in keys[i+1:]}
        tick=time.monotonic()
        for layer in layers:
            pairs=[pool[k][layer] for k in keys];a,bs,energy=project(pairs,rank)
            compressed[layer]=dict(A_shared=a,B_by_state=dict(zip(keys,bs)))
            for k,p,b in zip(keys,pairs,bs):
                e=relative_error(p,a,b);profiles.append(dict(layer=layer,state=k,**e))
                if rank==24:assert e['relative_error']<5e-6
            for i,k in enumerate(keys):
                for j in range(i+1,len(keys)):
                    other=keys[j];differences[f'{k}_to_{other}'].append(
                        dict(layer=layer,**difference_error(pairs[i],pairs[j],a,bs[i],bs[j])))
        payload=sum(p['A_shared'].numel()+sum(b.numel() for b in p['B_by_state'].values()) for p in compressed.values())
        filename=FAMILY/f'candidate_rank{rank}.pt'
        torch.save(dict(shared=compressed,states=keys,scale=4.,rank=rank,
            source_pool_sha256=sha(run/'final_state.pt'),no_QA_or_TEST_used=True),filename)
        per_state={}
        for key in keys:
            group=[p for p in profiles if p['state']==key]
            per_state[str(key)]=math.sqrt(sum(p['deltaW_error_norm']**2 for p in group)/sum(p['original_deltaW_norm']**2 for p in group))
        diff_summary={name:math.sqrt(sum(p['error_norm']**2 for p in group)/sum(p['original_deltaW_difference_norm']**2 for p in group))
                      for name,group in differences.items()}
        results[str(rank)]=dict(rank=rank,payload_parameters=payload,FP32_payload_bytes=4*payload,
            serialized_file_bytes=filename.stat().st_size,checkpoint_sha256=sha(filename),
            seconds=time.monotonic()-tick,function_relative_errors=per_state,
            state_difference_relative_errors=diff_summary,per_layer_functions=profiles,per_layer_state_differences=differences,
            proposed_deployment_candidate=rank==8,numerical_control_only=rank==24)
        samples.append(snapshot('after_rank'+str(rank)+'_probe'))
        del compressed
    result=dict(CPU_only=True,no_GPU_or_backbone_instance=True,adapter_weights_loaded=True,no_QA_or_TEST_parsed=True,
        bound_TEST_file_bytes_hashed_only=True,mechanism_checks=mechanism,
        analyzer_sha256=sha(Path(__file__)),source_complete_sha256=sha(proof_path),
        source_pool_sha256=sha(run/'final_state.pt'),original_parameters=original_count,
        original_FP32_payload_bytes=4*original_count,ranks=results,CPU_probe_snapshots=samples,
        limits=['Tensor payload is not process RSS or a measured deployed memory advantage.',
                'Probe retains original pool and temporary decompositions; snapshots are CPU probe scope, not TRAIN peaks.',
                'Small total deltaW error can conceal larger errors in differences; both reported.',
                'Weight reconstruction, even full-union FP32 control, does not prove BF16 logits or Answer F1 equality.',
                'Only one rank8 candidate plus rank24 numerical control, no TEST-selected rank or GPU queue change.',
                'Shared-subspace compression is established linear algebra, no novelty or paper reproduction claimed.'])
    out.write_text(json.dumps(result,indent=2)+'\n')
    c=results['8'];mi=2**20
    text=['# 共用基底的权重压缩可行性（CPU，不是F1试验）','',
        '**探究的问题：** 连续适配的多个功能状态能否少存重复基底，同时保留状态间的新方向？','',
        '**做的实验：** 只读封存3个状态的实际BA，用B加权的联合右子空间SVD；固定rank8候选及rank24完整并集数值控制。未读QA／TEST、不训练模型、不改GPU队列。','',
        f"**结果：** 原FP32张量载荷{4*original_count/mi:.4f}MiB，rank8共用A＋3个B为{c['FP32_payload_bytes']/mi:.4f}MiB；不是已部署RSS节省。各状态实际函数重构相对误差{c['function_relative_errors']}；状态差分误差{c['state_difference_relative_errors']}。原始值／逐层／CPU快照见JSON。",'',
        '**证明了什么：** 只说明此固定权重的共享表示及数值误差；不能用整体能量或权重误差推断新知识／BF16输出／F1保留。完整并集为数值正控制，不能拿较大控制载荷宣传轻量。','',
        '**改进的方法：** 若后续实际功能评测支持，可用共享基底而非多份完整A降低重复存储。当前候选仅权重原型，不替换已准入F1匹配，亦不把已有子空间压缩思想当创新。','',
        '**改进的效果：** 尚无GPU/F1或实际部署内存收益；当前两新匹配条件和所有既定对照照常运行。','',
        '[数值、差分误差与SHA](EVIDENCE.json)']
    (REPORT/'RESEARCH_FINDINGS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(original_MiB=4*original_count/mi,candidate_MiB=c['FP32_payload_bytes']/mi,
        function_errors=c['function_relative_errors'],difference_errors=c['state_difference_relative_errors'],GPU_or_F1_result=False)))


if __name__=='__main__':main()
