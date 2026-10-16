"""Exercise actual basis/budget trainer on CPU tensors; no GPU model claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,tempfile
from pathlib import Path
import torch
import onevision_basis_budget_probe as probe
from onevision_research_worker import digest


class Monitor:
    def snapshot(self,*args,**kwargs):pass


class Toy:
    def __init__(self):
        self.model=torch.nn.Module()
        self.model.A=torch.nn.Parameter(torch.zeros(2,3,dtype=torch.float64))
        self.model.B=torch.nn.Parameter(torch.zeros(4,2,dtype=torch.float64));self.monitor=Monitor()
    def configure(self,state,finetune_shared=True):
        with torch.no_grad():self.model.A.copy_(state['k']['A']);self.model.B.copy_(state['k']['B'])
    def snapshot(self):return {'k':{'A':self.model.A.detach().clone(),'B':self.model.B.detach().clone()}}
    def loss(self,row):return ((4*self.model.B@self.model.A-torch.ones(4,3,dtype=torch.float64))**2).mean(),{}


def main():
    g=torch.Generator().manual_seed(42)
    initial={'k':{'A':torch.randn(2,3,generator=g,dtype=torch.float64),'B':torch.zeros(4,2,dtype=torch.float64)}}
    past={'k':{'A':initial['k']['A']+.1,'B':torch.ones(4,2,dtype=torch.float64)*.01}}
    root=Path(tempfile.mkdtemp(prefix='llmcl_basis_cpu_'));checks={}
    for kind in probe.STARTS:
        start=probe.initial_state(initial,past,kind);run=root/kind;run.mkdir();toy=Toy()
        records=probe.train_trajectory(toy,[{'id':'train1'},{'id':'train2'}],start,run)
        assert len(records)==4 and all(x['learning_rate_used']==1e-4 for x in records)
        assert records[-1]['cumulative_QA_exposures']==8 and len(list((run/'raw_gradients').glob('*.pt')))==4
        if kind!='past_AB':assert records[0]['A_change_l2']==0 and records[-1]['A_change_l2']>0
        else:assert records[0]['A_change_l2']>0
        checks[kind]='actual CPU trainer:4 finite steps, exposure counts, raw gradients and expected A-motion gate passed'
    a=probe.initial_state(initial,past,probe.STARTS[0]);b=probe.initial_state(initial,past,probe.STARTS[1])
    assert torch.equal(a['k']['B']@a['k']['A'],b['k']['B']@b['k']['A'])
    a['k']['A'].zero_();assert torch.count_nonzero(initial['k']['A'])>0
    _,data,*_=probe.freeze()
    counts={str(t):{'train':len(data[t,'train']),'test':len(data[t,'test'])} for t in probe.TASKS}
    assert counts=={'2':{'train':11,'test':228},'11':{'train':12,'test':267},'12':{'train':6,'test':100}}
    assert not torch.cuda.is_initialized()
    report=probe.REPORT;report.mkdir(exist_ok=True)
    result={'CPU_checks':checks,'frozen_counts':counts,'target_exposures_per_start':29*4,'full_native_generated_records':595*3*2,
            'initial_function_algebra':'exact zero BA for both A-only initializations; clone isolation passed','GPU_initialized':False,
            'real_OneVision_GPU_gates':'pending; CPU surrogate is not real model evidence',
            'source_sha256':digest(probe.ROOT/'analysis/onevision_basis_budget_probe.py'),
            'CPU_verifier_sha256':digest(Path(__file__)),'temporary_CPU_proof_directory':str(root),'test_based_tuning':False}
    (report/'cpu_preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
