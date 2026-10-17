"""Inference-only mechanism check after existing full confirmation, GPU1 only.

No new method/training or test-based choices. Evaluate fixed snapshots on the
same six TRAIN and100 original TEST QA, plus explicitly labeled visual controls.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,fcntl,json,subprocess,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from onevision_knowledge_intervention import Intervention,reset_seed
from onevision_research_worker import digest
from onevision_trained_visual_probe import repair_tail
from onevision_capacity_probe import finish_resources
from run import freeze,hashed,write
REPORT=ROOT/'summary/onevision_diagnostics_20261016/rural_collapse'


def plan():
    manifest,data,_,_,_,_,protocol=freeze()
    proof=json.loads((ROOT/'summary/onevision_diagnostics_20261016/basis_budget/complete.json').read_text())
    root=ROOT/proof['run'];basis_spec=json.loads((root/'protocol.json').read_text())
    sources={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))}
    for path in ['analysis/onevision_knowledge_intervention.py','exp/onevision_diagnostics_20261016/run.py']:
        sources[path]=digest(ROOT/path)
    checkpoints={'past_before':ROOT/'exp/onevision_diagnostics_20261016/runs/20261016_164638/adapters/isolated_natural.pt'}
    assert digest(checkpoints['past_before'])==basis_spec['past_adapter_sha256']
    for start in ['random_A_zero_B','past_AB']:
        case=root/f'T12_{start}';completion=json.loads((case/'complete.json').read_text())
        for update in [1,4]:
            p=case/f'update_{update}.pt';assert digest(p)==completion['artifact_sha256'][p.name]
            checkpoints[f'{start}_update{update}']=p
    groups={'TRAIN':data[12,'train'],'TEST':data[12,'test']}
    assert [len(groups[k]) for k in ['TRAIN','TEST']]==[6,100]
    labels=sorted({r['answer'] for r in groups['TRAIN']});assert labels==['rural','urban']
    wrong={};pixels={}
    for split,rows in groups.items():
        ordered=sorted(rows,key=lambda r:hashed(r,'rural-wrong-image'))
        for i,row in enumerate(ordered):
            other=next(ordered[(i+offset)%len(rows)] for offset in range(1,len(rows)) if ordered[(i+offset)%len(rows)]['images']!=row['images'])
            for q in [row,other]:
                key=tuple(q['images'])
                if key not in pixels:pixels[key]=tuple(digest(ROOT/path) for path in key)
            assert pixels[tuple(row['images'])]!=pixels[tuple(other['images'])]
            wrong[split,row['id']]=other
    spec={'source_sha256':sources,'manifest_sha256':protocol['source_manifest_sha256'],
          'checkpoints':{name:{'path':str(path.relative_to(ROOT)),'sha256':digest(path)} for name,path in checkpoints.items()},
          'ids':{split:[r['id'] for r in rows] for split,rows in groups.items()},
          'wrong_image_ids':{split:{row['id']:wrong[split,row['id']]['id'] for row in rows} for split,rows in groups.items()},
          'views':['image','wrong','text'],'candidate_strings':labels+['The answer is '+label for label in labels],
          'candidates_from':'original TRAIN labels plus fixed literal English wrapper; no test-gold selection',
          'candidate_NLL':'auxiliary counterfactual likelihoods, sum and mean including EOS; not a new benchmark score',
          'optimizer_updates':0,'selection':'post-hoc collapse mechanism follow-up, all endpoints1/4 of random and complete pastAB; no test-based tuning',
          'scope':'106 original QA x5 checkpoints x3 views; not full21454 or formal CL; wrong/text controls are inference-only'}
    return manifest,groups,wrong,checkpoints,spec


def worker():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    manifest,groups,wrong,checkpoints,spec=plan()
    pointer=REPORT/'attempt.json'
    if pointer.exists():
        attempt=json.loads(pointer.read_text());assert attempt['spec']==spec;run=ROOT/attempt['run']
    else:
        run=ROOT/'exp/onevision_rural_collapse_20261017'/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
        write(pointer,{'run':str(run.relative_to(ROOT)),'spec':spec});write(run/'protocol.json',spec)
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    started=time.time()
    x=Intervention(manifest,run);x.process_started=started
    x.boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    x.model.eval();x.monitor.snapshot('inference_before',model=x.model)
    for name,path in checkpoints.items():
        state=torch.load(path,map_location='cpu',weights_only=True)['adapter']
        x.configure(state,finetune_shared=False);x.model.eval()
        assert not any(p.requires_grad for p in x.model.parameters())
        for split,rows in groups.items():
            for view in spec['views']:
                target=run/f'{name}_{split}_{view}.jsonl';repair_tail(target)
                existing=[json.loads(line) for line in target.read_text().splitlines()] if target.exists() else []
                assert len(existing)<=len(rows) and [r['id'] for r in existing]==[r['id'] for r in rows[:len(existing)]]
                for row in rows[len(existing):]:
                    bad=wrong[split,row['id']] if view=='wrong' else None
                    with torch.no_grad():
                        loss,details=x.loss(row,view,bad)
                        assert torch.isfinite(loss)
                        generated=x.generate(row,view,bad);candidates={}
                        for answer in spec['candidate_strings']:
                            synthetic=dict(row,answer=answer)
                            value,d=x.loss(synthetic,view,bad);assert torch.isfinite(value)
                            candidates[answer]={'mean_nll_with_EOS':float(value),'sum_nll_with_EOS':float(value)*d['target_tokens'],'tokens_with_EOS':d['target_tokens']}
                    result={'id':row['id'],'split':split,'view':view,'checkpoint':name,'gold_nll':float(loss),
                            'references':row['answers'],'wrong_image_id':bad['id'] if bad else None,
                            'candidates':candidates,**details,**generated}
                    with target.open('a') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                    write(REPORT/'status.json',{'phase':'inference','pid':os.getpid(),'checkpoint':name,'split':split,'view':view,'last_id':row['id'],'run':str(run.relative_to(ROOT))})
        actual=x.snapshot();assert all(torch.equal(actual[k][f],state[k][f]) for k in state for f in ['A','B'])
    x.monitor.snapshot('inference_after',model=x.model)
    finish_resources(x,run,'inference_before','inference_after','inference_only_no_optimizer; TRAIN evaluation is not training')
    write(REPORT/'complete.json',{'complete':True,'run':str(run.relative_to(ROOT)),'spec':spec,
          'artifact_sha256':{str(p.relative_to(run)):digest(p) for p in [*run.glob('*.jsonl'),run/'memory_summary.json',run/'protocol.json']},'optimizer_updates':0,'formal_full_test':False})


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:
        _,_,_,_,spec=plan();write(REPORT/'CPU_PLAN_CHECK.json',{'passed':True,'GPU_executed':False,'spec':spec});print('CPU plan/QA/checkpoint/wrong-pixel checks passed');return
    lock=(REPORT/('worker.lock' if args.worker else 'coordinator.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (REPORT/'needs_attention.json').exists():return
    if args.worker:worker();return
    while not (REPORT/'complete.json').exists():
        owner=int(subprocess.check_output(['systemctl','--user','show','llmcl-onevision-full-repair.service','-p','MainPID','--value'],text=True).strip() or 0)
        pids=subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        used,util=map(int,subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','))
        if owner or pids or used>=500 or util>=10:
            write(REPORT/'queue_state.json',{'phase':'waiting_full_confirmation_and_idle_GPU1','pid':os.getpid(),'full_service_pid':owner,'gpu_owner_pids':pids.splitlines()});time.sleep(15);continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6')
        with (REPORT/'worker_nohup.log').open('a') as log:
            p=subprocess.Popen(['/usr/bin/nohup',str(ROOT/'.vision-env/bin/python'),'-u',str(Path(__file__)),'--worker'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(REPORT/'queue_state.json',{'phase':'running','pid':os.getpid(),'worker_pid':p.pid});code=p.wait()
        assert code==0 and (REPORT/'complete.json').exists(),'Inference worker did not complete; inspect evidence'
    write(REPORT/'queue_state.json',{'phase':'completed','pid':os.getpid()})


if __name__=='__main__':
    try:main()
    except BlockingIOError:pass
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True);write(REPORT/'needs_attention.json',{'pid':os.getpid(),'traceback':traceback.format_exc()});raise
