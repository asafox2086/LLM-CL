"""Scheduled causal probe: inherited A vs inherited BA vs four TRAIN passes.

Separate from formal 1-epoch CL. No test-based decisions; wait existing queues.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,fcntl,json,math,subprocess,sys,time,traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from onevision_knowledge_intervention import Intervention,reset_seed,case_complete
from onevision_research_worker import atomic_torch,digest
from onevision_capacity_probe import finish_resources
from run import freeze,hashed,write
REPORT=ROOT/'summary/onevision_diagnostics_20261016/basis_budget'
TASKS=[2,11,12]
STARTS=['random_A_zero_B','past_A_zero_B','past_AB']


def initial_state(initial,past,kind):
    assert kind in STARTS
    return {k:{'A':(initial[k]['A'] if kind==STARTS[0] else past[k]['A']).clone(),
               'B':(past[k]['B'] if kind=='past_AB' else torch.zeros_like(initial[k]['B'])).clone()} for k in initial}


def weight_error(a,b):
    return max(float((a[k][f].double()-b[k][f].double()).abs().max()) for k in a for f in ['A','B'])


def wait_admission():
    services=['llmcl-onevision-research.service','llmcl-onevision-mechanisms.service',
              'llmcl-onevision-deterministic-native.service','llmcl-onevision-capacity.service']
    while True:
        owners={s:int(subprocess.check_output(['systemctl','--user','show',s,'-p','MainPID','--value'],text=True).strip() or 0) for s in services}
        if any(owners.values()):
            write(REPORT/'status.json',{'phase':'waiting_existing_queues','pid':os.getpid(),'service_PIDs':owners});time.sleep(30);continue
        gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        used,util=map(int,subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','))
        if not gpu and used<500 and util<10:return
        write(REPORT/'status.json',{'phase':'waiting_idle_gpu1','pid':os.getpid(),'GPU_PIDs':gpu.splitlines()});time.sleep(15)


def train_trajectory(x,rows,start,run):
    """One optimizer, four identical ordered passes, same constant LR prefix."""
    x.configure(start,finetune_shared=True);reset_seed();x.model.train()
    named=[(n,p) for n,p in x.model.named_parameters() if p.requires_grad]
    parameters=[p for _,p in named];optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0)
    assert 0<len(rows)<=16
    ordered=sorted(rows,key=lambda r:hashed(r,'training'));records=[]
    x.monitor.snapshot('target_training_before',model=x.model)
    for step in range(1,5):
        optimizer.zero_grad(set_to_none=True);losses=[]
        for row in ordered:
            loss,details=x.loss(row);assert torch.isfinite(loss);(loss/len(rows)).backward();losses.append(float(loss.detach()))
        vector=torch.cat([p.grad.detach().float().cpu().flatten() if p.grad is not None else torch.zeros(p.numel()) for _,p in named])
        grad=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(grad)
        optimizer.step();snapshot=x.snapshot()
        atomic_torch(run/'raw_gradients'/f'update_{step}.pt',{'gradient_preclip':vector,'layout':[(n,list(p.shape)) for n,p in named],
                      'TRAIN_ids':[r['id'] for r in ordered],'step':step})
        if step in [1,4]:atomic_torch(run/f'update_{step}.pt',{'adapter':snapshot,'updates':step})
        a=math.sqrt(sum(float((snapshot[k]['A'].double()-start[k]['A'].double()).square().sum()) for k in start))
        b=math.sqrt(sum(float((snapshot[k]['B'].double()-start[k]['B'].double()).square().sum()) for k in start))
        records.append({'update':step,'loss':sum(losses)/len(losses),'gradient_l2_preclip':float(grad),'learning_rate_used':1e-4,
                        'TRAIN_ids':[r['id'] for r in ordered],'cumulative_QA_exposures':step*len(rows),'A_change_l2':a,'B_change_l2':b})
        print(json.dumps({'event':'basis_budget_update',**records[-1]}),flush=True)
    optimizer.zero_grad(set_to_none=True);del optimizer
    x.monitor.snapshot('target_training_after',model=x.model)
    write(run/'training.json',{'constant_learning_rate':1e-4,'fresh_optimizer_at_trajectory_start':True,'records':records,
                              'one_update_equals_one_pass':True,'four_update_exposure_multiplier':4})
    return records


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    parser=argparse.ArgumentParser();parser.add_argument('--case-index',type=int);args=parser.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=(REPORT/('worker.lock' if args.case_index is None else f'case_{args.case_index}.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (REPORT/'needs_attention.json').exists(),'Previous failure needs inspection; do not retry blindly'
    wait_admission();torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    manifest,data,domains,tests,generation,probes,protocol=freeze()
    past_path=ROOT/'exp/onevision_diagnostics_20261016/runs/20261016_164638/adapters/isolated_natural.pt'
    past=torch.load(past_path,map_location='cpu',weights_only=True)['adapter']
    files=[Path(__file__),ROOT/'analysis/onevision_knowledge_intervention.py',ROOT/'analysis/onevision_research_worker.py',
           ROOT/'analysis/onevision_capacity_probe.py',ROOT/'exp/onevision_diagnostics_20261016/run.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    spec={'source_sha256':hashes,'manifest_sha256':protocol['source_manifest_sha256'],'past_adapter_sha256':digest(past_path),
          'past_adapter_selection':'first original natural domain, never test-best; prior347 TRAIN exposures and compute disclosed',
          'target_tasks':TASKS,'starts':STARTS,'update_counts':[1,4],'constant_lr':1e-4,'rank':8,'seed':42,
          'target_train_ids':{str(t):[r['id'] for r in data[t,'train']] for t in TASKS},
          'native_test_ids':{str(t):[r['id'] for r in data[t,'test']] for t in TASKS},
          'scope':'isolated native-task causal mechanism probe; NOT19-stage CL, formal one-epoch or full21454 benchmark',
          'prompt':'original unchanged, no knowledge context; same within every pair',
          'evaluation':'all original native test228/267/100;18 endpoint models; no routing/capacity/early stopping from tests',
          'resource_scope':'fresh GPU process per trajectory; training before/after/peaks exclude endpoint evaluation'}
    pointer=REPORT/'attempt.json'
    if pointer.exists():
        attempt=json.loads(pointer.read_text());assert attempt['spec']==spec;root=ROOT/attempt['run']
    else:
        root=ROOT/'exp/onevision_basis_budget_20261016'/time.strftime('%Y%m%d_%H%M%S');root.mkdir(parents=True)
        write(pointer,{'run':str(root.relative_to(ROOT)),'spec':spec});write(root/'protocol.json',spec)
    if args.case_index is None:
        for index in range(len(TASKS)*len(STARTS)):
            with (REPORT/f'case_{index}_nohup.log').open('a') as log:
                child=subprocess.Popen(['/usr/bin/nohup',sys.executable,'-u',str(Path(__file__)),'--case-index',str(index)],
                                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                write(REPORT/'status.json',{'phase':'waiting_live_worker','case_index':index,'pid':os.getpid(),'worker_pid':child.pid})
                code=child.wait()
            assert code==0 and not (REPORT/'needs_attention.json').exists(),f'Worker {index} failed, inspect log'
        write(REPORT/'complete.json',{'complete':True,'run':str(root.relative_to(ROOT)),'source_sha256':hashes,'full_five_domain_benchmark':False})
        write(REPORT/'status.json',{'phase':'completed','pid':os.getpid(),'run':str(root.relative_to(ROOT))});return
    assert 0<=args.case_index<len(TASKS)*len(STARTS)
    for task in TASKS:
        assert 0<len(data[task,'train'])<=16
        for kind in STARTS:
            if TASKS.index(task)*len(STARTS)+STARTS.index(kind)!=args.case_index:continue
            case=f'T{task:02d}_{kind}';directory=root/case;done=directory/'complete.json'
            if done.exists():
                proof=json.loads(done.read_text());assert proof['source_sha256']==hashes
                for rel,sha in proof['artifact_sha256'].items():assert digest(directory/rel)==sha
                continue
            trained=directory/'trained.json'
            if directory.exists() and not trained.exists():
                quarantine=root/'interrupted';quarantine.mkdir(exist_ok=True);directory.rename(quarantine/(case+'_'+str(time.time_ns())))
            directory.mkdir(exist_ok=True);write(REPORT/'status.json',{'phase':'running','case':case,'run':str(root.relative_to(ROOT)),'pid':os.getpid()})
            process_started=time.time();x=Intervention(manifest,directory);x.process_started=process_started;x.boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            x.prompt('original');x.knowledge=None;start=initial_state(x.original_initial,past,kind)
            if not trained.exists():
                # Same zero initial function, actual image forward checked without TEST answers.
                probe=data[task,'train'][0];losses={}
                for k in STARTS[:2]:
                    x.configure(initial_state(x.original_initial,past,k),finetune_shared=True);x.model.eval()
                    with torch.no_grad():loss,_=x.loss(probe)
                    losses[k]=float(loss)
                assert losses[STARTS[0]]==losses[STARTS[1]],'A-only initial function gate failed'
                records=train_trajectory(x,data[task,'train'],start,directory)
                if kind!='past_AB':assert records[0]['A_change_l2']==0 and records[0]['B_change_l2']>0
                finish_resources(x,directory,'target_training_before','target_training_after','one trajectory4 updates; endpoint1 shares exact prefix; no endpoint evaluation in training peaks')
                # resources.jsonl is append-only across inference/restart sessions;
                # immutable training memory_summary and weights are hash-locked.
                artifacts=[directory/'training.json',directory/'memory_summary.json',directory/'update_1.pt',directory/'update_4.pt',*sorted((directory/'raw_gradients').glob('*.pt'))]
                write(trained,{'source_sha256':hashes,'initial_function_gate':losses,'artifact_sha256':{str(p.relative_to(directory)):digest(p) for p in artifacts}})
            training_proof=json.loads(trained.read_text());assert training_proof['source_sha256']==hashes
            for rel,sha in training_proof['artifact_sha256'].items():assert digest(directory/rel)==sha
            selected={'native':data[task,'test']}
            for step in [1,4]:
                label=f'endpoint_{step}'
                if case_complete(directory,label,selected):continue
                checkpoint=torch.load(directory/f'update_{step}.pt',map_location='cpu',weights_only=True)
                x.configure(checkpoint['adapter'],finetune_shared=False);x.model.eval()
                x.evaluate(label,selected,selected)
            artifacts=[trained,*list((directory/'evaluations').rglob('*.jsonl')),*list((directory/'evaluations').rglob('durable_complete.json'))]
            x.monitor.done.set();x.monitor.thread.join(timeout=6)
            write(done,{'source_sha256':hashes,'artifact_sha256':{**training_proof['artifact_sha256'],**{str(p.relative_to(directory)):digest(p) for p in artifacts}}})
            del x
            import gc
            gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    try:main()
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        error={'traceback':traceback.format_exc(),'GPU_results_accepted':False,'pid':os.getpid()}
        write(REPORT/f'failure_{time.time_ns()}.json',error)
        if not (REPORT/'needs_attention.json').exists():write(REPORT/'needs_attention.json',error)
        print(error['traceback'],file=sys.stderr,flush=True)
        sys.exit(1)
