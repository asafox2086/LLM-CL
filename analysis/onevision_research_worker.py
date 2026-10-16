"""GPU1 continuation with durable stage states, full fixed-probe gradient samples.

No modification of the immutable original diagnostic runner or its completed
controls. Interrupted sequential stage1 is restarted from the fixed base.
"""
import fcntl,gc,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'exp/onevision_diagnostics_20261016'
REPORT=ROOT/'summary/onevision_diagnostics_20261016'
sys.path.insert(0,str(HERE))
from run import Experiment,freeze,write

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def atomic_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.partial')
    torch.save(value,temporary)
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    temporary.replace(path)
    meta={'file':str(path.name),'bytes':path.stat().st_size,'sha256':digest(path)}
    write(path.with_suffix('.json'),meta)
    return meta

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['cuda'])

class ResearchExperiment(Experiment):
    def gradients(self,label,probes):
        self.model.eval();named=[(key,p) for key,p in self.model.named_parameters() if p.requires_grad]
        offsets={};offset=0
        for key,p in named:offsets[key]={'offset':offset,'count':p.numel(),'shape':list(p.shape)};offset+=p.numel()
        means={};stats={};paths={}
        for domain,rows in probes.items():
            chosen=rows[:4];samples=[];losses=[]
            for row in chosen:
                self.model.zero_grad(set_to_none=True);loss,_=self.loss(row);loss.backward()
                vector=torch.cat([p.grad.detach().float().cpu().flatten() if p.grad is not None else torch.zeros(p.numel()) for _,p in named])
                assert torch.isfinite(vector).all();samples.append(vector);losses.append(float(loss.detach()))
            tensor=torch.stack(samples);mean=tensor.double().mean(0);variance=tensor.double().var(0,unbiased=len(chosen)>1)
            means[domain]=mean;layers={}
            for key,info in offsets.items():
                start,end=info['offset'],info['offset']+info['count'];block=tensor[:,start:end].double()
                layers[key]={'mean_gradient_norm':float(mean[start:end].norm()),'per_example_gradient_norm':block.norm(dim=1).tolist(),
                             'sum_coordinate_variance':float(variance[start:end].sum())}
            path=self.run/'raw_probe_gradients'/label/(domain+'.pt')
            paths[domain]=atomic_torch(path,{'ids':[x['id'] for x in chosen],'train_only':True,'eval_mode_no_dropout':True,
                'parameter_layout':offsets,'per_example_preclip_gradient':tensor,'mean_gradient':mean.float(),
                'coordinate_variance':variance.float(),'nll':losses})
            stats[domain]={'count':len(chosen),'probe_nll':sum(losses)/len(losses),'layers':layers}
        self.model.zero_grad(set_to_none=True)
        cosine={d:{e:(float(torch.dot(a,b)/(a.norm()*b.norm())) if float(a.norm()*b.norm()) else None) for e,b in means.items()} for d,a in means.items()}
        write(self.run/'gradients'/f'{label}.json',{'train_only':True,'probes_per_domain':4,'raw_sample_files':paths,
              'same_model_parameterization_gradient_cosine':cosine,'domains':stats,'interpretation':'fixed probe gradients, not historical optimizer gradients'})

def source_hashes():
    files=list(HERE.glob('*.py'))+[Path(__file__)]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}

def rollback_incomplete(run,last_durable):
    events=run/'events.jsonl'
    if events.exists():
        keep=[];drop=[]
        for line in events.read_text().splitlines():
            try:
                record=json.loads(line);stage=int(record['label'].split('T')[-1])
            except (ValueError,KeyError,json.JSONDecodeError):drop.append(line);continue
            (keep if stage<=last_durable else drop).append(line)
        if drop:
            folder=run/'interrupted';folder.mkdir(exist_ok=True)
            (folder/f'events_{time.time_ns()}.jsonl').write_text('\n'.join(drop)+'\n')
            events.write_text('\n'.join(keep)+('\n' if keep else ''))
    for path in (run/'adapters').glob('sequential_T*.pt'):
        if int(path.stem.split('T')[-1])>last_durable:
            folder=run/'interrupted';folder.mkdir(exist_ok=True);path.rename(folder/(str(time.time_ns())+'_'+path.name))

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    lock=(REPORT/'research_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # A launch file alone never proves GPU ownership. Admit only on a genuinely idle card.
    while True:
        used,util=map(int,subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        owners=subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if used<500 and util<10 and not owners:break
        time.sleep(15)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(6)
    manifest,data,domains,tests,generation,probes,protocol=freeze()
    hashes=source_hashes();pointer=REPORT/'research_current_attempt.json'
    if pointer.exists():
        info=json.loads(pointer.read_text());assert info['source_sha256']==hashes,'Do not resume changed source; create a new attempt explicitly'
        run=ROOT/info['run']
    else:
        run=HERE/'runs'/('sequential_recovery_'+time.strftime('%Y%m%d_%H%M%S'));run.mkdir(parents=True)
        old=HERE/'runs/20261016_164638'
        controls={str(p.relative_to(ROOT)):digest(p) for p in (old/'adapters').glob('*.pt')}
        assert len(controls)==24
        info={'run':str(run.relative_to(ROOT)),'source_sha256':hashes,'old_control_run':str(old.relative_to(ROOT)),
              'completed_control_sha256':controls,'rng_rule':'seed42 at each stage; matched native controls will be re-run with this rule',
              'interrupted_old_sequential_stage1':'discarded, no stage checkpoint; no partial training or predictions reused',
              'formal_full_test':False,'protocol':protocol}
        write(run/'recovery_protocol.json',info);write(pointer,info)
    statefile=run/'progress.json';state=json.loads(statefile.read_text()) if statefile.exists() else {'trained_stage':0,'evaluated_stage':0}
    x=ResearchExperiment(manifest,run)
    if not (run/'initial_adapter.pt').exists():atomic_torch(run/'initial_adapter.pt',{'adapter':x.initial,'rng':capture_rng(),'source_sha256':hashes})
    last=state['trained_stage'];rollback_incomplete(run,last)
    if last:
        path=run/'durable_states'/f'stage_{last:02d}.pt';meta=json.loads(path.with_suffix('.json').read_text());assert digest(path)==meta['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['source_sha256']==hashes
        x.restore(saved['adapter']);restore_rng(saved['rng'])
    initial_check=run/'initial_reference_check.json'
    if not initial_check.exists():
        x.model.eval()
        with torch.no_grad():loss,details=x.loss(probes['natural'][0])
        assert abs(float(loss)-6.940610885620117)<.015
        write(initial_check,{'base_loss':float(loss),'old_initial_preflight_loss':6.940610885620117,'matched_base':True,**details})
    for stage in range(state['evaluated_stage']+1,20):
        write(REPORT/'status.json',{'phase':'sequential_recovery','native_stage':stage,'run':str(run.relative_to(ROOT)),
              'pid':os.getpid(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'durable_stage':state['trained_stage']})
        if stage>state['trained_stage']:
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            adapter=x.train(f'sequential_T{stage:02d}',data[stage,'train'])
            atomic_torch(run/'durable_states'/f'stage_{stage:02d}.pt',{'adapter':adapter,'rng':capture_rng(),'stage':stage,'source_sha256':hashes})
            state['trained_stage']=stage;write(statefile,state)
        x.gradients(f'sequential_T{stage:02d}',probes)
        x.evaluate(f'sequential_T{stage:02d}',tests,generation)
        state['evaluated_stage']=stage;write(statefile,state)
    x.monitor.snapshot('training_after',model=x.model);x.monitor.finish(x.model)
    write(REPORT/'status.json',{'phase':'sequential_recovery_complete','native_stage':19,'run':str(run.relative_to(ROOT)),
          'next_required':'rerun matched native controls with stage RNG42; task/domain and knowledge interventions; full-test verification still pending'})

if __name__=='__main__':main()
