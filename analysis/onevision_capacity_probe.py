"""Controlled capped adapter-reuse probe, not a claimed published algorithm.

Each cap runs in its own process. A temporary candidate receives one original
epoch; TRAIN loss decides overwrite vs fork. Input prototypes use Q+image only.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,fcntl,hashlib,json,math,sys,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'exp/onevision_diagnostics_20261016'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'analysis'))
from run import Experiment,freeze,hashed,write
from onevision_knowledge_intervention import Intervention,reset_seed,case_complete,valid_checkpoint
from onevision_research_worker import atomic_torch,digest,ResearchExperiment
from onevision_deterministic_native import checked_events,quarantine_partial_events
from onevision_trained_visual_probe import repair_tail
from geometry import inner

REPORT=ROOT/'summary/onevision_diagnostics_20261016/capacity'
CAPS=[1,2,3,5]
CONFIRM_CAPS=[1,3]  # Declared before any capacity test score; resource-budget choice.


def router_inputs(row):
    return row['question'],tuple(row['images'])


def combine_feature(question,vision):
    q=F.normalize(question.float().mean(0),dim=0) if question.numel() else torch.zeros(vision.shape[-1])
    v=F.normalize(vision.float().mean(0),dim=0)
    return F.normalize(q.cpu()+v.cpu(),dim=0)


def route_prototype(feature,prototypes,assignments):
    if not prototypes:return None,None
    task=min(prototypes,key=lambda t:(-float(torch.dot(feature.double(),prototypes[t].double())),t))
    return assignments[task],task


def should_fork(size,cap,old_before,old_after,current_before,current_after):
    return size<cap and old_before is not None and old_after>old_before and current_after<current_before


def weight_cosine(a,b):
    aa=sum(inner(x,x) for x in a.values());bb=sum(inner(x,x) for x in b.values())
    return sum(inner(a[k],b[k]) for k in a)/math.sqrt(aa*bb) if aa*bb else None


class CapacityExperiment(Intervention):
    def __init__(self,manifest,run):
        self.process_started=time.time();self.boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        super().__init__(manifest,run);self.feature_cache={};self.feature_misses=0;self.set_route('fixed',{}, {}, {},'initial')

    def set_route(self,mode,pool,prototypes,assignments,revision):
        self.route_mode=mode;self.pool=pool;self.prototypes=prototypes;self.assignments=assignments
        self.revision=revision;self.active_key=None;self.last_route={}

    def feature(self,row):
        question,images=router_inputs(row)
        key=hashlib.sha256(json.dumps([question,images],ensure_ascii=False).encode()).hexdigest()
        if key not in self.feature_cache:
            self.feature_misses+=1
            self.model.eval()
            # A fixed generic template; actual task/domain/context/gold never enters routing.
            aux={'_task':1,'question':question,'images':list(images)}
            with torch.no_grad():
                batch=Experiment.inputs(self,aux)
                ids=self.processor.tokenizer(question.replace('<image>',''),add_special_tokens=False)['input_ids']
                text=self.model.get_input_embeddings()(torch.tensor([ids],device='cuda',dtype=torch.long))[0].detach().float().cpu()
                visual=batch['_features'].detach().float().cpu()
                self.feature_cache[key]=combine_feature(text,visual)
            assert torch.isfinite(self.feature_cache[key]).all()
        return self.feature_cache[key]

    def activate_route(self,row):
        if self.route_mode=='fixed':
            self.last_route={'route_mode':'fixed','route_expert':self.revision};return
        if self.route_mode=='input':
            expert,task=route_prototype(self.feature(row),self.prototypes,self.assignments)
        else:
            assert self.route_mode=='known_task_reference'
            task=row['_task'];expert=self.assignments[task]
        key=(self.revision,expert)
        if key!=self.active_key:
            self.restore(self.original_initial if expert is None else self.pool[expert]);self.active_key=key
        self.last_route={'route_mode':self.route_mode,'route_expert':expert,'route_prototype_task':task}

    def loss(self,row,view='image',wrong=None):
        self.activate_route(row);loss,details=super().loss(row,view,wrong)
        return loss,{**details,**self.last_route}

    def generate(self,row,view='image',wrong=None):
        self.activate_route(row);return super().generate(row,view,wrong)


def source_hashes():
    files=[Path(__file__),HERE/'run.py',HERE/'geometry.py',
           ROOT/'analysis/onevision_knowledge_intervention.py',ROOT/'analysis/onevision_research_worker.py',
           ROOT/'analysis/onevision_deterministic_native.py',ROOT/'analysis/onevision_trained_visual_probe.py',
           ROOT/'analysis/onevision_capacity_queue.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def finish_resources(x,run,before_phase,after_phase,scope):
    x.monitor.snapshot('training_after',model=x.model);x.monitor.finish(x.model)
    rows=[json.loads(line) for line in (run/'resources.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['pid']==os.getpid() and r['unix']>=x.process_started]
    before=next(r for r in rows if r['phase']==before_phase)
    after=next(r for r in rows if r['phase']==after_phase)
    keys=['VmRSS_bytes','Pss_bytes','cpu_uss_bytes','gpu_allocated_bytes','gpu_reserved_bytes','gpu_process_nvml_bytes']
    result={'unit':'bytes','pid':os.getpid(),'boot_id':x.boot_id,'process_started_unix':x.process_started,
            'before':before,'after':after,'scope':scope,
            'growth_bytes':{k:after[k]-before[k] if after.get(k) is not None and before.get(k) is not None else None for k in keys},
            'observed_peak_bytes':{k:max(r[k] for r in rows if r.get(k) is not None) if any(r.get(k) is not None for r in rows) else None for k in keys},
            'note':'Same-PID before/after; fresh process per cap. On recovery these describe resumed process only. Stage diagnostics are included in observed peaks; allocator cache is separate from parameter growth.'}
    all_rows=[json.loads(line) for line in (run/'resources.jsonl').read_text().splitlines()]
    result['all_process_observed_peak_bytes']={k:max(r[k] for r in all_rows if r.get(k) is not None) if any(r.get(k) is not None for r in all_rows) else None for k in keys}
    write(run/'memory_summary.json',result)


def stratify_full_results(run,full,diagnostic_ids):
    groups={'all':[],'in_diagnostic500':[],'outside_diagnostic500':[]}
    for domain,expected in full.items():
        path=run/'evaluations/final_full_input_router'/f'{domain}_image.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert [r['id'] for r in rows]==[r['id'] for r in expected]
        for row in rows:
            assert 'scores' in row
            groups['all'].append(row)
            groups['in_diagnostic500' if row['id'] in diagnostic_ids else 'outside_diagnostic500'].append(row)
    assert [len(groups[k]) for k in ['all','in_diagnostic500','outside_diagnostic500']]==[21454,500,20954]
    result={}
    for name,rows in groups.items():
        result[name]={'qa':len(rows),'normalized_EM_correct':sum(r['scores']['exact_match']==100 for r in rows),
                     'qa_weighted':{m:sum(r['scores'][m] for r in rows)/len(rows) for m in ['exact_match','token_f1','rougeL']}}
    write(run/'full_test_stratification.json',{'groups':result,
          'note':'All original public test QA retained. The500 diagnostic QA were examined during discovery; full test is not an untouched confirmation set. Report the20954 outside that diagnostic subset separately. No full-test FWT/BWT inferred from final-only evaluation.'})


def main(cap,confirm=False):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1' and cap in CAPS
    if confirm:assert cap in CONFIRM_CAPS
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=(REPORT/f'cap{cap}_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    manifest,data,domains,tests,_,probes,protocol=freeze();hashes=source_hashes()
    reference_file=ROOT/'summary/onevision_diagnostics_20261016/deterministic_native/complete.json'
    reference=json.loads(reference_file.read_text());native=ROOT/reference['run']
    assert json.loads((native/'full_T01_reproduction_gate.json').read_text())['exact_equal']
    assert digest(native/'native_history_comparison.json')==reference['comparison_sha256']
    native_spec=json.loads((native/'protocol.json').read_text())
    gids=native_spec['generation_ids'];generation={d:[r for r in rows if r['id'] in set(gids[d])] for d,rows in tests.items()}
    train={t['task']:data[t['task'],'train'] for t in manifest['tasks']}
    current_probes={t:sorted(rows,key=lambda r:hashed(r,'capacity-current-probe'))[:16] for t,rows in train.items()}
    memories={t:sorted(rows,key=lambda r:hashed(r,'capacity-old-memory'))[:4] for t,rows in train.items()}
    name=f'cap{cap}'+('_full_confirm' if confirm else '')
    spec={'source_sha256':hashes,'cap':cap,'confirm_full':confirm,'seed':42,'epochs_per_task':1,
          'manifest_sha256':protocol['source_manifest_sha256'],'model_revision':protocol['model_revision'],
          'backend':'deterministic default SDPA; CUBLAS_WORKSPACE_CONFIG=:4096:8',
          'native_reference_completion_sha256':digest(reference_file),'full_confirm_caps':CONFIRM_CAPS,
          'current_probe_ids':{str(t):[r['id'] for r in rows] for t,rows in current_probes.items()},
          'old_probe_memory_ids':{str(t):[r['id'] for r in rows] for t,rows in memories.items()},
          'allocation':'fork iff old TRAIN task-macro NLL increases and current TRAIN NLL decreases and retained pool below cap; else overwrite',
          'router':'nearest frozen Q+image native TRAIN prototype; no gold/task/domain/context in feature',
          'budget':'one candidate epoch/current task; extra TRAIN scoring and probe memory explicitly recorded',
          'full_confirmation_scope':'full21454 retained; separately report previously examined diagnostic500 and other20954, not an untouched-test claim',
          'cap_definition':'retained deployable LoRA experts, excludes temporary training candidate/backup and saved historical checkpoints'}
    pointer=REPORT/(name+'_attempt.json');complete=REPORT/(name+'_complete.json')
    if pointer.exists():
        info=json.loads(pointer.read_text());assert info['spec']==spec;run=ROOT/info['run']
    else:
        run=ROOT/'exp/onevision_capacity_20261016'/name/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
        write(pointer,{'run':str(run.relative_to(ROOT)),'spec':spec});write(run/'protocol.json',spec)
    if complete.exists():
        proof=json.loads(complete.read_text());assert proof['source_sha256']==hashes
        for relative,expected in proof['artifact_sha256'].items():assert digest(run/relative)==expected
        return
    if (REPORT/(name+'_needs_attention.json')).exists():return
    repair_tail(run/'resources.jsonl')
    x=CapacityExperiment(manifest,run);initial=x.original_initial
    reference_initial=torch.load(native/'initial_adapter.pt',map_location='cpu',weights_only=True)['adapter']
    assert all(torch.equal(initial[k][b],reference_initial[k][b]) for k in initial for b in ['A','B'])

    def evaluate(label,selected,generated,mode,pool,proto,mapping,fixed=None):
        x.set_route(mode,pool,proto,mapping,label)
        if fixed is not None:x.restore(fixed)
        if not case_complete(run,label,selected):x.evaluate(label,selected,generated)

    # Actual GPU feature invariant: changing gold/task/context cannot change router feature.
    gate=run/'router_input_gate.json'
    if not gate.exists():
        original=train[1][0];a=x.feature(original).clone()
        mutated=dict(original,_task=999,answer='not a target',answers=['not a target'],context='ignored metadata')
        x.feature_cache.clear();x.cache.clear();b=x.feature(mutated)
        assert torch.equal(a,b)
        write(gate,{'exact_feature_equal_after_gold_task_context_change':True,'train_id':original['id'],
                    'actual_dataset_unchanged':True,'optimizer_updates':0})

    if confirm:
        trained=json.loads((REPORT/f'cap{cap}_complete.json').read_text());trained_run=ROOT/trained['run']
        path=trained_run/'latest.pt';assert digest(path)==trained['final_state_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True);assert state['stage']==19 and state['cap']==cap and state['source_sha256']==hashes
        full={d:[row for t in tasks for row in data[t,'test']] for d,tasks in domains.items()}
        assert sum(map(len,full.values()))==21454
        x.monitor.snapshot('full_inference_before',model=x.model)
        evaluate('final_full_input_router',full,full,'input',state['pool'],state['prototypes'],state['assignments'])
        stratify_full_results(run,full,{row['id'] for group in tests.values() for row in group})
        x.monitor.snapshot('full_inference_after',model=x.model)
        finish_resources(x,run,'full_inference_before','full_inference_after','inference_only_full_confirmation; zero optimizer updates')
        write(complete,{'complete':True,'run':str(run.relative_to(ROOT)),'source_sha256':hashes,
              'full_test_qa':21454,'formal_full_test':True,'trained_state_sha256':trained['final_state_sha256'],
              'artifact_sha256':{str(p.relative_to(run)):digest(p) for p in list((run/'evaluations').rglob('*.jsonl'))+[run/'memory_summary.json',run/'resources.jsonl',run/'full_test_stratification.json']}})
        return

    latest=run/'latest.pt'
    if valid_checkpoint(latest):
        state=torch.load(latest,map_location='cpu',weights_only=True);assert state['source_sha256']==hashes
        assert state['cap']==cap
    else:state={'stage':0,'cap':cap,'pool':{},'prototypes':{},'assignments':{}}
    assert len(state['pool'])<=cap
    assert set(state['prototypes'])==set(state['assignments'])==set(range(1,state['stage']+1))
    assert all(e in state['pool'] for e in state['assignments'].values())
    resume_stage=state['stage'];x.monitor.snapshot('capacity_training_before',model=x.model)
    for tid in range(state['stage']+1,20):
        pool,proto,mapping=state['pool'],state['prototypes'],state['assignments']
        domain=manifest['tasks'][tid-1]['domain'];rows=train[tid];label=f'cap{cap}_T{tid:02d}'
        selected_test={domain:[r for r in tests[domain] if r['_task']==tid]}
        selected_gen={domain:[r for r in generation[domain] if r['_task']==tid]};assert selected_gen[domain]
        write(REPORT/'status.json',{'phase':'capacity_probe','cap':cap,'task':tid,'pid':os.getpid(),'run':str(run.relative_to(ROOT))})
        aux=[]

        def nll(adapter,qa,role):
            tick=time.monotonic()
            x.set_route('fixed',pool,proto,mapping,role);x.restore(adapter);x.model.eval();values=[]
            for row in qa:
                with torch.no_grad():loss,_=x.loss(row)
                assert torch.isfinite(loss);values.append(float(loss))
            aux.append({'role':role,'train_ids':[r['id'] for r in qa],'nll':values,'seconds':time.monotonic()-tick})
            return sum(values)/len(values)

        scores={e:nll(a,current_probes[tid],'select_'+e) for e,a in pool.items()}
        chosen=min(scores,key=lambda e:(scores[e],e)) if scores else None
        start=pool[chosen] if chosen is not None else initial
        before_current=scores[chosen] if chosen is not None else nll(start,current_probes[tid],'base_current')
        protected=[t for t in sorted(mapping) if mapping[t]==chosen]
        before_old=[nll(start,memories[t],f'old_before_T{t}') for t in protected]
        evaluate(label+'_pre_router',selected_test,selected_gen,'input',pool,proto,mapping)
        evaluate(label+'_pre_forced',selected_test,{domain:[]},'fixed',pool,proto,mapping,start)
        candidate_file=run/'candidates'/(label+'.pt')
        x.configure(start,finetune_shared=True);x.set_route('fixed',pool,proto,mapping,'training_candidate')
        if valid_checkpoint(candidate_file):
            saved=torch.load(candidate_file,map_location='cpu',weights_only=True);assert saved['source_sha256']==hashes
            assert checked_events(run,label,len(rows))==saved['events_sha256'];candidate=saved['adapter'];x.restore(candidate)
        else:
            quarantine_partial_events(run,label);x.cache.clear();reset_seed();candidate=x.train(label,rows)
            event_hash=checked_events(run,label,len(rows))
            with (run/'events.jsonl').open('rb') as stream:os.fsync(stream.fileno())
            atomic_torch(candidate_file,{'adapter':candidate,'source_sha256':hashes,'events_sha256':event_hash})
        if cap==1 or tid==1:
            reference_path=native/'adapters'/f'sequential_T{tid:02d}.pt'
            assert digest(reference_path)==reference['artifact_sha256'][str(reference_path.relative_to(native))]
            reference_state=torch.load(reference_path,map_location='cpu',weights_only=True)['adapter']
            exact=all(torch.equal(candidate[k][b],reference_state[k][b]) for k in candidate for b in ['A','B'])
            write(run/'gates'/(label+'.json'),{'exact_matches_deterministic_seqft':exact,'native_task':tid})
            assert exact,'Cap1 or first task failed exact deterministic SeqFT equivalence'
        after_current=nll(candidate,current_probes[tid],'current_after')
        after_old=[nll(candidate,memories[t],f'old_after_T{t}') for t in protected]
        old_before=sum(before_old)/len(before_old) if before_old else None
        old_after=sum(after_old)/len(after_old) if after_old else None
        fork=chosen is not None and should_fork(len(pool),cap,old_before,old_after,before_current,after_current)
        commit='e'+str(len(pool)) if chosen is None or fork else chosen
        cosine=weight_cosine(start,candidate)
        pool=dict(pool);pool[commit]=candidate;assert len(pool)<=cap
        # Prototype fits only current TRAIN images/questions; no target/test features used.
        tick=time.monotonic();misses=x.feature_misses
        proto=dict(proto);proto[tid]=F.normalize(torch.stack([x.feature(r) for r in rows]).mean(0),dim=0)
        prototype_seconds=time.monotonic()-tick;prototype_misses=x.feature_misses-misses
        mapping=dict(mapping);mapping[tid]=commit
        assert set(proto)==set(mapping)==set(range(1,tid+1))
        decision={'stage':tid,'selected':chosen,'committed':commit,'forked':fork,'retained_experts':len(pool),
            'persistent_lora_parameters':540672*len(pool),'active_candidate_parameters':540672,
            'prototype_parameters':sum(v.numel() for v in proto.values()),'protected_tasks':protected,
            'current_before':before_current,'current_after':after_current,'old_before':old_before,'old_after':old_after,
            'candidate_vs_selected_deltaW_cosine':cosine,'auxiliary_train_forwards':sum(len(a['train_ids']) for a in aux),
            'old_probe_memory_QA_available':sum(len(memories[t]) for t in mapping),'auxiliary_scores':aux,
            'current_optimizer_qa':len(rows),'selection_uses_test_scores':False,
            'prototype_train_ids':[r['id'] for r in rows],'prototype_seconds':prototype_seconds,
            'prototype_actual_new_features':prototype_misses,
            'gradient_probe_train_qa':min(4,len(current_probes[tid]))+min(4,sum(len(memories[t]) for t in protected))}
        write(run/'decisions'/(label+'.json'),decision)
        evaluate(label+'_post_forced',selected_test,{domain:[]},'fixed',pool,proto,mapping,candidate)
        evaluate(label+'_post_router',selected_test,selected_gen,'input',pool,proto,mapping)
        x.set_route('fixed',pool,proto,mapping,'gradient_probe');x.restore(candidate)
        gradients={'current_train':current_probes[tid][:4]}
        old_memory=sorted([r for t in protected for r in memories[t]],key=lambda r:hashed(r,'capacity-gradient-memory'))[:4]
        if old_memory:gradients['protected_train']=old_memory
        ResearchExperiment.gradients(x,label,gradients)
        state={'stage':tid,'cap':cap,'pool':pool,'prototypes':proto,'assignments':mapping,'source_sha256':hashes}
        atomic_torch(run/'states'/(label+'.pt'),state);atomic_torch(latest,state)
        x.monitor.snapshot(label+'_training_after',model=x.model)
    # Snapshot after optimizer cleanup, before final diagnostic inference.
    x.monitor.snapshot('capacity_training_after',model=x.model)
    evaluate('final_input_router',tests,generation,'input',state['pool'],state['prototypes'],state['assignments'])
    evaluate('final_known_task_reference',tests,{d:[] for d in tests},'known_task_reference',state['pool'],state['prototypes'],state['assignments'])
    scope=(f'training stages{resume_stage+1}..19' if resume_stage<19 else 'resume final inference; no optimizer updates in current process')
    finish_resources(x,run,'capacity_training_before','capacity_training_after',scope+'; after snapshot precedes final diagnostic inference')
    artifacts=[latest,run/'events.jsonl',run/'resources.jsonl',run/'memory_summary.json',run/'router_input_gate.json',run/'protocol.json']
    artifacts+=list((run/'evaluations').rglob('*.jsonl'))+list((run/'decisions').glob('*.json'))
    artifacts+=list((run/'candidates').glob('*.pt'))+list((run/'states').glob('*.pt'))+list((run/'raw_probe_gradients').rglob('*.pt'))
    write(complete,{'complete':True,'run':str(run.relative_to(ROOT)),'source_sha256':hashes,
          'cap':cap,
          'final_state_sha256':digest(latest),'retained_experts':len(state['pool']),
          'formal_full_test':False,'artifact_sha256':{str(p.relative_to(run)):digest(p) for p in artifacts}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cap',type=int,choices=CAPS,required=True);parser.add_argument('--confirm-full',action='store_true');args=parser.parse_args()
    try:main(args.cap,args.confirm_full)
    except BlockingIOError:raise SystemExit(0)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True);name=f'cap{args.cap}'+('_full_confirm' if args.confirm_full else '')
        write(REPORT/(name+'_needs_attention.json'),{'traceback':traceback.format_exc(),'pid':os.getpid(),
              'action':'Preserve run; do not relax gate, drop samples, or retry algorithm blindly.'})
