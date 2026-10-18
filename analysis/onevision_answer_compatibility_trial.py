"""Frozen-state TRAIN Answer F1 feedback and a small input metric.

Separate diagnostic: zero new backbone updates; original93 updates charged.
All TRAIN teacher feedback precedes both new TEST conditions, with no selection.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import fcntl
import json
from pathlib import Path
import time
import torch
import onevision_answer_compatibility_core as core
import onevision_state_cache_trial as parent
from onevision_knowledge_intervention import reset_seed,case_complete
from onevision_research_worker import atomic_torch,digest
from onevision_capacity_probe import finish_resources,stratify_full_results
from run import write

ROOT,BASE=parent.ROOT,parent.BASE
REPORT=BASE/'answer_compatibility_20261018/v1'
FAMILY=ROOT/'exp/onevision_answer_compatibility_20261018/v1'
VIEWS=['cache_AnswerF1','cache_AnswerF1_metric']


def read(p):return json.loads(p.read_text())


def plan():
    previous,values=parent.plan()
    assert parent.complete('build',None,previous)
    proof=read(parent.REPORT/'build_complete.json');build=ROOT/proof['run']
    sources=dict(previous['source_sha256'])
    for n in ['onevision_answer_compatibility_core.py','onevision_answer_compatibility_trial.py',
              'onevision_answer_compatibility_entry.py']:
        sources['analysis/'+n]=digest(ROOT/'analysis'/n)
    spec=dict(source_sha256=sources,protocol_sha256=digest(REPORT/'PROTOCOL.md'),
        parent_spec=previous,parent_complete_sha256=digest(parent.REPORT/'build_complete.json'),
        fixed_pool_sha256=digest(build/'final_state.pt'),views=VIEWS,teacher_TRAIN_QA=20,
        teacher_candidates=[12,18,19],teacher_generation_QA=60,
        metric_parameters=1792,metric_steps=core.STEPS,metric_lr=core.LR,
        kernel_temperature=core.KERNEL_TEMPERATURE,expert_temperature=core.EXPERT_TEMPERATURE,
        metric_L2=core.L2,metric_log_weight_limit=core.LOG_WEIGHT_LIMIT,
        both_full21454_regardless_diagnostic=True,new_backbone_updates=0,
        original_backbone_updates_charged=93,TEST_selection=False,
        same_image_mask_is_training_objective_not_independent_validation=True,
        original_control_full500_155='cache_geometry exact source; seq/NLL original sealed results referenced, not new reruns')
    return spec,values,build


def complete(phase,view,spec):
    name='build' if phase=='build' else 'full_'+view;p=REPORT/(name+'_complete.json')
    if not p.exists():return False
    proof=read(p);assert proof['complete'] and proof['spec']==spec
    run=ROOT/proof['run'];assert run.resolve().is_relative_to(FAMILY.resolve())
    for rel,s in proof['artifact_sha256'].items():
        p=run/rel;assert p.resolve().is_relative_to(run.resolve()) and digest(p)==s
    assert read(run/'original_cache_geometry_exact.json')['passed']
    if phase=='full':assert proof['full_QA']==21454 and read(run/'feedback_reload_exact.json')['passed']
    return True


class FeedbackExperiment(parent.CacheExperiment):
    def __init__(self,manifest,run):
        self.feedback=None
        super().__init__(manifest,run)

    def set_feedback(self,state=None,scores=None,theta=None,view=None):
        self.set_calling()
        self.feedback=None if state is None else (state,scores,theta,view)
        self.active_key=None

    def activate_route(self,row):
        if self.feedback is None:return super().activate_route(row)
        state,scores,theta,view=self.feedback;keys=sorted(state['pool'])
        selected,details=core.infer(self.feature(row),state['memory_features'],scores,theta,keys,
                                  torch.stack([state['prototypes'][k] for k in keys]))
        if self.active_key!=selected:
            self.restore(state['pool'][selected]);self.active_key=selected
        self.last_route=dict(route_mode=view,route_state=selected,route_details=details)


def compare(run,label,source,label_source,domains):
    for domain in domains:
        assert parent.semantic(parent.rows_at(run,label,domain),False)==parent.semantic(
            parent.rows_at(source,label_source,domain),False),f'Original exact gate failed: {domain}'


def run_trial(phase,view):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['0','1']
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    spec,values,source_build=plan()
    assert read(REPORT/'CPU_PLAN_CHECK.json')['spec']==spec
    assert read(REPORT/'ADMISSION.json')['spec']==spec
    assert phase=='build' or view in VIEWS
    lock=(REPORT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    name='build' if phase=='build' else 'full_'+view
    assert not (REPORT/(name+'_attempt.json')).exists(),'Preserve failures; no blind retry'
    manifest,data,domains,tests,full,native,native_proof=values
    run=FAMILY/name/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
    write(REPORT/(name+'_attempt.json'),dict(spec=spec,run=str(run.relative_to(ROOT)),pid=os.getpid()))
    write(run/'spec.json',spec)
    import onevision_research_device_entry as device
    cards=device.inventory();gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert device.select(cards,os.getuid())==gpu
    write(run/'physical_device_execution.json',dict(physical_GPU=gpu,admission_inventory=cards,
        actual_CUDA_VISIBLE_DEVICES=str(gpu),single_research_stream=True,max_our_GPU_cards=3))
    x=FeedbackExperiment(manifest,run)
    state=torch.load(source_build/'final_state.pt',map_location='cpu',weights_only=True)
    assert sorted(state['pool'])==spec['teacher_candidates'] and len(state['memory_ids'])==20
    x.configure(state['pool'][19],finetune_shared=False)
    assert not any(p.requires_grad for p in x.model.parameters())
    generation={d:[r for r in rs if r['id'] in set(spec['parent_spec']['generation_ids'][d])]
                for d,rs in tests.items()}
    assert sum(map(len,generation.values()))==155
    x.monitor.snapshot('feedback_process_before',model=x.model)
    if phase=='build':
        # One full original control establishes unchanged model/input/routing.
        # Other original sealed views stay explicitly referenced, never spliced.
        x.set_calling(state,'cache_geometry')
        x.evaluate('control_cache_geometry',tests,generation)
        assert case_complete(run,'control_cache_geometry',tests)
        compare(run,'control_cache_geometry',source_build,'cache_geometry',tests)
        write(run/'original_cache_geometry_exact.json',dict(passed=True,QA=500,generated=155,
            exact_route_and_semantic_fields=True,parent_complete_sha256=spec['parent_complete_sha256']))
        train={r['id']:r for t in range(1,20) for r in data[t,'train']}
        teacher_rows=[train[i] for i in state['memory_ids']]
        assert len(train)==1313 and len(teacher_rows)==20
        score_rows=[];teacher_sha={}
        for k in sorted(state['pool']):
            x.set_feedback();x.configure(state['pool'][k],finetune_shared=False)
            label='TRAIN_teacher_state'+str(k)
            selected={'TRAIN_memory':teacher_rows}
            x.evaluate(label,selected,selected)
            assert case_complete(run,label,selected) and parent.same(x.snapshot(),state['pool'][k])
            f=run/f'evaluations/{label}/TRAIN_memory_image.jsonl';teacher_sha[str(k)]=digest(f)
            predictions=[json.loads(line) for line in f.read_text().splitlines()]
            assert [r['id'] for r in predictions]==state['memory_ids']
            from answer_f1 import answer_f1
            for r,row in zip(predictions,teacher_rows):
                assert r['references']==row['answers']
                assert r['scores']['token_f1']==answer_f1(r['prediction'],r['references'])
            score_rows.append([r['scores']['token_f1']/100 for r in predictions])
        scores=torch.tensor(score_rows,dtype=torch.float32)
        groups=[tuple(r['images']) for r in teacher_rows]
        x.monitor.snapshot('CPU_router_TRAIN_before',model=x.model)
        tick=time.monotonic()
        theta,events,gradients=core.fit(state['memory_features'],scores,groups)
        seconds=time.monotonic()-tick
        x.monitor.snapshot('CPU_router_TRAIN_after',model=x.model)
        atomic_torch(run/'router_raw_gradients.pt',dict(gradients=gradients,TRAIN_ids=state['memory_ids']))
        atomic_torch(run/'feedback_state.pt',dict(theta=theta,scores=scores,TRAIN_ids=state['memory_ids'],
            source_pool_sha256=spec['fixed_pool_sha256'],source_complete_sha256=spec['parent_complete_sha256'],
            teacher_sha256=teacher_sha,metric_parameters=theta.numel()))
        write(run/'router_TRAIN.json',dict(events=events,CPU_seconds=seconds,TRAIN_ids=state['memory_ids'],
            theta_l2=float(theta.norm()),actual_nonzero_gradient_steps=sum(int(bool(torch.count_nonzero(g))) for g in gradients),
            actual_changed_theta_dimensions=int(torch.count_nonzero(theta)),teacher_generation_QA=60,
            teacher_sha256=teacher_sha,CPU_parameter_bytes=theta.numel()*theta.element_size(),
            no_TEST_input_to_fit=True,no_new_backbone_updates=True,source93updates_charged=True,
            image_exclusion_is_training_objective_not_validation=True))
        for which in VIEWS:
            actual_theta=theta if which.endswith('_metric') else torch.zeros_like(theta)
            x.set_feedback(state,scores,actual_theta,which)
            x.evaluate(which,tests,generation)
            assert case_complete(run,which,tests)
    else:
        assert complete('build',None,spec)
        build=ROOT/read(REPORT/'build_complete.json')['run']
        trained=torch.load(build/'feedback_state.pt',map_location='cpu',weights_only=True)
        assert trained['source_pool_sha256']==spec['fixed_pool_sha256']
        theta=trained['theta'] if view.endswith('_metric') else torch.zeros_like(trained['theta'])
        x.set_feedback(state,trained['scores'],theta,view)
        x.evaluate('candidate_reload',tests,generation)
        compare(run,'candidate_reload',build,view,tests)
        write(run/'feedback_reload_exact.json',dict(passed=True,QA=500,generated=155,original_route_exact=True))
        write(run/'original_cache_geometry_exact.json',dict(passed=True,
            inherited_build_gate_sha256=digest(build/'original_cache_geometry_exact.json')))
        x.evaluate('final_full_input_router',full,full)
        assert case_complete(run,'final_full_input_router',full)
        stratify_full_results(run,full,{r['id'] for rows in tests.values() for r in rows})
    assert digest(source_build/'final_state.pt')==spec['fixed_pool_sha256']
    assert not any(p.requires_grad for p in x.model.parameters())
    x.monitor.snapshot('feedback_process_after',model=x.model)
    finish_resources(x,run,'feedback_process_before','feedback_process_after',
        'samePID feedback/inference; zero new backboneTRAIN; CPU router window separate; historical93updates charged')
    files={str(p.relative_to(run)):digest(p) for p in run.rglob('*')
           if p.is_file() and p.suffix in ['.json','.jsonl','.pt']}
    write(REPORT/(name+'_complete.json'),dict(complete=True,spec=spec,phase=phase,view=view,
        run=str(run.relative_to(ROOT)),artifact_sha256=files,full_QA=21454 if phase=='full' else None,
        new_backbone_updates=0,parent_backbone_updates_charged=93,full_FWT=None,full_BWT=None))
    assert complete(phase,view,spec)


def main(check=False,phase=None,view=None):
    REPORT.mkdir(parents=True,exist_ok=True)
    if check:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        spec,values,build=plan();state=torch.load(build/'final_state.pt',map_location='cpu',weights_only=True)
        train={r['id']:r for t in range(1,20) for r in values[1][t,'train']}
        allowed=core.image_mask([tuple(train[i]['images']) for i in state['memory_ids']])
        assert allowed.shape==(20,20) and allowed.any(dim=1).all()
        write(REPORT/'CPU_PLAN_CHECK.json',dict(CPU_only=True,spec=spec,actual_TRAIN_mask_checked=True,
            real_TRAIN_generation_not_yet_run=True,GPU_started=False))
    else:run_trial(phase,view)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    p.add_argument('--phase',choices=['build','full']);p.add_argument('--view',choices=VIEWS)
    a=p.parse_args();main(a.check,a.phase,a.view)
