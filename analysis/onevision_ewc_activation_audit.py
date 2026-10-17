"""Read-only EWC first-update activation audit on actual saved checkpoints.

Not a method proposal or accuracy-causal claim. The original default-backend
4x6 control results retain their stage1 numerical and single-anchor confounds.
"""
import json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from onevision_task_domain_intervention import SEQUENCES,METHODS
from onevision_knowledge_intervention import case_complete
from onevision_research_worker import digest
from onevision_knowledge_pair_audit import paired_cluster
from run import freeze,score
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def read(p):return json.loads(p.read_text())
def l2(x):return float(torch.linalg.vector_norm(x.double()))


def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);assert not torch.cuda.is_initialized()
    manifest,data,domains,tests,*_=freeze();anchor=[r for r in tests['medical'] if r['_task']==4]
    pointer=ROOT/'summary/onevision_mechanisms_20261016/task_domain_attempt.json';info=read(pointer);run=ROOT/info['run']
    sources={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))}
    for rel,h in info['spec']['source_sha256'].items():assert digest(ROOT/rel)==h
    def load(p):
        assert digest(p)==read(p.with_suffix('.json'))['sha256']
        sources[str(p.relative_to(ROOT))]=digest(p);return torch.load(p,map_location='cpu',weights_only=True)
    formats=[];gradients=[];exact_optimizer=[];per_row={};nonstationarity=[]
    for sequence in SEQUENCES:
        for method in METHODS:
            case=sequence+'__'+method
            assert read(run/'cases'/f'{case}_complete.json')['complete']
            for stage in [1,3,5]:
                label=f'{case}_after{stage}';assert case_complete(run,label,{'anchor':anchor})
                p=run/f'evaluations/{label}/anchor_image.jsonl';sources[str(p.relative_to(ROOT))]=digest(p)
                rows=[json.loads(s) for s in p.read_text().splitlines()];assert len(rows)==23
                aux=wrapped=0;aux_scores=[]
                for row in rows:
                    text=row['prediction'].strip()
                    if text.lower().startswith('the answer is '):wrapped+=1;text=text[len('the answer is '):].strip()
                    value=score(text,row['references'])['exact_match'];aux+=value==100;aux_scores.append(value)
                per_row[sequence,method,stage]={'original':[r['scores']['exact_match'] for r in rows],'fixed_prefix_aux':aux_scores}
                formats.append({'sequence':sequence,'method':method,'stage':stage,'count':23,
                                'original_EM_correct':sum(r['scores']['exact_match']==100 for r in rows),
                                'fixed_prefix_aux_EM_correct':aux,'prefix_count':wrapped})
            if method.startswith('anchor_gradient_'):
                raw_path=run/'raw_step_gradients'/case/'stage_02.pt';raw=load(raw_path)
                previous_path=run/'adapters'/case/'stage_01.pt';previous=load(previous_path)
                assert raw['reference_train_ids']==info['spec']['train_ids']['4']
                g_old=raw['reference_gradient'].double();g_new=raw['pre_intervention_gradient'].double()
                assert torch.isfinite(g_old).all() and l2(g_old)>0
                nonstationarity.append({'sequence':sequence,'control':method,'stage':2,'old_TRAIN_QA':8,
                                         'old_TRAIN_mean_gradient_L2':l2(g_old),
                                         'new_minibatch_old_TRAIN_gradient_cosine':float(torch.dot(g_old,g_new))/(l2(g_old)*l2(g_new)),
                                         'source_checkpoint':str(previous_path.relative_to(ROOT)),
                                         'scope':'actual old-gradient at this control own checkpoint; not copied into a different EWC run',
                                         'centered_quadratic_old_gradient_at_this_point':0.0})
            if not method.startswith('EWC_'):continue
            coefficient=1 if method=='EWC_lambda1' else 100
            state=load(run/'case_states'/case/'latest.pt');assert state['trained_stage']==5 and len(state['fishers'])==5
            for stage in [2,3,4,5]:
                previous=load(run/'adapters'/case/f'stage_{stage-1:02d}.pt')['adapter']
                p=run/'raw_step_gradients'/case/f'stage_{stage:02d}.pt';raw=load(p)
                training_path=run/'training'/case/f'stage_{stage:02d}.json';training=read(training_path);sources[str(training_path.relative_to(ROOT))]=digest(training_path)
                event=training['events'][0];assert event['lr_used']>0
                regularizer=[];latest_term=[];penalty=0.;flat_parameters=[];exact_latest=True
                for name,shape in raw['parameter_layout']:
                    base,factor,_=name.rsplit('.',2);parameter=previous[base][factor].double();assert list(parameter.shape)==shape
                    flat_parameters.append(parameter.flatten().float())
                    terms=[]
                    for fisher in state['fishers'][:stage-1]:
                        delta=parameter-fisher['mean'][name].double();diagonal=fisher['diagonal'][name].double()
                        penalty+=coefficient*.5*float((diagonal*delta.square()).sum())
                        terms.append(coefficient*diagonal*delta)
                    exact_latest &= torch.equal(parameter,state['fishers'][stage-2]['mean'][name].double())
                    latest_term.append(terms[-1].flatten());regularizer.append(sum(terms).flatten())
                reg=torch.cat(regularizer);latest=torch.cat(latest_term);raw_g=raw['pre_intervention_gradient'].double();target=raw_g-reg
                assert exact_latest and int(torch.count_nonzero(latest))==0
                assert abs(penalty-event['penalty'])<=max(1e-6,1e-4*abs(event['penalty'])),(case,stage,penalty,event['penalty'])
                norm_r,norm_t=l2(reg),l2(target)
                gradients.append({'sequence':sequence,'method':method,'stage':stage,
                                  'total_EWC_gradient_l2':norm_r,'latest_task_EWC_gradient_l2':l2(latest),
                                  'reconstructed_target_gradient_l2':norm_t,'raw_total_gradient_l2':l2(raw_g),
                                  'EWC_over_target_gradient_ratio':norm_r/norm_t if norm_t else None,
                                  'EWC_target_cosine':float(torch.dot(reg,target))/(norm_r*norm_t) if norm_r*norm_t else None,
                                  'reconstructed_penalty':penalty,'recorded_penalty':event['penalty'],
                                  'prior_Fisher_terms':stage-1,'latest_term_mean_exactly_previous_checkpoint':exact_latest,
                                  'reconstruction_scope':'first minibatch only; CPU FP64 derivative of saved penalty, not a separately saved data-only gradient'})
                if stage==2:
                    assert torch.count_nonzero(reg)==0
                    starting=torch.cat(flat_parameters);results=[]
                    for coefficient_test in [0,1,100]:
                        parameter=torch.nn.Parameter(starting.clone());parameter.grad=(target+coefficient_test*latest).float().clone()
                        torch.nn.utils.clip_grad_norm_([parameter],1.)
                        optimizer=torch.optim.AdamW([parameter],lr=1e-4,weight_decay=0);optimizer.step()
                        results.append(parameter.detach().clone())
                    if not all(torch.equal(results[0],value) for value in results[1:]):
                        failure={'sequence':sequence,'method':method,'stage':stage,'gate_failed':True,
                                 'max_abs_differences':[float((results[0]-value).abs().max()) for value in results[1:]],
                                 'source_sha256':digest(Path(__file__)),'GPU_executed':False}
                        (REPORT/'ewc_cpu_counterfactual_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                        raise AssertionError(failure)
                    exact_optimizer.append({'sequence':sequence,'checkpoint_method':method,'stage':stage,
                                            'same_saved_start_and_target_gradient':True,'CPU_flat_fresh_AdamW_lambdas':[0,1,100],
                                            'updated_parameters_exact_equal':True,'full_GPU_update_or_accuracy_test':False})
    assert len(gradients)==32 and len(exact_optimizer)==8 and len(formats)==72 and len(nonstationarity)==8 and not torch.cuda.is_initialized()
    domain_contrasts=[]
    for method in METHODS:
        for treatment,control in [('within_medical','repeat_anchor'),('cross_short','within_medical'),('cross_five_domains','within_medical'),('cross_five_domains','cross_short')]:
            contrasts={}
            for metric in ['original','fixed_prefix_aux']:
                a1=per_row[treatment,method,1][metric];a5=per_row[treatment,method,5][metric]
                b1=per_row[control,method,1][metric];b5=per_row[control,method,5][metric]
                contrasts[metric]=paired_cluster([w-x-y+z for w,x,y,z in zip(a5,a1,b5,b1)],[tuple(r['images']) for r in anchor])
            domain_contrasts.append({'method':method,'treatment':treatment,'control':control,'difference_in_changes_EM_pp':contrasts})
    result={'source_sha256':sources,'run':info['run'],'first_minibatch_regularizer':gradients,
            'CPU_same_state_optimizer_counterfactual':exact_optimizer,'anchor_format':formats,
            'CPU_optimizer_backend':'deterministic algorithms,4 CPU threads; no GPU execution',
            'initial_CPU_assertion_failure':'ewc_cpu_counterfactual_initial_failure.json; original uncontexualized failure preserved, cause not established',
            'domain_contrasts':domain_contrasts,
            'old_TRAIN_checkpoint_nonstationarity':nonstationarity,
            'conclusion':'Latest-task centered quadratic EWC term has zero first derivative at each new-stage start; at stage2 all EWC terms are zero. Earlier-task terms can be nonzero later.',
            'limits':['mathematical activation property, not proof EWC useless or cause of accuracy deficits',
                      'actual pilot has4 updates/stage; within-stage later EWC terms are active',
                      'CPU optimizer counterfactual uses flat parameter/gradient tensors; not a claimed exact GPU first-step reconstruction',
                      'default-backend same stage1 weights differ; no causal method ranking',
                      'one23-QA anchor/one seed/order; other regularizers may behave differently',
                      'fixed prefix is post-hoc sensitivity only, original main scores unchanged']}
    (REPORT/'EWC_ACTIVATION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    out=['# EWC在极少更新设置下何时开始起作用','',
         'CPU只读实际8个EWC条件、32个阶段首批梯度和对应检查点／Fisher，全部sidecar SHA验证。另核验4序列×6机制×3锚时点的72个已完成输出。原运行和评分不变。','',
         '## 可直接验证的机制性质','',
         'EWC项为λ/2 Σ F(θ−θ_old)²。在刚进入新任务时，最新旧任务的中心恰等于当前参数，因此其梯度为0。实际32/32个阶段首批均验证这一点；进入第二阶段仅有这一项，因此8/8条件的整个EWC梯度均为0。',
         '', '用实际保存的相同起点、同一个首批目标梯度，在严格确定性CPU模式独立构造fresh AdamW：λ=0／1／100的第一步参数逐元素相同（8/8条件）。这是条件固定的梯度／优化器反事实，不是原GPU轨迹完全复现或准确率因果证据。首次未设确定性的CPU断言失败已保留；当时未记录具体失败条件，后续检查未稳定复现，不能据此认定确定性开关就是其成因。严格相等标准未放宽，原GPU源文件未改。',
         '', '**这意味着：只有一次更新的新任务，不能指望以最新参数为中心的二次EWC项在该第一步阻止偏离。** 但更多旧任务的中心可能已不同，进入第三阶段后它们可立即产生非零梯度；其他正则化方法未必具有同一性质。不能概括为EWC或LoRA毫无作用。',
         '', '## 同时核查：旧任务是否已到驻点','',
         '另读取noop／projection八个控制各自阶段1权重后、阶段2首批更新前保存的原8条Organ TRAIN均值梯度。全部非零：范数约63.96–72.50。相同点若构造以它自身为中心的二次项，梯度仍为0。因此这些有限预算检查点的真实旧任务一阶项未消失，中心二次项本身并不匹配它。',
         '', '这是各控制自身权重的性质，不把其梯度当作另一个EWC条件的梯度，也不把任意非零梯度当作“新任务一定损害旧任务”。新／旧梯度可能相容；实际准确率因果影响未测定。原4×6条件未达到相同权重／确定性复现要求，不能用跨条件小差异作因果证明。',
         '', '## 后续阶段并非全部失活','',
         '| 序列 | EWC λ | 阶段 | 首批EWC／目标梯度范数 | 最新任务项范数 |', '| --- | ---: | ---: | ---: | ---: |']
    for row in gradients:
        out.append(f"| {row['sequence']} | {1 if row['method']=='EWC_lambda1' else 100} | {row['stage']} | {row['EWC_over_target_gradient_ratio']:.6f} | {row['latest_task_EWC_gradient_l2']:.1f} |")
    out+=['','原试验每阶段4步，而非只做1步；第二步起相对于最新中心的偏移可触发保护项。记录的正则标量可达约0.24，不能把整段训练解释成正则项从未生效。首批已存raw梯度包含正则项，data-only梯度由保存的Fisher／中心解析相减，不是额外保存的原GPU数据梯度。','',
          '## 固定前缀敏感性：原锚结果是否只是格式','',
          '| 序列 | 机制 | S1原／辅助 | S3原／辅助 | S5原／辅助 |', '| --- | --- | ---: | ---: | ---: |']
    index={(r['sequence'],r['method'],r['stage']):r for r in formats}
    for sequence in SEQUENCES:
        for method in METHODS:
            values=[index[sequence,method,s] for s in [1,3,5]]
            out.append(f"| {sequence} | {method} | "+' | '.join(f"{r['original_EM_correct']}/{r['fixed_prefix_aux_EM_correct']}" for r in values)+' |')
    out+=['','以上分母均23。原／辅助只区分固定英文前缀，不代表语义人工评分；跨域SeqFT最终原EM4/23、辅助5/23，同域最终两者均9/23，说明这一个前缀不能消除全部差距。',
          '', '## 同样任务数下的格式敏感性对比','',
          '| 机制 | 序列差（S5−S1变化之差） | 原EM pp [95%] | 固定前缀辅助 pp [95%] |',
          '| --- | --- | --- | --- |']
    for row in domain_contrasts:
        def show(v):return f"{v['mean']:+.2f} [{v['image_cluster_percentile95'][0]:+.2f}, {v['image_cluster_percentile95'][1]:+.2f}]"
        values=row['difference_in_changes_EM_pp']
        out.append(f"| {row['method']} | {row['treatment']} − {row['control']} | {show(values['original'])} | {show(values['fixed_prefix_aux'])} |")
    out+=['','区间按原图像路径组bootstrap10000次、seed42，未校正多重比较。差上差不能消除S1参数数值差，也不能消除任务内容／答案长度差；不能据此作领域数唯一成因或方法优劣的严格因果结论。',
          '', '## 因果边界','',
          '仍需确定性后端、相同初始权重及任务／内容／token预算控制来判定这种首步失活是否造成最终性能问题。原默认4×6对照S1权重已有数值差异，不能以此报告细小方法优劣；单锚也不能证明领域数是唯一原因。本审计直接证明的是更新机制的启用条件，尚非它对生成准确率的因果影响。',
          '', '完整首批范数、正则值、CPU反事实、72格输出和SHA见[机器可读记录](EWC_ACTIVATION_AUDIT.json)。']
    (REPORT/'EWC_ACTIVATION_AUDIT.md').write_text('\n'.join(out)+'\n')
    print(json.dumps({'first_minibatches':len(gradients),'latest_EWC_term_exact_zero':32,'whole_EWC_zero_at_stage2':8,
                      'CPU_same_start_updates_exact':8,'format_cells':len(formats),
                      'format_cells_changed':sum(r['original_EM_correct']!=r['fixed_prefix_aux_EM_correct'] for r in formats)}))


if __name__=='__main__':main()
