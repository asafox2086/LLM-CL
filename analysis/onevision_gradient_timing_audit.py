"""CPU localization of recorded TRAIN gradient conflicts within a stage.

Uses completed old controls; records are not deterministic matched-causal proof.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,math
from pathlib import Path
import torch
from answer_f1 import answer_f1,VERSION

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    sources={}
    def read(path):
        sources[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    pointer=read(ROOT/'summary/onevision_mechanisms_20261016/task_domain_attempt.json')
    for name,digest in pointer['spec']['source_sha256'].items():assert sha(ROOT/name)==digest
    prior=read(REPORT/'task_domain_mechanism_audit.json');assert prior['run']==pointer['run']
    run=ROOT/pointer['run'];result=[];raw_verified=0;total_events=[]
    for sequence in pointer['spec']['sequences']:
        for method in ['anchor_gradient_noop','anchor_gradient_projection']:
            case=sequence+'__'+method
            complete=read(run/'cases'/f'{case}_complete.json');assert complete['complete']
            events=[]
            for stage in range(2,6):
                path=run/f'training/{case}/stage_{stage:02d}.json'
                assert sha(path)==prior['source_sha256'][str(path.relative_to(ROOT))]
                log=read(path);assert len(log['events'])==4
                for index,event in enumerate(log['events'],1):
                    assert event['stage']==stage and event['update']==index
                    assert event['lr_used']>0 and math.isfinite(event['loss']) and math.isfinite(event['gradient_l2_preclip'])
                    dot=event['anchor_gradient_dot'];assert dot is not None and math.isfinite(dot)
                    applied=method=='anchor_gradient_projection' and dot<0
                    assert event['projection_applied']==applied
                    events.append({'stage':stage,'update':index,'dot':dot,'negative_dot':dot<0,
                                   'projection_applied':applied,'train_ids':event['train_ids'],
                                   'reference_QA':event['reference_extra_QA'],
                                   'basis':'recorded per-update FP64 dot; full vectors only saved for update1'})
                path=run/f'raw_step_gradients/{case}/stage_{stage:02d}.pt'
                assert sha(path)==prior['source_sha256'][str(path.relative_to(ROOT))]
                side=read(path.with_suffix('.json'));assert sha(path)==side['sha256']
                raw=torch.load(path,map_location='cpu',weights_only=True)
                a=raw['pre_intervention_gradient'].double();b=raw['reference_gradient'].double()
                assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
                assert raw['reference_train_ids']==pointer['spec']['train_ids']['4']
                dot=float(torch.dot(a,b));recorded=log['events'][0]['anchor_gradient_dot']
                assert math.isclose(dot,recorded,rel_tol=1e-5,abs_tol=1e-7)
                assert (dot<0)==(recorded<0)
                sources[str(path.relative_to(ROOT))]=sha(path);raw_verified+=1
            endpoints={}
            for stage in [1,5]:
                path=run/f'evaluations/{case}_after{stage}/anchor_image.jsonl'
                durable=read(path.parent/'durable_complete.json');assert sha(path)==durable['files'][path.name]
                data=[json.loads(line) for line in path.read_text().splitlines()]
                assert [r['id'] for r in data]==pointer['spec']['anchor_test_ids']
                for row in data:assert answer_f1(row['prediction'],row['references'])==row['scores']['token_f1']
                sources[str(path.relative_to(ROOT))]=sha(path)
                endpoints[stage]=sum(r['scores']['token_f1'] for r in data)/len(data)
            first=[e for e in events if e['update']==1];later=[e for e in events if e['update']>1]
            entry={'sequence':sequence,'method':method,'first_updates_negative':sum(e['negative_dot'] for e in first),
                   'first_updates_total':len(first),'later_updates_negative':sum(e['negative_dot'] for e in later),
                   'later_updates_total':len(later),'negative_total':sum(e['negative_dot'] for e in events),
                   'all_updates_total':len(events),'projection_activations':sum(e['projection_applied'] for e in events),
                   'anchor_Answer_F1_S1':endpoints[1],'anchor_Answer_F1_S5':endpoints[5],
                   'anchor_F1_change':endpoints[5]-endpoints[1],'events':events}
            result.append(entry);total_events+=events
    assert len(result)==8 and len(total_events)==128 and raw_verified==32
    negative=[e for e in total_events if e['negative_dot']]
    total={'recorded_updates':128,'raw_vector_first_updates_recomputed':32,'negative_dot_updates':len(negative),
           'negative_first_updates':sum(e['update']==1 for e in negative),
           'negative_later_updates':sum(e['update']>1 for e in negative),
           'stage2_first_updates_negative':sum(e['stage']==2 and e['update']==1 and e['negative_dot'] for e in total_events)}
    out={'CPU_only':True,'metric':'Answer F1','metric_version':VERSION,'run':pointer['run'],
         'summary':total,'cases':result,'sources':sources,'audit_source_sha256':sha(Path(__file__)),
         'limits':['Original default-backend stage1 parameters differ; no method superiority or strict domain-count causality.',
                   'Only first minibatch full gradients retained; later signs are actual recorded FP64 dot logs, not recomputed complete vectors.',
                   'Reference is mean gradient of8 TRAIN anchor QA at each current weight, eval-mode; current TRAIN gradient uses train-mode/dropout.',
                   'Raw-gradient opposition is a local direction property, not measured old TEST loss/quality change and not an Adam update guarantee.',
                   'Updates/conditions are dependent; conflict frequencies are descriptive, not independent Bernoulli confidence tests.',
                   'Different conflict counts on projection/noop are different own trajectories; initial numerical differences forbid causal attribution.',
                   'Matched two-anchor48-case GPU study still pending; no new method or changed GPU training.']}
    for name,digest in sources.items():assert sha(ROOT/name)==digest
    assert not torch.cuda.is_initialized()
    (REPORT/'GRADIENT_TIMING_EVIDENCE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    lines=['# TRAIN梯度冲突发生在首步，还是同阶段后续更新？','',
           '只读已完成旧8条noop/projection轨迹；128个后续阶段更新日志与32个保存首批原始向量，SHA核验。不是新GPU实验，也不是确定性双锚实验的替代。','',
           f"共{total['negative_dot_updates']}次负点积，其中首批{total['negative_first_updates']}次、阶段内部后续批次{total['negative_later_updates']}次。进入stage2的8个首批均非负。多数负点积不在保存完整梯度的阶段首批，不能拿首批性质概括整阶段。",'',
           '| 序列 | 原机制 | 首批负点积/4 | 后续负点积/12 | 真正投影次数 | 锚Answer F1 S1→S5 |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in result:
        lines.append(f"| {row['sequence']} | {row['method']} | {row['first_updates_negative']}/4 | {row['later_updates_negative']}/12 | {row['projection_activations']} | {row['anchor_Answer_F1_S1']:.3f}→{row['anchor_Answer_F1_S5']:.3f} |")
    lines+=['','## 证明与边界','',
            '重复原锚的两轨迹没有记录到负点积，同医学新任务有少量，跨域轨迹更多；这是实际固定序列的描述性线索。它支持进一步检验阶段内部的局部方向冲突，不能证明跨域数量本身是原因。', '',
            '32个首批的原始向量在CPU重算点积全部通过；其余96更新仅有原运行当时记录的点积/范数，不声称保留了全部原始梯度。旧EWC最新中心首步梯度0的性质继续成立，但不能把它直接解释为所有后续冲突或性能缺陷的原因。', '',
            '质量列统一是原始Answer F1；原首阶段权重差异尚在。不能把投影与noop的终点差异称为投影收益，也不能将负点积解释成有限Adam更新后必然掉分。核心双锚匹配对照仍需完成。','']+['- '+v for v in out['limits']]
    (REPORT/'GRADIENT_TIMING_EVIDENCE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(total))


if __name__=='__main__':main()
