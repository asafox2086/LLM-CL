"""Read-only native final20 TRAIN routing fit; no TEST predictions/answers."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import onevision_state_cache_core as core

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1'
REPORT=ROOT/'summary/onevision_diagnostics_20261016/train_calling_fit_20261018'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate(state,train,leave_image_out):
    keys=sorted(state['pool']);memory=state['memory_features']
    losses=torch.tensor([[state['losses'][k][i] for i in state['memory_ids']] for k in keys])
    prototype=torch.stack([state['prototypes'][k] for k in keys]);records=[]
    for j,feature in enumerate(memory):
        similarities=F.normalize(memory.float(),dim=1)@F.normalize(feature.float(),dim=0)
        allowed=torch.tensor([not(set(train[ident]['images'])&set(train[state['memory_ids'][j]]['images']))
                              for ident in state['memory_ids']]) if leave_image_out else torch.ones(len(memory),dtype=torch.bool)
        assert allowed.any()
        weight=F.softmax((similarities/core.TEMPERATURE).masked_fill(~allowed,float('-inf')),dim=0)
        estimates=losses@weight;index=int(estimates.argmin())
        if not leave_image_out:
            original,details=core.route(feature,keys,prototype,memory,losses,True)
            assert original==keys[index] and details['estimated_TRAIN_NLL']==estimates.tolist()
        actual=losses[:,j];best=int(actual.argmin())
        records.append(dict(TRAIN_id=state['memory_ids'][j],chosen=keys[index],lowest_NLL=keys[best],
            chosen_NLL=float(actual[index]),lowest_NLL_value=float(actual[best]),
            NLL_regret=float(actual[index]-actual[best]),weights=weight.tolist(),
            self_weight=float(weight[j]),effective_anchors=float(1/weight.square().sum()),
            actual_NLL_by_state={str(k):float(actual[i]) for i,k in enumerate(keys)}))
    return dict(rows=records,chosen_counts=dict(Counter(r['chosen'] for r in records)),
        lowest_NLL_counts=dict(Counter(r['lowest_NLL'] for r in records)),
        mismatches=sum(r['chosen']!=r['lowest_NLL'] for r in records),
        mean_NLL_regret=sum(r['NLL_regret'] for r in records)/len(records))


def main():
    torch.set_num_threads(1);proof_path=SOURCE/'build_complete.json';proof=json.loads(proof_path.read_text())
    assert proof['complete']
    for name,s in proof['spec']['source_sha256'].items():assert sha(ROOT/name)==s
    run=ROOT/proof['run']
    assert run.resolve().is_relative_to((ROOT/'exp/onevision_state_cache_20261018/v1').resolve())
    for name,s in proof['artifact_sha256'].items():
        p=run/name;assert p.resolve().is_relative_to(run.resolve()) and sha(p)==s
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest_path)==proof['spec']['manifest_sha256']
    manifest=json.loads(manifest_path.read_text());train={}
    for task in manifest['tasks']:
        f=task['files']['train'];p=ROOT/f['path'];assert sha(p)==f['sha256']
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        train.update({r['id']:r for r in rows})
    assert len(train)==1313
    state=torch.load(run/'final_state.pt',map_location='cpu',weights_only=True)
    assert len(state['memory_ids'])==20 and all(i in train for i in state['memory_ids'])
    result=dict(CPU_only=True,analyzer_sha256=sha(Path(__file__)),source_complete_sha256=sha(proof_path),
        final_state_sha256=sha(run/'final_state.pt'),TEST_predictions_or_answers_parsed=False,
        bound_TEST_file_bytes_hashed_only=True,GPU_started=False,new_optimizer_updates=0,
        original_self_inclusive=evaluate(state,train,False),TRAIN_leave_same_image_out=evaluate(state,train,True),
        limits=['TRAIN NLL routing fit, not TEST Answer F1 or its accuracy cause.',
                'Leave-image-out removes only interpolation identity, not source/model/retention TRAIN exposure.',
                'Loss interpolation can miss a measured TRAIN optimum; no claim input representation cannot learn.',
                'A new TRAIN Answer F1 feedback experiment is prepared separately; not admitted or run by this audit.'])
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n')
    a=result['original_self_inclusive'];b=result['TRAIN_leave_same_image_out']
    text=f'''# 质量调用在已有TRAIN锚上的拟合偏差

**探究的问题：** 当前距离核插值能否在已知各候选NLL的原20条TRAIN上选出最低者？

**做的实验：** 只读封存native最后3状态与原20TRAIN向量/NLL，执行原始函数精确比对，并做仅插值层的同图像排除。没有GPU、权重更新或TEST预测/答案解析。

**结果：** 原含自身核在20条上有{a['mismatches']}条不选最低NLL；平均NLL差{a['mean_NLL_regret']:.6f}。调用计数{a['chosen_counts']}，实际最低计数{a['lowest_NLL_counts']}。同图像排除后不匹配{b['mismatches']}条、均差{b['mean_NLL_regret']:.6f}。逐题差距保留，不能把极小差距称重大损害。

**证明了什么：** 本固定核的匹配存在训练锚拟合偏差，不能称已知有用候选一定会被调用。它没有证明这8类/条偏差造成TEST F1下降，也没有分离距离、平滑、有限锚覆盖和NLL/生成目标差异。来源模型/保留决策已见TRAIN，排除自身不是独立泛化评测。

**改进的方法：** 先让固定原TRAIN锚提供各冻结候选的真实Answer F1反馈，再独立比较原几何、NLL、F1与轻量可学习距离；所有设置TRAIN预定，禁止TEST选指标/路由。

**改进的效果：** 此处只完成偏差核验，不算方法F1增益；新核心CPU合成检查不能代替真实反馈和GPU成效。

[原始逐题/权重/SHA](EVIDENCE.json) · [独立后续协议](../answer_compatibility_20261018/v1/PROTOCOL.md)
'''
    (REPORT/'RESEARCH_FINDINGS.md').write_text(text)
    print(json.dumps(dict(self_inclusive_mismatches=a['mismatches'],self_mean_regret=a['mean_NLL_regret'],
        image_LOO_mismatches=b['mismatches'],image_LOO_mean_regret=b['mean_NLL_regret'],TEST_F1_evidence=False)))


if __name__=='__main__':main()
