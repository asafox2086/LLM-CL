"""Read-only CPU synthesis of durable completed controls; no model loading."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'exp/onevision_diagnostics_20261016'))
from run import freeze
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def lines(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def paired(a,b,images):
    assert [x['id'] for x in a]==[x['id'] for x in b]
    groups={}
    for x,y in zip(a,b):
        groups.setdefault(tuple(images[x['id']]),[]).append(y['scores']['exact_match']-x['scores']['exact_match'])
    totals=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(42);est=[]
    for _ in range(10000):
        idx=rng.integers(0,len(totals),len(totals));est.append(float(totals[idx].sum()/counts[idx].sum()))
    return {'delta_EM_pp':float(totals.sum()/counts.sum()),'image_cluster_percentile_95':np.percentile(est,[2.5,97.5]).tolist(),
            'QA':len(a),'image_groups':len(groups),'bootstrap_seed':42,'bootstrap_repeats':10000}


def verify(run,proof):
    for name,expected in proof['artifact_sha256'].items():assert sha(run/name)==expected,(run,name)


def main():
    _,data,_,_,_,_,_=freeze()
    images={r['id']:r['images'] for (_,split),rows in data.items() if split=='test' for r in rows}
    basis_proof=REPORT/'basis_budget/complete.json';basis_root=ROOT/read(basis_proof)['run']
    basis=[];prediction={}
    for path in sorted(basis_root.glob('*/complete.json')):
        run=path.parent;proof=read(path);verify(run,proof)
        training=read(run/'training.json');assert len(training['records'])==4
        assert all(r['learning_rate_used']>0 for r in training['records'])
        endpoints={}
        for update in [1,4]:
            p=run/f'evaluations/endpoint_{update}/native_image.jsonl';rows=lines(p)
            endpoints[update]={'count':len(rows),'correct':sum(r['scores']['exact_match']==100 for r in rows),
                               'summary':read(p.parent/'summary.json')['native:image']}
            prediction[run.name,update]=rows
        basis.append({'case':run.name,'endpoints':endpoints,'memory_bytes':read(run/'memory_summary.json'),
                      'proof_sha256':sha(path),'training':training})
    assert len(basis)==9
    contrasts={}
    for task in ['T02','T11','T12']:
        for start in ['random_A_zero_B','past_A_zero_B','past_AB']:
            name=task+'_'+start
            contrasts[name+'_4_minus_1']=paired(prediction[name,1],prediction[name,4],images)
        for update in [1,4]:
            contrasts[f'{task}_pastAB_minus_pastAzeroB_at{update}']=paired(prediction[task+'_past_A_zero_B',update],prediction[task+'_past_AB',update],images)
    caps=[];cap_predictions={}
    for cap in [1,2,3,5]:
        p=REPORT/f'capacity/cap{cap}_complete.json';proof=read(p);run=ROOT/proof['run']
        rows=[]
        for f in sorted((run/'evaluations/final_input_router').glob('*_image.jsonl')):
            assert sha(f)==proof['artifact_sha256'][str(f.relative_to(run))]
            rows += [r for r in lines(f) if 'scores' in r]
        rows.sort(key=lambda r:r['id']);assert len(rows)==155
        cap_predictions[cap]=rows
        caps.append({'cap':cap,'retained_experts':proof['retained_experts'],'generated_QA':155,
                     'EM_correct':sum(r['scores']['exact_match']==100 for r in rows),'completion_proof_sha256':sha(p)})
    cap_contrasts={str(cap):paired(cap_predictions[1],cap_predictions[cap],images) for cap in [2,3,5]}
    result={'basis_completion_sha256':sha(basis_proof),'basis_cases':basis,'basis_paired_contrasts':contrasts,
            'capacity_diagnostic':caps,'capacity_paired_vs_cap1':cap_contrasts,
            'limitations':['single seed, no multiplicity correction','basis uses extra4 passes, not formal one-epoch CL',
                          '155 generated/500 NLL diagnostic QA are not full21454 test',
                          'cap1/cap3 full confirmation restarted separately after image-token mismatch; no full result yet']}
    (REPORT/'COMPLETED_CONTROLS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    out=['# 已完成的基底／预算与容量控制（2026-10-17）','',
         '只汇总完成证明及逐题文件SHA验证通过的结果。配对区间按原图像组bootstrap 10000次，seed42，未校正多重比较。',
         '', '## 基底与预算：9条轨迹、18个终点','',
         '同一目标任务按冻结TRAIN顺序做4次完整更新，终点1是同轨迹前缀。三个起点分别为随机A+零B、旧自然A+零B、旧自然A+B。前两者初始函数相同；旧自然训练347条的历史预算不免费。每轨迹保存4步原始梯度、损失、实际学习率、参数、具体内存前后与峰值。',
         '', '| 原生任务（TRAIN/TEST） | 起点 | 1更新EM | 4更新EM | 配对增量95%区间（百分点） |',
         '| --- | --- | ---: | ---: | --- |']
    for case in basis:
        name=case['case'];e=case['endpoints'];ci=contrasts[name+'_4_minus_1']['image_cluster_percentile_95']
        out.append(f"| {name.split('_')[0]} | {name[4:]} | {e[1]['correct']}/{e[1]['count']} | {e[4]['correct']}/{e[4]['count']} | [{ci[0]:.2f}, {ci[1]:.2f}] |")
    out+=['','T02医学Modality为11/228，T11医学KG为12/267，T12遥感rural为6/100。1更新与4更新不是独立训练重复。',
          '', '完整旧A+B明显区别于仅继承旧A：优势包含旧函数／输出习惯，不能归于A基底或专业知识。增加更新对医学两任务有帮助，但旧A+B在遥感由76/100降至44/100，反驳“训练越多必定越好”。单任务、单seed及格式混杂仍在。',
          '', '## 受限容量：只报告诊断集','', '| Kmax | 实际保留专家 | 生成EM | 相对cap1配对95%区间（百分点） |', '| ---: | ---: | ---: | --- |']
    for case in caps:
        cap=case['cap'];ci=cap_contrasts.get(str(cap),{}).get('image_cluster_percentile_95')
        out.append(f"| {cap} | {case['retained_experts']} | {case['EM_correct']}/155 | {f'[{ci[0]:.2f}, {ci[1]:.2f}]' if ci else '参考'} |")
    out+=['','四种容量都达到上限；不能宣称模型已学会低于上限自动停止。收益主要来自遥感，金融并未随容量改善。155题为19原生任务覆盖的诊断生成集；不能替代500题NLL或21454题全量评测，也不能跨Qwen基座比较排名。',
          '', '## 全量确认故障与修复','',
          '原cap1全量运行在FinMME:5082触发严格断言：2052个处理器image token与2106行实际视觉特征不匹配。原失败目录和结果保留且不计为完成。独立renderer只修正单图片连续占位token数量，原图像像素、问题、答案、预处理与参数不变。五域TRAIN首题及故障题的官方完整forward与缓存forward NLL差均为0；兼容题输入ID逐元素不变。',
          '', '新nohup服务`llmcl-onevision-full-repair.service`仅使用GPU1；预先指定cap1和cap3从头生成完整21454条，不拼接旧部分结果。报告500个已看诊断样本与其余20954题；最终评测不能推导完整FWT/BWT。修复不影响旧Qwen基线源码或活动进程。',
          '', '完整数值、哈希、训练和内存字节值见[机器可读证据](COMPLETED_CONTROLS.json)。']
    (REPORT/'COMPLETED_CONTROLS.md').write_text('\n'.join(out)+'\n')
    print(json.dumps({'basis_cases':len(basis),'capacity_cases':len(caps),'verified':True}))


if __name__=='__main__':main()
