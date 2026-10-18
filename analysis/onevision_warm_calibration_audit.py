"""Warm-head TRAIN exclusion audit, reusing the exact frozen cold audit.

Only four path literals change. No TEST rows, model forward, GPU, deployment,
or parameter selection; old cold evidence is reused without fitting it again.
"""
import ast,copy,hashlib,json,os
from pathlib import Path
import torch
import onevision_functional_calibration_audit as old

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'summary/onevision_diagnostics_20261016'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    target=BASE/'warm_quality_20261018/calibration_audit'
    assert not (target/'EVIDENCE.json').exists(),'Do not repeat completed TRAIN evidence'
    cold_evidence_path=BASE/'functional_router/calibration_audit/EVIDENCE.json'
    cold=json.loads(cold_evidence_path.read_text())
    source=ROOT/'analysis/onevision_functional_calibration_audit.py'
    assert sha(source)==cold['source_sha256'][str(source.relative_to(ROOT))]
    definition=copy.deepcopy(next(n for n in ast.parse(source.read_text()).body
                                if isinstance(n,ast.FunctionDef) and n.name=='main'))
    replacements={
      'functional_router/pilot_attempt.json':'warm_quality_20261018/pilot_attempt.json',
      'exp/onevision_functional_router_20261018/pilot':'exp/onevision_warm_quality_20261018/pilot',
      'learned_reuse/learned_cosine_K3_train_complete.json':'warm_start_20261018/warm_inherit_K3_v1_train_complete.json',
      'functional_router/calibration_audit':'warm_quality_20261018/calibration_audit'}
    counts={k:0 for k in replacements}
    class Paths(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in replacements:
                counts[node.value]+=1
                return ast.copy_location(ast.Constant(replacements[node.value]),node)
            return node
    definition=Paths().visit(definition)
    assert all(v==1 for v in counts.values()),counts
    namespace=dict(vars(old));namespace['__file__']=str(Path(__file__))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[])),str(source),'exec'),namespace)
    namespace['main']()
    output=target/'EVIDENCE.json';result=json.loads(output.read_text())
    pointers={v:json.loads((BASE/f'{v}/pilot_attempt.json').read_text()) for v in ['functional_router','warm_quality_20261018']}
    payload={}
    calibration_sha={}
    for name,p in pointers.items():
        path=ROOT/p['run']/'calibration.pt'
        summary=json.loads((path.parent/'calibration_summary.json').read_text())
        assert sha(path)==summary['calibration_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['TEST_used'] is False and saved['spec']==p['spec']
        payload[name]=saved;calibration_sha[name]=sha(path)
    a,b=payload['functional_router'],payload['warm_quality_20261018']
    assert a['TRAIN_ids']==b['TRAIN_ids'] and torch.equal(a['cosines'],b['cosines'])
    assert cold['TRAIN_QA']==result['TRAIN_QA']==43
    matched={'same43TRAIN_ids_and_order':True,'input_cosines_bitwise_exact':True,
             'base_e0_NLL_bitwise_exact':torch.equal(a['TRAIN_NLL'][:,:2],b['TRAIN_NLL'][:,:2]),
             'functional_minimum_NLL_class_changes':int((a['TRAIN_NLL'].argmin(1)!=b['TRAIN_NLL'].argmin(1)).sum()),
             'quality_head_route_changes_cold_to_warm':sum(x['fit_class']!=y['fit_class'] for x,y in zip(cold['paired_TRAIN_rows'],result['paired_TRAIN_rows'])),
             'calibration_sha256':calibration_sha,'cold_completed_audit_sha256':sha(cold_evidence_path),
             'original_audit_source_sha256':sha(source),'path_literal_replacements':replacements,
             'cold_refits_repeated':False,'warm_refits':result['analysis_CPU_refits'],
             'limit':'Identical geometry does not encode changes of expert functional quality. TRAIN NLL ranks are not TEST F1 or proof of knowledge sufficiency.'}
    result['matched_cold_warm_TRAIN_geometry']=matched
    result['source_sha256'][str(source.relative_to(ROOT))]=sha(source)
    output.write_text(json.dumps(result,indent=2)+'\n')
    doc=target/'VERIFIED_RESULTS.md';s=doc.read_text()
    s+='\n## 与已完成冷池的配对TRAIN对照\n\n'
    s+=f"43条TRAIN及全部输入余弦逐元素相同；四候选最小NLL标签改变{matched['functional_minimum_NLL_class_changes']}/43，质量头调用改变{matched['quality_head_route_changes_cold_to_warm']}/43。base/e0 NLL精确相同：{matched['base_e0_NLL_bitwise_exact']}。冷池既有审计未重跑，额外CPU拟合仅来自本温池。\n\n"
    s+='这验证本TRAIN缓存上输入几何与专家功能状态可以分离；排除同图仅检查质量头拟合稳定性，专家/投影/记忆仍看过TRAIN，不是整个流程的未见测试，也不能把TRAIN最低NLL当最佳生成F1或部署oracle。所有原GPU头/路由/阈值/队列保持。\n'
    doc.write_text(s)
    print(json.dumps({'matched_TRAIN_geometry':matched,'warm_fit':result['quality_hard_fit_same_TRAIN'],
                      'warm_excluded':result['quality_hard_leave_image_group_out_head_only']}))

if __name__=='__main__':main()
