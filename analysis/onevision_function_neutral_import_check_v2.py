"""Check real worker imports without initializing CUDA; detect the v1 bug."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import math
import torch
from onevision_function_neutral_gradients import gradient_values

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    outcomes=[]
    for module,device in [('onevision_function_neutral_init_trial','1'),
                          ('onevision_function_neutral_init_trial_v2','0'),
                          ('onevision_function_neutral_init_trial_v2','1')]:
        program=("import sys,os,json;sys.path.insert(0,'analysis');before=os.environ['CUDA_VISIBLE_DEVICES'];"
                 f"import {module};import torch;"
                 "print(json.dumps(dict(before=before,after=os.environ['CUDA_VISIBLE_DEVICES'],CUDA_initialized=torch.cuda.is_initialized())))")
        result=subprocess.run([sys.executable,'-c',program],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='1'),
            capture_output=True,text=True,check=True)
        actual=json.loads(result.stdout.strip());actual['module']=module
        assert not actual['CUDA_initialized']
        if module.endswith('_v2'):assert actual['after']==actual['before']==device
        else:assert actual['before']=='1' and actual['after']=='','Original failure positive control not detected'
        outcomes.append(actual)
    path=ROOT/'analysis/onevision_initialization_source_geometry.py'
    text=path.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='gradient_values')
    extracted=ast.Module(body=[node],type_ignores=[]);space=dict(torch=torch,math=math)
    exec(compile(extracted,str(path),'exec'),space)
    helper=ROOT/'analysis/onevision_function_neutral_gradients.py'
    new_node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='gradient_values')
    assert ast.dump(node,include_attributes=False)==ast.dump(new_node,include_attributes=False)
    old_tree=ast.parse((ROOT/'analysis/onevision_function_neutral_init_trial.py').read_text())
    new_tree=ast.parse((ROOT/'analysis/onevision_function_neutral_init_trial_v2.py').read_text())
    def functions(tree):
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
                if isinstance(n,ast.FunctionDef) and n.name!='plan'}
    assert functions(old_tree)==functions(new_tree),'Scientific worker bodies must stay exact'
    p=ROOT/'summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1/cold_complete.json'
    proof=json.loads(p.read_text());run=ROOT/proof['run'];comparisons=[]
    for step in [1,2]:
        relative=f'raw_update_gradients/target_T04/step{step}.pt';f=run/relative
        assert sha(f)==proof['artifact_sha256'][relative]
        raw=torch.load(f,map_location='cpu',weights_only=True)
        assert gradient_values(raw)==space['gradient_values'](raw)
        comparisons.append(dict(step=step,sha256=sha(f),exact_profile=True))
    result=dict(CPU_only=True,actual_import_outcomes=outcomes,old_bug_positive_control=True,
        v2_devices_0_and_1_preserved=True,CUDA_initialized=False,helper_AST_exact_original=True,
        nonplan_scientific_worker_functions_AST_exact_v1=True,
        actual_original_cold_gradient_profiles_exact=comparisons,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),path,helper,
            ROOT/'analysis/onevision_function_neutral_init_trial_v2.py',
            ROOT/'analysis/onevision_function_neutral_init_entry_v2.py']},GPU_started=False)
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'DEVICE_IMPORT_GATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(CPU_only=True,old_bug_detected=True,v2_GPU_env_preserved=True,
        actual_gradient_profiles_exact=2,CUDA_initialized=False)))


if __name__=='__main__':main()
