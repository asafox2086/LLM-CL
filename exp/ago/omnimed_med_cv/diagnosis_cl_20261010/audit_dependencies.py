import hashlib,json
from pathlib import Path
exp=Path(__file__).resolve().parent;root=exp.parents[2]
paths=[root/'exp'/name for name in ['common.py','cv_data.py','cv_runtime.py','cv_reflection_med.py','run_olora.py','recovery.py']]
paths += [root/'code'/f'{method}.py' for method in ['seq_lora','migu_lora','olora','sapt_lora']]
paths += [root/'exp/configs'/f'{method}_qwen2vl_med.json' for method in ['seq_lora','migu_lora','olora','sapt_lora']]
paths += [root/'model/Qwen2-VL-2B-Instruct/config.json']
records=[dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
(exp/'dependency_audit.json').write_text(json.dumps(records,indent=2))
index=root/'summary/medical_research_comparison.md'
link='[三类诊断持续学习对比：MedQwen、RA-LDL 与统一 Qwen 基线](omnimed_med_cv/diagnosis_cl_20261010/README.md)'
text=index.read_text()
if link not in text:
    lines=text.splitlines();lines.insert(2,link+'。本实验的报告、设置和结果独立保存。\n');index.write_text('\n'.join(lines)+'\n')
print('Dependency hashes and summary index saved.')
