from pathlib import Path
import json,hashlib,subprocess
R=Path('/data1/liyapeng/program/LLMCL')
P=R/'summary/edge_deployment_20261019/osft_short_factor_control_20261022/v2'
G=R/'exp/osft_short_factor_control_20261022/v2/private/NATIVE_GATE.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
complete=json.loads((P/'COMPLETE.json').read_text())
assert complete['complete'] and sha(G)==complete['native_gate_sha256']
gate=json.loads(G.read_text());counts=[]
def visit(x):
 if isinstance(x,dict):
  if 'image_tokens' in x:counts.append(int(x['image_tokens']))
  for y in x.values():visit(y)
 elif isinstance(x,list):
  for y in x:visit(y)
visit(gate)
config=json.loads((R/'model/LLaVA-OneVision-Qwen2-0.5B-OV/config.json').read_text())
h=int(config['text_config']['hidden_size']);assert h==896
paths=['exp/two_backbone_native_cl_20261017_v2/onevision_bridge.py','exp/two_backbone_native_cl_20261017_v2/frozen_vision.py','exp/two_backbone_native_cl_20261017_v2/runtime_utils.py']
print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
 source_sha256={n:sha(R/n) for n in paths},complete_sha256=sha(P/'COMPLETE.json'),gate_sha256=sha(G),
 original_gate_scalar_count=len(counts),image_tokens=counts,hidden_size=h,
 packed_FP16_feature_payload_bytes=[n*h*2 for n in counts],
 not_all_FIT_or_replay_bank=True,not_peak_RAM_or_final_parameters=True,
 no_QA_or_image_or_weight_tensor_read=True,no_model_or_GPU_or_old_score_recompute=True)))
