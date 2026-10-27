"""Describe sealed S1 calibration-gradient concentration, never QA/model/F1."""
from pathlib import Path
import hashlib,json,math,statistics

META_SHA='8f58dc67f5a4eb2493a63177c345b242abfe3127e8f13c4eb3bdb0d21f77acdc'
BINDING_SHA='4313ef543860e0ef9919b24e1c6a29f5ee40caf49118add470587b9d101edfbb'
RUN='exp/task_gradient_formal_20261028/runs/onevision/pct01/TaskGradReuse/training_v1'
def sha(data):return hashlib.sha256(data).hexdigest()
def describe(rows):
 total=sum(x['total_squared_singular_energy'] for x in rows)
 top=sum(x['top_r_squared_singular_energy'] for x in rows)
 valid=[x for x in rows if x['total_squared_singular_energy']>0]
 values=[x['top_r_energy_fraction'] for x in valid]
 return dict(matrices=len(rows),zero_energy_matrices=len(rows)-len(valid),
  sum_total_squared_singular_energy=total,sum_top8_squared_singular_energy=top,
  energy_weighted_top8_fraction=top/total if total else None,
  matrix_median_top8_fraction=statistics.median(values) if values else None,
  matrix_mean_top8_fraction=statistics.fmean(values) if values else None,
  minimum_matrix=min(valid,key=lambda x:x['top_r_energy_fraction']) if valid else None,
  maximum_matrix=max(valid,key=lambda x:x['top_r_energy_fraction']) if valid else None)

def produce(root):
 root=Path(root)
 binding_bytes=(root/'summary/task_gradient_formal_20261028/BINDING.json').read_bytes()
 assert sha(binding_bytes)==BINDING_SHA
 binding=json.loads(binding_bytes)
 assert binding['method']=='TaskGradReuse' and binding['condition']=='pct01'
 assert binding['manifest_sha256']=='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
 meta_path=root/RUN/'method_state/task_gradient_stage_01.json';raw=meta_path.read_bytes()
 assert sha(raw)==META_SHA
 v=json.loads(raw);c=v['calibration']
 assert v['stage']==1 and v['preflight_only'] is False and v['main_optimizer_coordinates']==540672
 assert c['QA']==c['backwards']==16 and c['optimizer_updates']==0
 assert c['current_task_TRAIN_only'] and c['QA_equal_weight']
 rows=[]
 for x in v['direction']:
  name=x['module'];kind='q' if name.endswith('.q_proj') else 'v'
  assert name.endswith(('.q_proj','.v_proj'))
  assert x['gradient_shape']==([896,896] if kind=='q' else [128,896])
  assert x['svd_dtype']=='torch.float32' and x['svd_driver']=='default' and x['B_zero']
  s=x['singular_values_top_r'];assert len(s)==8 and all(math.isfinite(y) and y>=0 for y in s)
  total=x['total_squared_singular_energy'];top=x['top_r_squared_singular_energy'];fraction=x['top_r_energy_fraction']
  assert all(math.isfinite(y) and y>=0 for y in (total,top,fraction))
  assert top<=total*(1+1e-6) and fraction<=1+1e-6
  assert math.isclose(top,sum(y*y for y in s),rel_tol=1e-5,abs_tol=1e-20)
  assert math.isclose(fraction,top/total if total else 0.,rel_tol=1e-6,abs_tol=1e-8)
  rows.append(dict(module=name,kind=kind,total_squared_singular_energy=total,
    top_r_squared_singular_energy=top,top_r_energy_fraction=fraction))
 assert len(rows)==len({x['module'] for x in rows})==48
 assert all(sum(x['kind']==kind for x in rows)==24 for kind in ('q','v'))
 assert {x['module'] for x in rows}=={f'model.layers.{i}.self_attn.{kind}_proj' for i in range(24) for kind in ('q','v')}
 return dict(status='S1_TRAIN_MEAN_GRADIENT_DESCRIPTION_NOT_QUALITY',stage=1,calibration_QA=16,
  metadata_path=str(meta_path),metadata_sha256=META_SHA,binding_sha256=BINDING_SHA,
  scope='Frobenius energy of 48 mean dense q/v gradients from this 16-QA S1 calibration only',
  all=describe(rows),by_projection={k:describe([x for x in rows if x['kind']==k]) for k in ('q','v')},
  matrices=rows,new_models=0,new_scores=0,new_QA_reads=0,checkpoint_reads=0,
  interpretation_limits=['Not actual main-training/dropout gradients or Adam updates',
   'QA gradient cancellation and 16-QA sampling can affect concentration',
   'No ordinary Seq/random projection measurement or causal comparison',
   'Neither high nor low coverage establishes F1 capacity, generalization, or direction-method efficacy',
   'No threshold, rank selection, method change, or formal quality substitution'])

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 print(json.dumps(produce(p.parse_args().root),allow_nan=False))
