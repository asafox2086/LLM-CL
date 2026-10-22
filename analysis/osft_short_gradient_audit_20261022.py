"""Read existing scalar gradient records once; no torch, model, QA or optimizer."""
from pathlib import Path
import json,hashlib,math,statistics,time,ast

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/osft_short_gradient_audit_20261022/v1'
SCREEN=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1'
SHAPE=ROOT/'summary/edge_deployment_20261019/update_geometry_literature_20261022/PROTECTED_BLOCK_COUNTS.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summary(values):return dict(min=min(values),median=statistics.median(values),max=max(values),mean=statistics.mean(values))
def main():
    assert not (REPORT/'EVIDENCE.json').exists()
    begin=time.monotonic();complete=json.loads((SCREEN/'OSFT_all_COMPLETE.json').read_text())
    ready=json.loads((SCREEN/'READY.json').read_text())
    assert complete['complete'] and complete['TEST_QA']==0 and complete['updates']==87
    assert complete['ready_sha256']==sha(SCREEN/'READY.json')
    assert complete['source_sha256']==sha(ROOT/'exp/osft_increment_screen_20261019/v1/run.py')
    for n,h in ready['sources'].items():assert sha(ROOT/n)==h,n
    work=ROOT/complete['checkpoint_root'];ep=work/'logs/events.jsonl'
    events=[json.loads(x) for x in ep.read_text().splitlines()]
    assert len(events)==87
    shapes=json.loads(SHAPE.read_text())['by_position'];records=[]
    files={str(ep.relative_to(ROOT)):sha(ep)};stage_counts={}
    for e in events:
        k,j=e['stage'],e['step'];stage_counts[k]=stage_counts.get(k,0)+1
        assert j==stage_counts[k] and e['gradient_clip_threshold']==1.
        assert all(math.isfinite(v) and v>0 for v in e['learning_rate_used'])
        p=work/'gradients'/f'S{k:02d}_{j:03d}.json';raw=json.loads(p.read_text())['raw']
        assert len(raw)==504
        energy=dict(short=0.,long=0.,S=0.);per_position={n:dict(short=0.,long=0.,S=0.) for n in shapes}
        modules={}
        for n,v in raw.items():
            module,pos,factor=n.rsplit('.',2);x=shapes[pos]
            assert factor in ['U_low','S_low','V_low'] and v['finite'] and math.isfinite(v['norm']) and v['norm']>=0
            short='V_low' if x['m']>=x['n'] else 'U_low'
            label='S' if factor=='S_low' else 'short' if factor==short else 'long'
            sq=v['norm']**2;energy[label]+=sq;per_position[pos][label]+=sq
            modules.setdefault(module+'.'+pos,set()).add(factor)
        assert len(modules)==168 and all(v=={'U_low','S_low','V_low'} for v in modules.values())
        total=sum(energy.values());retained=energy['long']+energy['S'];assert total>0 and retained>0
        norm=math.sqrt(total);old=e['gradient_l2_preclip']
        error=abs(norm-old)/max(norm,old,1e-30);assert error<=1e-5
        full=min(1.,1./(norm+1e-6));keep=min(1.,1./(math.sqrt(retained)+1e-6))
        assert math.isfinite(e['gradient_l2_postclip']) and 0<=e['gradient_l2_postclip']<=1.00001
        records.append(dict(stage=k,step=j,energy=energy,per_position_energy=per_position,
            short_squared_norm_fraction=energy['short']/total,S_squared_norm_fraction=energy['S']/total,
            reconstructed_global_norm=norm,event_global_norm=old,relative_norm_discrepancy=error,
            original_clip_coefficient_from_stored_norms=full,static_deleted_short_clip_coefficient=keep,
            static_retained_gradient_multiplier_ratio=keep/full,actual_ShortFixed_trajectory=False))
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    assert len(stage_counts)==19
    for k,count in stage_counts.items():
        p=work/'validity'/f'stage_{k:02d}.json';v=json.loads(p.read_text())
        assert v['passed'] and v['actual_parameters_changed'] and v['positive_lr_updates']==v['updates']==count
        files[p.relative_to(ROOT).as_posix()]=sha(p)
    rows=[r for r in records if r['step']==1];assert len(rows)==19
    out=dict(complete=True,kind='current_snapshot_of_original_sealed_run_scalar_gradient_records_not_new_training',
        steps=87,stages=19,all_original_steps_clipped=sum(r['original_clip_coefficient_from_stored_norms']<1 for r in records),
        short_energy_fraction=summary([r['short_squared_norm_fraction'] for r in records]),
        S_energy_fraction=summary([r['S_squared_norm_fraction'] for r in records]),
        static_multiplier_ratio=summary([r['static_retained_gradient_multiplier_ratio'] for r in records]),
        first_steps_static_multiplier_ratio=summary([r['static_retained_gradient_multiplier_ratio'] for r in rows]),
        first_steps_short_energy_fraction=summary([r['short_squared_norm_fraction'] for r in rows]),
        max_relative_norm_discrepancy=max(r['relative_norm_discrepancy'] for r in records),
        per_step=records,scalar_record_sha256=files,
        provenance={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),SHAPE,SCREEN/'READY.json',SCREEN/'OSFT_all_COMPLETE.json',SCREEN/'SCREEN_COMPLETE.json',REPORT/'QUESTION.md']},
        historical_raw_and_event_hashes_in_original_COMPLETE=False,
        raw_hashes_are_new_snapshot_not_retroactive_seal=True,
        no_model_tensor_QA_TEST_generation_optimizer_or_GPU=True,scientific_conditions_not_changed=True,
        seconds=time.monotonic()-begin)
    (REPORT/'EVIDENCE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['per_step','scalar_record_sha256','provenance']}))
if __name__=='__main__':main()
