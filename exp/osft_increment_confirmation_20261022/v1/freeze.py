"""One pre-attempt metadata/source freeze, with no TEST rows or model loaded."""
import ast
import json
from pathlib import Path
from common import ROOT,PORT,REPORT,SCREEN,study,sha,write,original_data,ANCHORS

def main():
    assert not (REPORT/'READY.json').exists()
    review=json.loads((REPORT/'REVIEW.json').read_text());assert review['decision']=='accept'
    for p,h in review['files_sha256'].items():assert sha(ROOT/p)==h,p
    old=json.loads((SCREEN/'READY.json').read_text())
    for p,h in old['sources'].items():assert sha(ROOT/p)==h,p
    assert sha(SCREEN/'READY.json')=='75b86e6272295b548c82cb73fd29dc60671e18b23314b625d46486232b0d1893'
    assert sha(SCREEN/'SCREEN_COMPLETE.json')=='1f56aa877ccecf7f93eb61c8520df9bdd81137939b5d669ca66851d518363174'
    assert json.loads((SCREEN/'SCREEN_COMPLETE.json').read_text())['possible_full_retrain_candidates']==['LoRA_all']
    path,m,data,plan,lockpath=original_data()
    ledger=json.loads((SCREEN/'SPLIT.json').read_text())
    images={im for rows in data.values() for row in rows for im in row['images']}
    assert images==set(ledger['image_sha256'])
    for im,h in ledger['image_sha256'].items():assert sha(ROOT/im)==h
    for p in Path(__file__).parent.glob('*.py'):compile(p.read_text(),str(p),'exec')
    assert study.training_ast()[1]==old['train_AST_sha256']
    assert len(plan)==19 and sum(p['QA'] for p in plan)==1313 and sum(p['updates'] for p in plan)==93
    assert sum(t['files']['test']['count'] for t in m['tasks'])==21454
    assert set(t['domain'] for t in m['tasks'])==set(ANCHORS['OSFT']['domains'])
    sources=dict(old['sources'])
    sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')})
    sources['analysis/onevision_research_device_entry.py']=sha(ROOT/'analysis/onevision_research_device_entry.py')
    evidence_paths=[REPORT/'PROTOCOL.md',REPORT/'REVIEW.json',SCREEN/'READY.json',SCREEN/'SPLIT.json',SCREEN/'SCREEN_COMPLETE.json',
        SCREEN/'FULL_CONFIRMATION_DECISION.md',path,lockpath,path.parent/'frozen_ids.json',
        ROOT/'summary/two_backbone_pct01_20261017/pass_moe_verified_20261021/PUBLICATION.json',
        ROOT/'summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/completed/5193ee4c6861655d85e8.json']
    base=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    bases={p.relative_to(ROOT).as_posix():sha(p) for p in base.iterdir() if p.is_file() and p.suffix in ['.json','.txt','.safetensors','.model']}
    for p,h in old['base_files'].items():assert bases[p]==h
    write(REPORT/'ADMISSION.json',dict(original_TRAIN_prefix_AST_exact=True,training_plan=plan,TRAIN_QA=1313,updates=93,
        TEST_QA_declared=21454,TEST_rows_opened=0,no_new_model_forward=True,
        existing_actual_model_gates_and_pilot_reused=True,no_tiny_model_repeat=True))
    evidence_paths.append(REPORT/'ADMISSION.json')
    write(REPORT/'READY.json',dict(sources=sources,evidence={p.relative_to(ROOT).as_posix():sha(p) for p in evidence_paths},
        base_files=bases,TRAIN_image_sha256=ledger['image_sha256'],manifest_sha256=sha(path),
        train_AST_sha256=study.training_ast()[1],training_plan=plan,anchors=ANCHORS,
        primary='five_domain_QA_weighted_within_domain_then_equal_macro',primary_threshold=46.619976395158304,
        TRAIN_QA=1313,stages=19,updates=93,TEST_QA=21454,scientific_sources_FROZEN=True,
        coverage='final_only_19_tasks_not_74_cells',one_fresh_training_one_final_generation=True))
    print(json.dumps(dict(ready_sha256=sha(REPORT/'READY.json'),TRAIN_QA=1313,updates=93,TEST_rows_opened=0)))
if __name__=='__main__':main()
