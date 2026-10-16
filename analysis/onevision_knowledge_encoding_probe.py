"""Read-only sanity: did auxiliary facts enter tiny LoRA / usable context?

Training-fact recall and fixed paraphrases only; NOT medical-test performance,
new benchmark tasks, or evidence of clinical knowledge generalization.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,sys,time,traceback
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from onevision_knowledge_intervention import Intervention,reset_seed,render_facts
from onevision_research_worker import digest
from onevision_capacity_probe import finish_resources
from onevision_trained_visual_probe import repair_tail
from run import freeze,write
REPORT=ROOT/'summary/onevision_diagnostics_20261016/capacity'


def fact_rows(facts,cohort):
    rows=[]
    for template in ['original','paraphrase']:
        for f in facts:
            if template=='original':
                question=f"What is the {f['relation']} of {f['subject']}?" if f['language']=='en' else f"{f['subject']}的{f['relation']}是什么？"
            else:
                question=f"For {f['subject']}, give its {f['relation']}." if f['language']=='en' else f"请直接给出{f['subject']}的{f['relation']}。"
            rows.append({'id':f"{cohort}:{template}:{f['id']}",'question':question,'images':[],
                         'answer':f['value'],'answers':[f['value']],'_task':11,'template':template,
                         'synthetic_mismatch_target':cohort=='own_permuted'})
    return rows


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1';torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    manifest,*_=freeze();mp=ROOT/'summary/onevision_mechanisms_20261016'
    original=json.loads((mp/'knowledge_complete.json').read_text());knowledge_run=ROOT/original['run']
    ref=json.loads((knowledge_run/'shared_adapter_reference.json').read_text());shared_path=ROOT/ref['path'];assert digest(shared_path)==ref['sha256']
    pool_path=ROOT/'data/OneVisionMechanisms/20261016/knowledge16.json';pool=json.loads(pool_path.read_text())
    for key,expected in original['source_sha256'].items():
        source=pool_path if key=='knowledge16.json' else ROOT/key
        assert digest(source)==expected, 'Knowledge checkpoint reconstruction source changed'
    files=[Path(__file__),ROOT/'analysis/onevision_knowledge_intervention.py',ROOT/'analysis/onevision_capacity_probe.py',ROOT/'exp/onevision_diagnostics_20261016/run.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    spec={'source_sha256':hashes,'knowledge_completion_sha256':digest(mp/'knowledge_complete.json'),
          'knowledge_pool_sha256':digest(pool_path),'shared_sha256':ref['sha256'],'optimizer_updates':0,
          'scope':'auxiliary16-fact training-target recall and fixed paraphrases; no target medical test answers',
          'prompt':'same predeclared both as original knowledge intervention; no prompt selection',
          'synthetic_mismatch':'own_permuted targets intentionally wrong; measure memorization, never factual correctness'}
    pointer=REPORT/'knowledge_sanity_attempt.json';complete=REPORT/'knowledge_sanity_complete.json'
    if pointer.exists():
        info=json.loads(pointer.read_text());assert info['spec']==spec;run=ROOT/info['run']
    else:
        run=ROOT/'exp/onevision_knowledge_encoding_20261016'/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
        write(pointer,{'run':str(run.relative_to(ROOT)),'spec':spec});write(run/'protocol.json',spec)
    if complete.exists():
        proof=json.loads(complete.read_text());assert proof['source_sha256']==hashes
        for relative,expected in proof['artifact_sha256'].items():assert digest(run/relative)==expected
        return
    repair_tail(run/'resources.jsonl');started=time.time();x=Intervention(manifest,run);x.process_started=started
    x.boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip();x.prompt('both')
    shared=torch.load(shared_path,map_location='cpu',weights_only=True)['adapter']
    x.monitor.snapshot('encoding_inference_before',model=x.model);summaries=[]

    def evaluate(case,cohort,rows):
        path=run/'evaluations'/case/(cohort+'_text.jsonl');path.parent.mkdir(parents=True,exist_ok=True);repair_tail(path)
        records=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        assert [r['id'] for r in records]==[r['id'] for r in rows[:len(records)]]
        x.model.eval()
        with path.open('a') as stream:
            for row in rows[len(records):]:
                with torch.no_grad():loss,details=x.loss(row,view='text');generated=x.generate(row,view='text')
                assert torch.isfinite(loss) and details['visual_tokens']==0
                record={'id':row['id'],'template':row['template'],'nll':float(loss),'prediction':generated['prediction'],
                        'reference':row['answer'],'scores':generated['scores'],'synthetic_mismatch_target':row['synthetic_mismatch_target']}
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();records.append(record)
            os.fsync(stream.fileno())
        for template in ['original','paraphrase']:
            part=[r for r in records if r['template']==template];assert len(part)==16
            summaries.append({'case':case,'cohort':cohort,'template':template,'count':16,
                'mean_nll':sum(r['nll'] for r in part)/16,'normalized_exact_match':sum(r['scores']['exact_match'] for r in part)/16,
                'reference_semantics':'synthetic training targets, NOT true facts' if cohort=='own_permuted' else 'author graph facts'})

    relevant=fact_rows(pool['correct'],'relevant_true')
    for name,base in [('base',x.original_initial),('shared',shared)]:
        for kind in ['none','correct','unrelated','permuted']:
            x.configure(base);x.knowledge=None if kind=='none' else render_facts(pool[kind]);x.model.eval()
            evaluate(f'context_{name}_{kind}','relevant_true',relevant)
    for rank in [1,2,8]:
        for kind in ['correct','unrelated','permuted']:
            case=f'residual_rank{rank}_{kind}' if rank<8 else f'continue_rank8_{kind}'
            path=knowledge_run/'adapters'/(case+'.pt');meta=json.loads(path.with_suffix('.json').read_text());assert digest(path)==meta['sha256']
            x.configure(shared,rank if rank<8 else None);x.restore(torch.load(path,map_location='cpu',weights_only=True)['adapter']);x.knowledge=None;x.model.eval()
            evaluate(case,'relevant_true',relevant)
            if kind!='correct':evaluate(case,'own_'+kind,fact_rows(pool[kind],'own_'+kind))
    x.monitor.snapshot('encoding_inference_after',model=x.model)
    finish_resources(x,run,'encoding_inference_before','encoding_inference_after','read-only auxiliary fact recall; zero optimizer updates')
    write(run/'encoding_summary.json',{'rows':summaries,'target_medical_QA_tested':0,'optimizer_updates':0,
          'limitations':['Original templates are training targets; paraphrases retain same16 facts, not independent knowledge generalization.',
                         'Synthetic mismatch scores measure writing wrong targets, not factual correctness.',
                         'Reconstruction uses deterministic inference; original auxiliary training used its original backend.']})
    artifacts=list((run/'evaluations').rglob('*.jsonl'))+[run/'encoding_summary.json',run/'memory_summary.json']
    write(complete,{'complete':True,'run':str(run.relative_to(ROOT)),'source_sha256':hashes,
          'artifact_sha256':{str(p.relative_to(run)):digest(p) for p in artifacts},'medical_performance_claim':False})


if __name__=='__main__':
    try:main()
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True);write(REPORT/'knowledge_sanity_needs_attention.json',{'traceback':traceback.format_exc(),'optimizer_updates':0})
