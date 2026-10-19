"""Publish an explicit SHA-bound derivative of the verified resumed branch.

Original run/events/resources remain untouched. All prediction/checkpoint bytes
are identical to the coherent current branch; discarded work remains recorded.
No GPU, training, generation, test selection or runner modification.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import v2_verified_baseline_report as verifier

ROOT=verifier.ROOT
SOURCE=ROOT/'exp/two_backbone_native_cl_20261017_v2/runs/qwen/pct01/SeqFT_LoRA/training_20261017_161209'
DEST=SOURCE.parent/'training_20261019_verified_resume_branch'
REPORT=ROOT/'summary/two_backbone_pct01_20261017/qwen_seqft_io_recovery_20261019'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)


def main():
    assert not DEST.exists() and not (REPORT/'BRANCH_PUBLICATION_COMPLETE.json').exists()
    lineage=read(REPORT/'LINEAGE_VERIFIED.json')
    assert lineage['verified'] and lineage['stage18_fresh_evaluations_after_restart']
    assert lineage['validity_chain_continuous_all19'] and lineage['discarded_updates']==7
    source_rel=str(SOURCE.relative_to(ROOT))
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:assert source_rel.encode() not in (p/'cmdline').read_bytes(), 'Source worker still live'
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    assert sha(SOURCE/'logs/events.jsonl')==lineage['original_events_sha256']
    assert sha(SOURCE/'resources.jsonl')==lineage['original_resources_sha256']
    before_status=sha(SOURCE/'run_status.json')
    events_path=SOURCE/'logs/events.jsonl';lines=events_path.read_bytes().splitlines(keepends=True)
    drop=set(lineage['interrupted_event_indices']);assert len(drop)==7
    dropped=[json.loads(lines[i]) for i in sorted(drop)]
    assert all(e['event']=='train_update' and e['stage']==18 for e in dropped)
    assert [e['step'] for e in dropped]==list(range(1,8))
    kept=[line for i,line in enumerate(lines) if i not in drop]
    updates=[json.loads(line) for line in kept if json.loads(line).get('event')=='train_update']
    assert len(updates)==93
    latest=read(SOURCE/'checkpoints/latest.json')
    assert latest['stage']==19
    DEST.mkdir()
    copy_records=[]
    # Only immutable scientific artifacts; no old status, malformed resource log,
    # interrupted predictions or supervisor/launcher metadata are transplanted.
    folders=['evaluations','stages','validity','runner_source','checkpoints']
    for folder in folders:
        for src in sorted((SOURCE/folder).rglob('*')):
            if not src.is_file():continue
            assert not src.is_symlink()
            rel=src.relative_to(SOURCE);dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            if src.suffix=='.pt':
                # Trusted checkpoints are immutable. Hard links retain their
                # original byte identity without duplicating ~85 GB of weights.
                os.link(src,dst)
            else:shutil.copy2(src,dst)
            copy_records.append(dict(path=str(rel),bytes=src.stat().st_size,sha256=sha(src),
                                     identical_destination=sha(src)==sha(dst)))
    for name in ['config.json','metrics_by_score.json','domain_results.json','evaluation_coverage.json',
                 'score_matrices.json','token_f1_sparse.csv','rougeL_sparse.csv','exact_match_sparse.csv']:
        src=SOURCE/name;shutil.copy2(src,DEST/name)
        copy_records.append(dict(path=name,bytes=src.stat().st_size,sha256=sha(src),identical_destination=sha(src)==sha(DEST/name)))
    assert all(x['identical_destination'] for x in copy_records)
    (DEST/'logs').mkdir(exist_ok=True);(DEST/'logs/events.jsonl').write_bytes(b''.join(kept))
    (DEST/'logs/interrupted_stage18_events.jsonl').write_bytes(b''.join(lines[i] for i in sorted(drop)))
    # The source full JSONL is retained outside the canonical event stream.
    shutil.copy2(events_path,DEST/'logs/original_all_attempts_events.jsonl')
    manifest=read(ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json')
    extra_qa=manifest['tasks'][17]['files']['train']['count']
    groups={};bad=[]
    for i,line in enumerate((SOURCE/'resources.jsonl').read_bytes().splitlines(),1):
        try:row=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError):bad.append(i);continue
        groups.setdefault(str(row.get('pid','not_recorded')),[]).append(row)
    partial={}
    for key,rows in groups.items():
        fields=sorted({k for row in rows for k,v in row.items() if isinstance(v,(int,float)) and k.endswith('_bytes')})
        partial[key]=dict(valid_samples=len(rows),observed_maxima={k:max(row[k] for row in rows if isinstance(row.get(k),(int,float))) for k in fields},
                         full_process_peak_proven=False)
    memory=dict(unit='bytes',resource_log_incomplete=True,invalid_source_line_numbers=bad,
        before_after_growth=None,observations_by_PID=partial,original_resources_sha256=sha(SOURCE/'resources.jsonl'),
        note='Original samples retained at source. No missing values reconstructed, no cross-PID subtraction, no memory advantage asserted.')
    write(DEST/'memory_summary.json',memory)
    provenance=dict(source_run=source_rel,lineage_proof_sha256=sha(REPORT/'LINEAGE_VERIFIED.json'),
        no_training_or_generation_or_TEST_selection=True,all_prediction_and_checkpoint_bytes_unchanged=True,
        exact_copied_files=copy_records,original_events_sha256=sha(events_path),
        canonical_events_sha256=sha(DEST/'logs/events.jsonl'),discarded_indices=sorted(drop),
        discarded_train_updates=7,discarded_train_QA=extra_qa,extra_interrupted_evaluation_work_retained_at_source=True,
        final_lineage_positive_updates=93,total_actual_train_updates_including_discarded=100,
        resource_log_incomplete=True,original_source_status_sha256=before_status)
    write(DEST/'recovery_provenance.json',provenance)
    pending=dict(status='CPU_verifying_recovered_branch',completed_stage=19)
    write(DEST/'run_status.json',pending)
    original_eligible=verifier.eligible;verifier.eligible=lambda s:s==pending
    try:pre=verifier.verify(str(DEST.relative_to(ROOT)),'new3090')
    finally:verifier.eligible=original_eligible
    assert pre['rescored_cells']==74 and pre['final_QA']==21454 and pre['positive_finite_training_stages']==19
    write(REPORT/'BRANCH_PREPUBLICATION_VERIFICATION.json',pre)
    write(DEST/'run_status.json',dict(status='completed',completed_stage=19,final_checkpoint=latest['path'],
        final_parameter_file_bytes=latest['bytes'],recovered_from=source_rel,
        recovery_provenance_sha256=sha(DEST/'recovery_provenance.json'),resource_log_incomplete=True,
        discarded_compute_included_in_provenance=True,no_training_or_prediction_repeated=True))
    final=verifier.verify(str(DEST.relative_to(ROOT)),'new3090')
    assert final['aggregate']==pre['aggregate'] and final['prediction_sha256']==pre['prediction_sha256']
    write(REPORT/'BRANCH_FINAL_VERIFICATION.json',final)
    assert sha(SOURCE/'run_status.json')==before_status
    assert sha(events_path)==lineage['original_events_sha256']
    assert sha(SOURCE/'resources.jsonl')==lineage['original_resources_sha256']
    write(REPORT/'BRANCH_PUBLICATION_COMPLETE.json',dict(complete=True,CPU_only=True,run=str(DEST.relative_to(ROOT)),
        final_proof_sha256=sha(REPORT/'BRANCH_FINAL_VERIFICATION.json'),
        provenance_sha256=sha(DEST/'recovery_provenance.json'),original_all_artifacts_unchanged=True,
        aggregate=final['aggregate'],rescored_cells=74,final_QA=21454,resource_log_incomplete=True,
        no_GPU_or_training_or_generation_started=True))
    print(json.dumps(dict(complete=True,run=str(DEST.relative_to(ROOT)),aggregate=final['aggregate'],
                         resource_log_incomplete=True,original_artifacts_unchanged=True)),flush=True)


if __name__=='__main__':main()
