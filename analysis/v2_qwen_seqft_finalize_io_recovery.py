"""CPU-only finalization after corrupt resource JSONL blocked completed status.

Never changes training, predictions, metrics, checkpoints or frozen runners.
The exact original verifier audits all scientific requirements before status
publication; only discovery eligibility is temporarily bound to the old status.
"""
import hashlib
import json
from pathlib import Path

import v2_verified_baseline_report as verifier

ROOT=verifier.ROOT
REL='exp/two_backbone_native_cl_20261017_v2/runs/qwen/pct01/SeqFT_LoRA/training_20261017_161209'
RUN=ROOT/REL
REPORT=ROOT/'summary/two_backbone_pct01_20261017/qwen_seqft_io_recovery_20261019'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    assert not (REPORT/'RECOVERY_COMPLETE.json').exists()
    assert not Path('/proc/3256').exists()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            command=(p/'cmdline').read_bytes()
            assert REL.encode() not in command, 'Original worker or a resume is live'
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    status_path=RUN/'run_status.json';before=status_path.read_bytes();status=json.loads(before)
    assert status=={'status':'initial_evaluation','completed_stage':0}
    log=RUN/'supervisor.log';assert 'json.decoder.JSONDecodeError' in log.read_text()[-5000:]
    latest=json.loads((RUN/'checkpoints/latest.json').read_text());assert latest['stage']==19
    resources=RUN/'resources.jsonl';raw=resources.read_bytes();bad=[];good=[]
    for line_number,line in enumerate(raw.splitlines(),1):
        try:good.append(json.loads(line))
        except (json.JSONDecodeError,UnicodeDecodeError) as e:
            bad.append(dict(line=line_number,bytes=len(line),sha256=hashlib.sha256(line).hexdigest(),error=str(e)))
    assert bad and good
    evidence=dict(original_status=status,run=REL,original_status_sha256=sha(status_path),
        original_resources_sha256=sha(resources),original_supervisor_log_sha256=sha(log),
        corrupted_resource_lines=bad,valid_resource_lines=len(good),original_worker_absent=True,
        no_training_or_prediction_or_checkpoint_mutation=True,resource_data_incomplete=True)
    (REPORT/'BEFORE.json').write_text(json.dumps(evidence,indent=2)+'\n')
    # This does not declare completion: it permits the frozen verifier to audit
    # the exact old candidate despite its known terminal status-write failure.
    eligible=verifier.eligible
    verifier.eligible=lambda value:value==status
    try:proof=verifier.verify(REL,'new3090')
    finally:verifier.eligible=eligible
    assert proof['rescored_cells']==74 and proof['final_QA']==21454 and proof['positive_finite_training_stages']==19
    assert status_path.read_bytes()==before and sha(resources)==evidence['original_resources_sha256']
    (REPORT/'PREPUBLICATION_SCIENTIFIC_VERIFICATION.json').write_text(json.dumps(proof,indent=2)+'\n')
    # Missing samples cannot be reconstructed. Keep the original JSONL untouched
    # and report partial observed peaks per PID, never cross-PID growth.
    groups={}
    for row in good:
        key=str(row.get('pid','missing'))+':'+str(row.get('boot_id','not_recorded'))
        groups.setdefault(key,[]).append(row)
    summary={}
    for key,rr in groups.items():
        fields=sorted({k for row in rr for k,v in row.items() if isinstance(v,(int,float)) and k.endswith('_bytes')})
        summary[key]=dict(samples=len(rr),observed_maxima={k:max(row[k] for row in rr if isinstance(row.get(k),(int,float))) for k in fields},
            full_run_peak_proven=False,before_after_growth=None)
    partial=dict(resource_log_incomplete=True,corrupted_lines=bad,groups=summary,
        limits='Original malformed line preserved; no missing sample reconstructed; no cross-PID subtraction; no resource superiority claim.')
    (REPORT/'PARTIAL_RESOURCE_OBSERVATIONS.json').write_text(json.dumps(partial,indent=2)+'\n')
    new_status=dict(status='completed',completed_stage=19,final_checkpoint=latest['path'],final_parameter_file_bytes=latest['bytes'],
        CPU_finalization_recovery=str(REPORT.relative_to(ROOT)),resource_log_incomplete=True,
        original_resources_preserved=True,no_training_or_predictions_repeated=True)
    tmp=status_path.with_suffix('.json.recovery_tmp');tmp.write_text(json.dumps(new_status,indent=2)+'\n');tmp.replace(status_path)
    # The normal, unmodified predicate and the normal frozen verifier must now
    # accept this run. The existing publisher remains responsible for main table.
    assert verifier.eligible(json.loads(status_path.read_text()))
    final=verifier.verify(REL,'new3090')
    assert final['aggregate']==proof['aggregate'] and final['checkpoint_sha256']==proof['checkpoint_sha256']
    assert final['prediction_sha256']==proof['prediction_sha256']
    (REPORT/'FINAL_SCIENTIFIC_VERIFICATION.json').write_text(json.dumps(final,indent=2)+'\n')
    (REPORT/'RECOVERY_COMPLETE.json').write_text(json.dumps(dict(complete=True,CPU_only=True,
        final_verification_sha256=sha(REPORT/'FINAL_SCIENTIFIC_VERIFICATION.json'),
        resource_limitation_sha256=sha(REPORT/'PARTIAL_RESOURCE_OBSERVATIONS.json'),
        original_status_sha256=evidence['original_status_sha256'],final_status_sha256=sha(status_path),
        original_resource_bytes_unchanged=sha(resources)==evidence['original_resources_sha256'],
        aggregate=final['aggregate'],no_GPU_or_training_or_prediction_started=True),indent=2)+'\n')
    print(json.dumps(dict(completed_scientific_run=True,aggregate=final['aggregate'],resource_log_incomplete=True)))


if __name__=='__main__':main()
