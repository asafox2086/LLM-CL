"""Resume unchanged full-confirmation queue after pain-point controls.

CPU coordinator only. Current cap1 GPU worker is allowed to finish unchanged;
cap3 waits rural inference and knowledge manipulation validation. No hot edits.
"""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def write(value):
    p=REPORT/'deferred_full_confirmation_status.json';q=p.with_suffix('.partial')
    q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)


def main():
    lock=(REPORT/'deferred_full_confirmation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        folders=[REPORT/'rural_collapse',REPORT/'knowledge_fit',REPORT/'rural_weights']
        failures=[str(f.relative_to(ROOT)) for f in folders if (f/'needs_attention.json').exists()]
        if failures:
            write({'phase':'diagnostic_needs_attention','pid':os.getpid(),'folders':failures});return
        missing=[str(f.relative_to(ROOT)) for f in folders if not (f/'complete.json').exists()]
        if missing:
            write({'phase':'waiting_painpoint_diagnostics','pid':os.getpid(),'pending':missing});time.sleep(15);continue
        # Original queue verifies idle GPU1 and preserves its exact pointers,
        # renderer, checkpoint references, prediction resume and failure rules.
        write({'phase':'resuming_unchanged_full_queue','pid':os.getpid()})
        import onevision_full_repair_queue
        onevision_full_repair_queue.main()
        complete=all((REPORT/'full_input_repair'/f'cap{cap}_full_confirm_complete.json').exists() for cap in [1,3])
        write({'phase':'completed' if complete else 'full_queue_needs_attention','pid':os.getpid()});return


if __name__=='__main__':main()
