"""Latest user prioritizes native pain-point research; defer formal full trials."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import onevision_formal_cache_queue as previous

ROOT=previous.ROOT
REPORT=previous.STUDY/'simple_research_queue'


def defer(action,inherited,built,full_done,gpu):
    # Preserve the current worker and the old diagnostic/control stream.
    # Formal candidates remain registered, but are not selected ahead of it.
    return action


def main(check=False):
    REPORT.mkdir(parents=True,exist_ok=True)
    auth=json.loads((REPORT/'AUTHORIZATION.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==auth['simple_focus_source_sha256']
    previous.REPORT=REPORT
    if not check:previous.choose=defer
    previous.main(check)
    if check:
        cases=[('wait',None),('acquired_anchor',4),('flex_research',{'kind':'factorial','index':2}),('complete',None)]
        assert all(defer(a,False,False,[False,False],None)==a for a in cases)
        (REPORT/'FOCUS_CPU_CHECK.json').write_text(json.dumps(dict(CPU_only=True,checks=len(cases),
            original_controls_preserved=True,formal_candidates_deferred_not_deleted=True,GPU_started=False))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    try:main(a.check)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
