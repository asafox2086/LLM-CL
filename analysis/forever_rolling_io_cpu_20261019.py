"""Crash-boundary controls for the new runner I/O; no trained model or GPU."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT/'exp/two_backbone_forever_20261019_v3'
REPORT = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v3'
sys.path.insert(0,str(FRAME))
import rolling_io as io


def fake_stage(run, stage, count):
    """Synthetic files exercise transaction boundaries, not ML correctness."""
    weight = run/'checkpoints'/f'stage_{stage:02d}.pt'
    weight.parent.mkdir(exist_ok=True)
    raw = f'SYNTHETIC_NOT_A_MODEL_STAGE_{stage}'.encode()
    weight.write_bytes(raw)
    meta = {'stage':stage,'path':str(weight.relative_to(run)),
            'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    io.write(run/'checkpoints/latest.json',meta)
    io.write(weight.with_suffix('.json'),meta)
    io.write(run/'stages'/f'stage_{stage:02d}.json',
             {'stage':stage,'complete':True,'weights_saved':True})
    io.write(run/'validity'/f'stage_{stage:02d}.json',{'synthetic_stage':stage})
    io.write(run/'method_state'/f'stage_{stage:02d}'/'synthetic.json',{'stage':stage})
    io.write(run/'evaluations'/f'stage_{stage:02d}'/'synthetic.json',{'stage':stage})
    path=run/'logs'/f'stage_{stage:02d}.events.jsonl'
    path.parent.mkdir(exist_ok=True)
    with path.open('x') as stream:
        for step in range(1,count+1):
            stream.write(json.dumps({'event':'train_update','stage':stage,'step':step})+'\n')
    io.write(run/'run_status.json',{'status':'stage_complete','completed_stage':stage})
    return meta


def main():
    destination = REPORT/'ROLLING_IO_CPU_CHECKS.json'
    assert not destination.exists()
    checks=[]
    manifest={'tasks':[{'files':{'train':{'count':n}}} for n in [17,2,32]]}
    with tempfile.TemporaryDirectory(prefix='forever-rolling-') as temporary:
        run=Path(temporary)
        try:io.prepare_resume(run,manifest)
        except AssertionError:pass
        else:raise AssertionError('Nonexistent committed checkpoint accepted')
        checks.append('missing committed stage cannot silently resume dirty weights')
        fake_stage(run,1,2);io.commit_stage(run,1,manifest)
        fake_stage(run,2,1);io.commit_stage(run,2,manifest)
        assert not (run/'checkpoints/stage_01.pt').exists()
        assert (run/'checkpoints/stage_02.pt').exists()
        assert (run/'checkpoints/stage_01.json').exists()
        checks.append('only latest committed weight retained; older SHA metadata retained')
        # Crash after writing a newer weight/latest pointer but BEFORE commit.
        fake_stage(run,3,2)
        with (run/'logs/stage_03.events.jsonl').open('a') as stream:
            stream.write('{interrupted JSON tail')
        original=(run/'logs/stage_03.events.jsonl').read_bytes()
        resumed=io.prepare_resume(run,manifest)
        assert resumed['stage']==2
        assert json.loads((run/'checkpoints/latest.json').read_text())['stage']==2
        assert not (run/'checkpoints/stage_03.pt').exists()
        assert (run/'checkpoints/stage_02.pt').exists()
        branches=list((run/'interrupted_branches').iterdir());assert len(branches)==1
        branch=branches[0]
        assert (branch/'logs/stage_03.events.jsonl').read_bytes()==original
        assert (branch/'checkpoints/stage_03.pt').exists()
        assert (branch/'method_state/stage_03/synthetic.json').exists()
        assert (branch/'evaluations/stage_03/synthetic.json').exists()
        assert (branch/'validity/stage_03.json').exists()
        checks.append('uncommitted model/events/replay/evaluation/validity archived intact')
        events=[json.loads(x) for x in (run/'logs/events.jsonl').read_text().splitlines()]
        assert [(x['stage'],x['step']) for x in events]==[(1,1),(1,2),(2,1)]
        checks.append('canonical events contain committed updates only, despite damaged future tail')
        fake_stage(run,3,2);io.commit_stage(run,3,manifest)
        assert list((run/'checkpoints').glob('*.pt'))==[run/'checkpoints/stage_03.pt']
        events=[json.loads(x) for x in (run/'logs/events.jsonl').read_text().splitlines()]
        assert [(x['stage'],x['step']) for x in events]==[(1,1),(1,2),(2,1),(3,1),(3,2)]
        checks.append('rerun future stage counted once and rolling final retained')
        # Reject a path escaping the intended run, even if it is a real file.
        try:io.inside(run,run/'../outside.pt')
        except AssertionError:pass
        else:raise AssertionError('Escaping archive/delete target accepted')
        checks.append('archive/delete paths confined to the exact run')
        monitor=io.ProcessResourceMonitor(run)
        monitor.snapshot('training_before');monitor.snapshot('training_after')
        first=monitor.finish()
        assert first['full_run_growth_available']
        monitor=io.ProcessResourceMonitor(run,resume=True)
        monitor.snapshot('training_before');monitor.snapshot('training_after')
        second=monitor.finish()
        assert not second['full_run_growth_available'] and second['full_run_growth_bytes'] is None
        assert second['no_cross_process_memory_subtraction']
        checks.append('resume resource segment cannot claim full-run or cross-PID growth')
        import torch
        assert not torch.cuda.is_initialized()
    proof={'status':'ROLLING_TRANSACTION_AND_RECOVERY_CPU_PASS',
           'checks':checks,'checks_passed':len(checks),
           'GPU':False,'ML_training_or_performance':False,
           'synthetic_checkpoint_bytes_not_models':True,
           'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [FRAME/'rolling_io.py',Path(__file__)]}}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))


if __name__ == '__main__':main()
