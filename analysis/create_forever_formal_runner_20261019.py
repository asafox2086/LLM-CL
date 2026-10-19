"""Isolated v3 I/O fork; preserves the two-backbone GPU-tested model kernels."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'exp/two_backbone_forever_20261019_v2'
TARGET = ROOT/'exp/two_backbone_forever_20261019_v3'
REPORT = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v3'


def main():
    assert not TARGET.exists()
    plan = json.loads((REPORT.parent/'v2/GPU_GATE_PLAN.json').read_text())
    for name,digest in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest,name
    TARGET.mkdir()
    for p in BASE.glob('*.py'):
        (TARGET/p.name).write_bytes(p.read_bytes())
    source = (BASE/'run.py').read_text()
    run = source
    needle = 'from checkpoints import save as save_checkpoint, load as load_checkpoint\n'
    assert run.count(needle) == 1
    run = run.replace(needle,needle+
        'from rolling_io import load as load_checkpoint, commit_stage, ProcessResourceMonitor as ResourceMonitor\n')
    run = run.replace('monitor = ResourceMonitor(run)',
                      'monitor = ResourceMonitor(run, resume=args.resume)')
    run = run.replace('load_checkpoint(run) if args.resume else None',
                      'load_checkpoint(run, manifest) if args.resume else None')
    needle = "    config['external_helper_sha256'] = "
    assert run.count(needle) == 1
    run = run.replace(needle,"    config['rolling_checkpoint'] = True\n"+needle)
    run = run.replace("assert config['checkpoint_all_stages']",
                      "assert config['rolling_checkpoint']")
    run = run.replace("{'status': 'initial_evaluation', 'completed_stage': 0}",
        "{'status': 'resume_loaded' if resume_obj else 'initial_evaluation', 'completed_stage': resume_obj['stage'] if resume_obj else 0}")
    run = run.replace("if not resume_obj: monitor.snapshot('training_before', model=model)",
                      "monitor.snapshot('training_before', model=model)")
    needle = "with (run / 'logs/events.jsonl').open('a', encoding='utf-8') as events:"
    assert run.count(needle) == 1
    run = run.replace(needle,
        "with (run / 'logs' / f'stage_{stage:02d}.events.jsonl').open('x', encoding='utf-8') as events:")
    needle = "        monitor.snapshot('stage_training_end_with_optimizer', stage, model)"
    assert run.count(needle) == 1
    run = run.replace(needle,"            events.flush(); os.fsync(events.fileno())\n"+needle)
    run = run.replace("if config['checkpoint_all_stages'] or stage==len(manifest['tasks']):",
                      "if config['rolling_checkpoint'] or config['checkpoint_all_stages'] or stage==len(manifest['tasks']):")
    needle = "        write(run / 'run_status.json', {'status': 'stage_complete', 'completed_stage': stage})"
    assert run.count(needle) == 1
    run = run.replace(needle,needle+'\n        commit_stage(run, stage, manifest)')
    needle = '    aggregate_sparse(run, manifest, write)'
    assert run.count(needle) == 1
    run = run.replace(needle,"    if resume_obj and resume_obj['stage'] == len(manifest['tasks']):\n        monitor.snapshot('training_after', resume_obj['stage'], model)\n"+needle)
    (TARGET/'run.py').write_text(run)
    (TARGET/'rolling_io.py').write_bytes((ROOT/'analysis/forever_rolling_io_20261019.py').read_bytes())
    policy = json.loads((BASE/'execution_policy.json').read_text())
    policy['experiment_id'] = TARGET.name
    policy['checkpoint_other_methods'] = 'rolling latest committed stage and final'
    (TARGET/'execution_policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    # The actual minibatch math/optimizer/update loop must be exactly unchanged.
    def math_loop(text):
        tree = ast.parse(text)
        main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        outer = next(n for n in main.body if isinstance(n,ast.For) and ast.unparse(n.target)=='stage')
        context = next(n for n in outer.body if isinstance(n,ast.With))
        loop = next(n for n in context.body if isinstance(n,ast.For))
        return ast.dump(loop)
    assert math_loop(source) == math_loop(run)
    for p in BASE.glob('*.py'):
        if p.name != 'run.py':
            assert p.read_bytes() == (TARGET/p.name).read_bytes(),p.name
    proof = {
        'status':'V3_IO_ONLY_BEFORE_FORMAL_ADMISSION',
        'minibatch_loss_gradient_optimizer_update_AST_exact_v2':True,
        'all_other_model_adapter_input_evaluation_checkpoint_files_exact_v2':True,
        'I_O_changes':['separate committed checkpoint pointer','retain only latest stage weight',
                      'archive uncommitted branch on resume','canonical completed-stage events only',
                      'per-process resource scope; no cross-PID growth subtraction'],
        'v2_gate_training_source_plan_sha256':hashlib.sha256(
            (REPORT.parent/'v2/GPU_GATE_PLAN.json').read_bytes()).hexdigest(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [BASE/'run.py',TARGET/'run.py',TARGET/'rolling_io.py']},
        'GPU_or_formal_run_launched':False,
    }
    REPORT.mkdir(parents=True,exist_ok=True)
    p=REPORT/'IO_EQUIVALENCE.json';assert not p.exists()
    p.write_text(json.dumps(proof,indent=2)+'\n')
    (TARGET/'COPY_PROVENANCE.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'created':str(TARGET),'training_loop_AST_exact':True}))


if __name__ == '__main__':
    main()
