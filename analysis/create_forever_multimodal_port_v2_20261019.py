"""Independent repair: unchanged training, explicit AMP cleanup before saving."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'exp/two_backbone_forever_20261019_v1'
TARGET = ROOT/'exp/two_backbone_forever_20261019_v2'
REPORT = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v2'


def main():
    assert not TARGET.exists()
    plan = json.loads((REPORT.parent/'v1/GPU_GATE_PLAN.json').read_text())
    for name,digest in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    old = ast.parse((ROOT/'analysis/forever_multimodal_replay_20261019.py').read_text())
    new = ast.parse((ROOT/'analysis/forever_multimodal_replay_v2_20261019.py').read_text())
    old_nodes = {getattr(n,'name',str(i)): n for i,n in enumerate(old.body)}
    new_nodes = {getattr(n,'name',str(i)): n for i,n in enumerate(new.body)}
    assert set(old_nodes) == set(new_nodes)
    for key in old_nodes:
        if key != 'execute_replay':
            assert ast.dump(old_nodes[key]) == ast.dump(new_nodes[key]), key
    a,b = old_nodes['execute_replay'],new_nodes['execute_replay']
    def cut(function):
        return next(i for i,node in enumerate(function.body)
                    if ast.unparse(node) == 'stage.replay_complete()')
    ai,bi = cut(a),cut(b)
    assert ai == bi
    assert ast.dump(ast.Module(body=a.body[:ai+1],type_ignores=[])) == ast.dump(ast.Module(body=b.body[:bi+1],type_ignores=[]))
    report_index = next(i for i,node in enumerate(b.body[bi+1:],bi+1)
                        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='report' for t in node.targets))
    assert ast.dump(ast.Module(body=a.body[ai+1:],type_ignores=[])) == ast.dump(ast.Module(body=b.body[report_index:],type_ignores=[]))
    TARGET.mkdir()
    for p in BASE.glob('*.py'):
        raw = p.read_bytes()
        if p.name in ['run.py','forever_adapter.py']:
            raw = raw.replace(b'forever_multimodal_replay_20261019',
                              b'forever_multimodal_replay_v2_20261019')
        (TARGET/p.name).write_bytes(raw)
    policy = json.loads((BASE/'execution_policy.json').read_text())
    policy['experiment_id'] = TARGET.name
    (TARGET/'execution_policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    worker = (ROOT/'analysis/forever_multimodal_gate_worker_20261019.py').read_bytes()
    worker = worker.replace(b'two_backbone_forever_20261019_v1',
                            b'two_backbone_forever_20261019_v2')
    worker = worker.replace(b'forever_multimodal_20261019/v1',
                            b'forever_multimodal_20261019/v2')
    p = ROOT/'analysis/forever_multimodal_gate_worker_v2_20261019.py'
    assert not p.exists();p.write_bytes(worker)
    verifier = (ROOT/'analysis/forever_multimodal_formal_verify_20261019.py').read_bytes()
    verifier = verifier.replace(b'two_backbone_forever_20261019_v1',
                                b'two_backbone_forever_20261019_v2')
    verifier = verifier.replace(b'forever_multimodal_20261019/v1',
                                b'forever_multimodal_20261019/v2')
    verifier = verifier.replace(b'forever_multimodal_replay_20261019',
                                b'forever_multimodal_replay_v2_20261019')
    needle = b"        assert completed['status'] == 'ACTUAL_REPLAY_COMPLETE'"
    assert verifier.count(needle) == 1
    extra = b"""        cleanup = json.loads(checked(replay/'AMP_UNWRAP_EXACT.json', hashes).read_text())
        assert cleanup['keep_fp32_wrapper'] is False
        assert cleanup['model_identity_storage_versions_exact']
        assert cleanup['common_explicit_autocast_NLL_exact']
        assert cleanup['NLL_before'] == cleanup['NLL_after']
        assert cleanup['extra_TRAIN_NLL_forwards'] == 2 and cleanup['extra_optimizer_updates'] == 0
"""
    verifier = verifier.replace(needle,extra+needle)
    p = ROOT/'analysis/forever_multimodal_formal_verify_v2_20261019.py'
    assert not p.exists();p.write_bytes(verifier)
    proof = {
        'status':'INDEPENDENT_V2_REPAIR_PRE_GPU',
        'v1_scientific_sources_untouched':True,
        'all_non_replay_functions_AST_identical':True,
        'all_training_and_optimizer_statements_AST_identical':True,
        'difference':'post-TRAIN unwrap_model(keep_fp32_wrapper=False) before common graph checkpoint',
        'post_TRAIN_checks':'same parameter identity/storage/versions/hash and one old TRAIN NLL before/after; restore RNG',
        'extra_TRAIN_forward_per_replay':2,'extra_optimizer_updates':0,
        'all_unchanged_port_files':[p.name for p in BASE.glob('*.py') if p.name not in ['run.py','forever_adapter.py']],
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [ROOT/'analysis/forever_multimodal_replay_20261019.py',
                                  ROOT/'analysis/forever_multimodal_replay_v2_20261019.py']},
    }
    REPORT.mkdir(parents=True,exist_ok=True)
    p=REPORT/'V1_REPAIR_EQUIVALENCE.json';assert not p.exists()
    p.write_text(json.dumps(proof,indent=2)+'\n')
    (TARGET/'COPY_PROVENANCE.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'independent_v2_created':str(TARGET),'training_AST_unchanged':True,
                      'GPU_admitted':False}))


if __name__ == '__main__':
    main()
