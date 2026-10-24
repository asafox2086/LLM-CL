"""Metadata plus the one admitted tiny CPU gate. Never a real-model launcher."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path
from common import ASSETS, HERE, PORT, REPORT, ROOT, sha, training_ast, write


def main():
    assert 'torch' not in sys.modules
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    assert not any((REPORT / name).exists() for name in ('CPU_GATE_STARTED.json', 'CPU_GATE.json', 'READY.json'))
    source_review = json.loads((REPORT / 'SOURCE_REVIEW.json').read_text())
    assert source_review['decision'] == 'accept_sparse_pilot_source_for_single_CPU_gate'
    assert source_review['protocol_sha256'] == sha(REPORT / 'PROTOCOL.md')
    for relative, digest in source_review['sources_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    for relative, digest in source_review['dependencies_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    necessity = json.loads((REPORT / 'NECESSITY_REVIEW.json').read_text())
    assert necessity['decision'] == 'accept_fixed_sparse_carrier_implementation_only'
    assert necessity['protocol_sha256'] == sha(REPORT / 'PROTOCOL.md')
    author = json.loads((ASSETS / 'SOURCE_RECEIPT.json').read_text())
    assert author['revision'] == '147d2159dc4f3e9f73e47b32c04d7b3708f44436'
    assert {name: record['sha256'] for name, record in author['files'].items()} == {
        'sparsegpt.py': 'cca2ec0a006f963bc7750ebec8e2320aa2728fb242924cc3f2d5d23fb98530a7',
        'LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'}
    for name, record in author['files'].items():
        assert sha(ASSETS / name) == record['sha256']
    old = json.loads((ASSETS / 'screen_READY.json').read_text())
    split = json.loads((ASSETS / 'SPLIT.json').read_text())
    assert sha(ASSETS / 'SPLIT.json') == old['split_sha256']
    assert training_ast()[1] == old['train_AST_sha256']
    plan = [dict(stage=int(k), QA=len(ids), updates=(len(ids) + 15) // 16)
            for k, ids in sorted(split['fit_ids'].items(), key=lambda row: int(row[0]))]
    assert sum(p['QA'] for p in plan) == 1213 and sum(p['updates'] for p in plan) == 87
    fit1 = set(split['fit_ids']['1']); groups = {}
    for gid, group in split['groups'].items():
        ids = fit1 & set(group['ids'])
        if ids:
            groups[gid] = ids
    ordered = sorted(groups, key=lambda g: hashlib.sha256(('42:sparse-current-S1:' + g).encode()).digest())
    reps = [min(groups[g], key=lambda i: hashlib.sha256(('42:sparse-QA:' + i).encode()).digest()) for g in ordered]
    calibration = [hashlib.sha256(i.encode()).hexdigest() for i in reps[:16]]
    diagnostic = [hashlib.sha256(i.encode()).hexdigest() for i in reps[16:32]]
    assert len(calibration) == len(diagnostic) == 16
    base = ROOT / 'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    # Tokenizer/processor/config contracts are new bindings, not a retrospective
    # claim that old experiments sealed all of them. Full weights hash at init.
    base_metadata = {p.relative_to(ROOT).as_posix(): sha(p) for p in base.iterdir()
                     if p.is_file() and p.suffix in ('.json', '.txt', '.model', '.jinja')}
    packages = {}
    installed_code = {}
    for name in ('torch', 'transformers', 'tokenizers', 'cloudpickle'):
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin
        packages[name] = dict(version=importlib.metadata.version(name), origin=spec.origin)
        installed_code[str(Path(spec.origin).resolve())] = sha(spec.origin)
    transformers_root = Path(packages['transformers']['origin']).parent
    for relative in ('models/llava_onevision/modeling_llava_onevision.py',
                     'models/llava_onevision/processing_llava_onevision.py',
                     'models/llava_onevision/image_processing_llava_onevision.py',
                     'models/qwen2/modeling_qwen2.py', 'models/siglip/modeling_siglip.py',
                     'generation/utils.py', 'modeling_utils.py', 'tokenization_utils_base.py'):
        path = (transformers_root / relative).resolve()
        assert path.is_file(), relative
        installed_code[str(path)] = sha(path)
    assert 'torch' not in sys.modules
    write(REPORT / 'CPU_GATE_STARTED.json', dict(pid=os.getpid(), CUDA_visible='', source_review_sha256=sha(REPORT / 'SOURCE_REVIEW.json')))
    def expired(*unused):
        raise TimeoutError('Unique tiny CPU gate exceeded 600 seconds; preserve and stop')
    signal.signal(signal.SIGALRM, expired); signal.alarm(600)
    begin = time.monotonic()
    from cpu_gate import run
    gate = run(); gate['CPU_seconds'] = time.monotonic() - begin
    signal.alarm(0)
    write(REPORT / 'CPU_GATE.json', gate)
    assert gate['passed'] and not gate['CUDA_initialized']
    evidence_paths = [REPORT / n for n in ('PROTOCOL.md', 'NECESSITY_REVIEW.json', 'SOURCE_REVIEW.json', 'CPU_GATE.json')]
    evidence_paths += [ASSETS / n for n in ('SOURCE_RECEIPT.json', 'sparsegpt.py', 'LICENSE', 'screen_READY.json', 'SPLIT.json', 'reference_answer_guided_READY.json')]
    ready = dict(sources=source_review['sources_sha256'],
        dependencies={**source_review['dependencies_sha256'], **base_metadata, **installed_code},
        evidence={p.relative_to(ROOT).as_posix(): sha(p) for p in evidence_paths},
        base_files=old['base_files'], manifest_sha256=old['manifest_sha256'], split_sha256=old['split_sha256'],
        training_ast_sha256=training_ast()[1], training_plan=plan,
        calibration_ID_SHA256=calibration, diagnostic_ID_SHA256=diagnostic,
        source_revision=author['revision'], cases=['Frozen', 'Plastic'], host='OLD',
        allowed_physical_GPUs=[0, 1, 2, 3],
        init_budget_seconds=3600, case_budget_seconds=7200, aggregate_model_budget_seconds=18000,
        packages=packages, executable=sys.executable,
        final_weight_values=737323040, pattern_bytes=78446592,
        no_model_admission=True, scientific_sources_frozen=True, TEST_QA=0,
        CPU_admission_not_real_VLM_or_quality=True)
    write(REPORT / 'READY.json', ready)
    print(json.dumps(dict(ready_sha256=sha(REPORT / 'READY.json'), CPU_seconds=gate['CPU_seconds'], no_model_admission=True)))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        write(REPORT / 'CPU_GATE_FAILURE.json', dict(traceback=traceback.format_exc(), no_automatic_retry=True))
        raise
