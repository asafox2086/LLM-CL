"""Synthetic CPU integrity checks only; never writes experiment outcomes."""
import contextlib, io, json, tempfile
from pathlib import Path
import torch
import onevision_knowledge_fit_evidence as audit


def main():
    source = Path(audit.__file__)
    pool = audit.read(audit.ROOT / 'data/OneVisionMechanisms/20261016/knowledge16.json')
    with tempfile.TemporaryDirectory(prefix='knowledge-collector-check-') as tmp:
        root = Path(tmp); report = root / 'report'; (report / 'knowledge_fit').mkdir(parents=True)
        def put(p, value):
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value) + '\n')
        def lines(p, rows):
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        pool_path = root / 'data/OneVisionMechanisms/20261016/knowledge16.json'; put(pool_path, pool)
        expected = [{'id': 'a', 'answers': ['yes'], 'images': ['a.png']},
                    {'id': 'b', 'answers': ['no'], 'images': ['b.png']}]
        audit.ROOT = root; audit.REPORT = report; audit.__file__ = str(root / 'collector.py')
        Path(audit.__file__).write_bytes(source.read_bytes())
        audit.freeze = lambda: (None, {(11, 'test'): expected}, None, {'medical': expected}, None, None, None)
        proofs = []
        for name in audit.CASES:
            run = root / name; run.mkdir()
            spec = {'protocol_version': 3, 'pool_sha256': audit.digest(pool_path), 'source_sha256': {},
                    'auxiliary_exposures_at64': 512, 'updates': 64, 'batch_QA': 8,
                    'cases': [[n, 1 if 'rank1' in n else 8, 'permuted' if n.endswith('permuted') else 'correct'] for n in audit.CASES],
                    'clinical_ids': {g: ['a', 'b'] for g in ['medical', 'medical_KG']}}
            put(run / 'protocol.json', spec); put(run / 'initial_function_gate.json', {'passed': True})
            put(run / 'memory_summary.json', {'synthetic': True}); training = []
            for step in range(1, 65):
                p = run / 'raw_gradients' / f'update_{step:02d}.pt'; p.parent.mkdir(exist_ok=True)
                torch.save({'preclip_gradient': torch.tensor([3., 4.]), 'layout': [('toy', [2])]}, p)
                training.append({'step': step, 'lr_used': 1e-4, 'cumulative_auxiliary_QA': step * 8,
                                 'pre_update_mean_loss': 1., 'gradient_L2_preclip': 5.})
            lines(run / 'training.jsonl', training)
            for step in [0, 8, 64]:
                torch.save({'updates': step, 'adapter': {'toy': {'A': torch.tensor([[1., 2.]]),
                           'B': torch.tensor([[step / 64.]])}}}, run / f'update_{step:02d}.pt')
                for cohort in ['relevant_true', 'own_target']:
                    kind = 'permuted' if name.endswith('permuted') and cohort == 'own_target' else 'correct'
                    rows = [{'id': r['id'], 'reference': r['answer'], 'template': r['template'], 'prediction': 'synthetic',
                             'nll': 1., 'scores': audit.score('synthetic', [r['answer']])} for r in audit.fact_rows(pool[kind], cohort)]
                    lines(run / 'evaluations' / f'update_{step:02d}' / (cohort + '_text.jsonl'), rows)
                for group in ['medical', 'medical_KG']:
                    lines(run / 'evaluations' / f'clinical_update_{step:02d}' / (group + '_image.jsonl'),
                          [{'id': r['id'], 'references': r['answers'], 'prediction': 'yes', 'nll': 1.,
                            'scores': audit.score('yes', r['answers'])} for r in expected])
            lines(run / 'resources.jsonl', [{'pid': 1, 'unix': i, 'phase': phase} for i, phase in enumerate(
                  ['fit_window_1_before', 'fit_window_1_after', 'fit_window_2_before', 'fit_window_2_after'])])
            artifacts = [*run.rglob('*.pt'), *run.rglob('*.jsonl'), run / 'protocol.json',
                         run / 'memory_summary.json', run / 'initial_function_gate.json']
            proof = report / 'knowledge_fit' / (name + '_complete.json')
            put(proof, {'complete': True, 'formal_full_test': False, 'run': name, 'spec': spec,
                        'artifact_sha256': {str(p.relative_to(run)): audit.digest(p) for p in artifacts}}); proofs.append(proof)
        with contextlib.redirect_stdout(io.StringIO()): audit.main()
        result = audit.read(report / 'KNOWLEDGE_FIT_EVIDENCE.json')
        assert len(result['summaries']) == 72 and len(result['paired_content_controls']) == 12
        assert result['summaries'][0]['ROUGE_L'] == 0
        assert set(result['paired_content_controls'][0]['metrics']) == {'exact_match', 'token_f1', 'rougeL', 'prefix_aux_EM', 'NLL'}
        artifact = root / audit.CASES[0] / 'training.jsonl'; artifact.write_text(artifact.read_text() + '{}\n')
        try: audit.main()
        except AssertionError: pass
        else: raise AssertionError('Tampered artifact accepted')
        for p in proofs: p.unlink()
        (report / 'KNOWLEDGE_FIT_EVIDENCE.json').unlink()
        with contextlib.redirect_stdout(io.StringIO()): audit.main()
        assert not (report / 'KNOWLEDGE_FIT_EVIDENCE.json').exists() and not torch.cuda.is_initialized()
    print(json.dumps({'synthetic_fixture_only': True, 'collector_sha256': audit.digest(source),
                      'four_cases_72_summary_12_paired_cells': True, 'all_five_paired_metrics': True,
                      'tampered_artifact_rejected': True, 'no_completion_no_scores': True, 'GPU_used': False}))


if __name__ == '__main__': main()
