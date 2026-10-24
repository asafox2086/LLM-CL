"""Consume the existing unique VERIFIED scalars; never load models or score QA."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def paired(left, right):
    assert [r['id'] for r in left] == [r['id'] for r in right]
    delta = [a['F1'] - b['F1'] for a, b in zip(left, right)]
    return dict(higher=sum(d > 0 for d in delta), lower=sum(d < 0 for d in delta),
                equal=sum(d == 0 for d in delta))


def main(root):
    tick = time.monotonic()
    report = root / 'summary/final_weight_reduction_20261024/ffn_role_retention_v1'
    work = root / 'exp/ffn_role_retention_20261024/v1/private'
    target = report / 'OUTCOME.json'
    assert not target.exists(), 'Outcome already consumed; do not repeat'
    verified = read(report / 'VERIFIED.json')
    assert verified['complete'] and verified['TEST_QA'] == 0
    assert verified['ready_sha256'] == sha(report / 'READY.json')
    rows = verified['rows']
    assert set(rows) == {'init_parent', 'KeepGate_initial', 'KeepGate_final', 'KeepUp_initial', 'KeepUp_final'}
    init_complete_path = report / 'init_COMPLETE.json'
    assert sha(init_complete_path) == verified['case_complete_sha256']['init']
    init_complete = read(init_complete_path)
    assert sha(work / 'init/initializations.json') == init_complete['files']['init/initializations.json']
    meta = read(work / 'init/initializations.json')
    parent_count = meta['parent_layout']['weights']['coefficients']
    assert parent_count == 893675552
    costs = {}
    for case in ('init', 'KeepGate', 'KeepUp'):
        complete_path = report / (case + '_COMPLETE.json')
        assert sha(complete_path) == verified['case_complete_sha256'][case]
        complete = read(complete_path)
        exit_ = read(report / (case + '_EXIT.json'))
        assert exit_['returncode'] == 0 and exit_['identity_after'] is None
        resource_path = work / case / 'resources.jsonl'
        assert sha(resource_path) == complete['files'][case + '/resources.jsonl']
        resources = [json.loads(x) for x in resource_path.read_text().splitlines()]
        fields = ('VmRSS_bytes', 'VmHWM_bytes', 'Pss_bytes', 'cpu_uss_bytes',
                  'gpu_allocated_bytes', 'gpu_reserved_bytes', 'gpu_peak_allocated_bytes',
                  'gpu_peak_reserved_bytes', 'gpu_process_nvml_bytes')
        peaks = {k: max((r[k] for r in resources if r.get(k) is not None), default=None) for k in fields}
        cost = dict(worker=complete['worker'], run_seconds=complete['seconds'],
                    observed_exit_seconds=exit_['observed_elapsed_seconds'],
                    resource_rows=len(resources), observed_resource_max_bytes=peaks,
                    resources_sha256=complete['files'][case + '/resources.jsonl'])
        if case == 'init':
            cost['parent_layout'] = meta['parent_layout']
            cost['initial_checkpoints'] = meta['checkpoints']
        else:
            proof_path = work / case / 'result_proof.json'
            witness_path = work / case / 'optimizer.jsonl'
            assert sha(proof_path) == complete['files'][case + '/result_proof.json']
            assert sha(witness_path) == complete['files'][case + '/optimizer.jsonl']
            proof = read(proof_path)
            witness = [json.loads(x) for x in witness_path.read_text().splitlines()]
            cost.update(final_layout=complete['final_layout'], final_checkpoint=proof['final_checkpoint'],
                        updates=complete['updates'], stages=complete['stages'], FIT_QA=complete['FIT_QA'],
                        observed_gradient_bytes=sorted({r['gradient_bytes'] for r in witness}),
                        observed_cpu_moments_bytes=sorted({r['cpu_moments_bytes'] for r in witness}),
                        trainable_coordinates=sorted({r['parameter_count'] for r in witness}),
                        stage_adapter_files=19,
                        stage_adapter_package_bytes=sum((work / case / f'stage_adapters/{i:02d}.pt').stat().st_size for i in range(1, 20)),
                        gradient_files=87,
                        gradient_package_bytes=sum((work / case / f'gradients/{i:03d}.pt').stat().st_size for i in range(1, 88)))
        costs[case] = cost
    compact_count = costs['KeepGate']['final_layout']['weights']['coefficients']
    assert compact_count == costs['KeepUp']['final_layout']['weights']['coefficients'] == 816728096
    comparisons = {}
    for left, right in (('KeepGate_initial', 'init_parent'), ('KeepUp_initial', 'init_parent'),
                        ('KeepGate_final', 'KeepGate_initial'), ('KeepUp_final', 'KeepUp_initial'),
                        ('KeepGate_final', 'KeepUp_final')):
        a, b = rows[left], rows[right]
        comparisons[left + '_minus_' + right] = dict(
            macro_difference_pp=a['macro'] - b['macro'],
            domain_differences_pp={k: a['domains'][k] - b['domains'][k] for k in a['domains']},
            per_id_comparison=paired(a['per_id'], b['per_id']))
    out = dict(complete=True, source_VERIFIED_sha256=sha(report / 'VERIFIED.json'),
               consumer_sha256=sha(Path(__file__)), rows=rows, comparisons=comparisons,
               parent_weight_coefficients=parent_count, final_weight_coefficients=compact_count,
               weight_reduction_coefficients=parent_count-compact_count,
               weight_reduction_percent=100*(parent_count-compact_count)/parent_count,
               old_wideqv_anchor=verified['old_wideqv_anchor'],
               final_minus_anchor_pp={case: rows[case+'_final']['macro']-verified['old_wideqv_anchor'] for case in ('KeepGate', 'KeepUp')},
               primary_point_pass=verified['primary_point_pass'],
               decision='retain_for_framework_design_review' if verified['primary_point_pass'] else 'close_fixed_role_rank_points',
               costs=costs, model_elapsed_observed_sum=verified['model_elapsed_observed_sum'],
               cost_scope='Model phase RUN and observed exit bounds, complete checkpoint metadata, sampled process memory, and current historical adapter/gradient file sizes; overlapping times must not be added.',
               separately_recorded_costs=['tiny CPU gate and its supervisor', 'implementation and independent review',
                                          'asset transport and metadata freeze', 'resource waiting and SSH',
                                          'unique verification and this consumer', 'all prior research history'],
               generated_QA=500, FIT_QA_exposures=2426, optimizer_updates=174,
               limitations=['explored DEV, 20 QA per domain, one training seed',
                            'initial functions differ; adaptation differences do not isolate pure trainability',
                            'continuous q/v adapter; not a cold-start causal test',
                            'observed memory is this full execution, not controlled inference peaks',
                            'no automatic full confirmation, new rank/LR/recovery point, or novel-framework claim'],
               new_model_calls=0, score_calls=0, bootstrap_calls=0, TEST_QA=0,
               consumer_seconds_before_write=time.monotonic()-tick)
    with target.open('x', encoding='utf8') as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    print(json.dumps(dict(OUTCOME_sha256=sha(target), primary_point_pass=out['primary_point_pass'],
                         rows={k: dict(macro=v['macro'], domains=v['domains']) for k, v in rows.items()},
                         decision=out['decision']), ensure_ascii=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())
