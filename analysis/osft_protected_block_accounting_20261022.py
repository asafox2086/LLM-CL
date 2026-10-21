"""Integer accounting from sealed matrix shapes, not a model experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'summary/edge_deployment_20261019/cpu_update_space_20261020/v1/EVIDENCE.json'
OUT = ROOT/'summary/edge_deployment_20261019/update_geometry_literature_20261022/PROTECTED_BLOCK_COUNTS.json'


def main():
    evidence = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = evidence['OSFT_layers']
    assert len(rows) == 168
    totals = dict(raw_coordinates=0, constraint_normal_coordinates=0,
                  constrained_factor_coordinates=0, factor_redundancy=0, full_core_coordinates=0)
    groups = {}
    for row in rows:
        m, n = row['shape']
        h = 3*min(m, n)//4
        k = min(m, n)-h
        p, q = m-h, n-h
        assert min(p, q) == k
        raw, normal, core = k*(m+n+1), 2*h*k, p*q
        constrained = raw-normal
        redundant = constrained-core
        assert redundant == k*k+k
        assert raw == row['trainable_coordinates']
        assert core == row['protected_local_dimension']
        values = dict(raw_coordinates=raw, constraint_normal_coordinates=normal,
                      constrained_factor_coordinates=constrained,
                      factor_redundancy=redundant, full_core_coordinates=core)
        positions = {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
        matches = positions.intersection(row['name'].split('.'))
        assert len(matches) == 1
        name = matches.pop()
        group = groups.setdefault(name, dict(m=m, n=n, preserved_rank=h, trainable_rank=k,
                                            core_shape=[p,q], layers=0, **{key:0 for key in totals}))
        assert (group['m'], group['n']) == (m,n)
        group['layers'] += 1
        for key, value in values.items():
            totals[key] += value
            group[key] += value
    assert totals['raw_coordinates'] == evidence['OSFT_trainable_coordinates'] == 113766144
    assert totals['full_core_coordinates'] == evidence['OSFT_final_protected_local_dimension'] == 71245824
    assert totals['constraint_normal_coordinates']+totals['factor_redundancy'] == 42520320
    result = dict(kind='reused_shapes_and_exact_integer_accounting_not_empirical_dimension_measurement',
        source=SOURCE.relative_to(ROOT).as_posix(), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        old_two_totals_are_not_new_findings=True, totals=totals, by_position=groups,
        assumptions=['fixed_stage_high_bases', 'exact_arithmetic_orthogonality', 'full_complements_not_arbitrary_rank16'],
        new_model_forward=0, new_backward=0, new_optimizer_steps=0, GPU_calls=0,
        no_checkpoint_or_QA_read=True, no_F1_or_RAM_or_speed_claim=True,
        initial_readonly_grouping_error='First attempt grouped by penultimate name segment, stopped before output at mixed-shape assertion; fixed by explicit seven-position names; no model was run')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
