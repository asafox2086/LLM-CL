"""Plot sealed five-window resource measurements; no model or rescoring."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'summary/edge_resource_comparison_20261025/v1'
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
LABELS = ('TextVQA', 'SLAKE', 'RSVQA-LR', 'FinMME', 'CODA-LM')
EXPECTED = {
    'summary/plain_r8_gpu_control_20261025/v1/VERIFIED.json': '3b172989a57817cb99dbc4f70649efa0193ca5ee94ba6f5d80c1b274f5e3bc66',
    'summary/strong_pas_resource_20261025/v1/VERIFIED.json': 'c987ec2ac68980944a1ebdc515fab08a792667f923d142f6f8489d3725753d08',
    'summary/strong_qwen_resource_20261025/v1/VERIFIED.json': 'dd97300219ecbfaeb5052c571c68a4fd61fc3dd8455ab24eff5aa8627ca05e7c',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    # Local isolated plotting dependencies only; never initialize a model.
    local = ROOT / '.deployment/figure_runtime_20261025'
    if importlib.util.find_spec('matplotlib') is None and local.is_dir():
        sys.path.insert(0, str(local))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    sys.path.insert(0, str(OUT))
    from style import METHOD_COLORS, darken_color, polish_axes, save_png_pdf, setup_style

    for name, digest in EXPECTED.items():
        assert sha(ROOT / name) == digest, name
    naive, pas, qwen = [read(ROOT / name) for name in EXPECTED]
    assert all(v['complete'] for v in (naive, pas, qwen))
    pr = ROOT / 'summary/strong_pas_resource_20261025/v1'
    qr = ROOT / 'summary/strong_qwen_resource_20261025/v1'
    assert sha(pr / 'memory.json') == pas['memory_sha256']
    assert sha(pr / 'COMPLETE.json') == pas['COMPLETE_sha256']
    assert sha(qr / 'memory.json') == qwen['memory_sha256']
    assert sha(qr / 'COMPLETE.json') == qwen['COMPLETE_sha256']
    assert read(qr / 'memory.json') == qwen['memory']
    assert pas['physical_GPU'] == qwen['physical_GPU'] == 1
    assert pas['GPU_uuid'] == qwen['GPU_uuid']
    nr = ROOT / 'summary/plain_r8_gpu_control_20261025/v1'
    ns = read(nr / 'eval_qv_STARTED.json')
    assert ns['GPU'] == 1
    nc = read(nr / 'eval_qv_COMPLETE.json')
    assert sha(nr / 'eval_qv_COMPLETE.json') == naive['source_COMPLETE_sha256']['eval_qv']
    assert nc['identity'] == ns['identity']
    assert ns['boot'] == read(pr / 'STARTED.json')['boot'] == read(qr / 'STARTED.json')['boot']
    windows = (naive['memory']['plain_r8'], read(pr / 'memory.json'), qwen['memory'])
    for rows in windows:
        assert len(rows) == 5 and tuple(x['domain'] for x in rows) == DOMAINS
        for row in rows:
            assert row['new_tokens'] == 32 and row['common_image_area_cap'] == 401408
            assert row['peak_allocated'] > 0 and not row['visual_cache_hit']
    common = ('id', 'domain', 'raw_input_sha256', 'image_SHA256', 'image_shapes',
              'common_image_area_cap', 'new_tokens')
    for triple in zip(*windows):
        for key in common:
            assert triple[0][key] == triple[1][key] == triple[2][key], key
        # Same-backbone prepared tensors also match. Across backbones they may differ.
        assert triple[0]['prepared_sha256'] == triple[1]['prepared_sha256']
    values = np.array([[x['peak_allocated'] / 1048576 for x in rows] for rows in windows])
    assert values.shape == (3, 5) and np.isfinite(values).all()
    labels = ('LLaVA plain-r8 (folded)', 'LLaVA PAS (original)', 'Qwen SeqLoRA (original)')
    colors = tuple(METHOD_COLORS[k] for k in ('Method A', 'Method B', 'Method C'))
    provenance = {
        'source_proofs': EXPECTED,
        'extra_bindings': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in
            (pr / 'memory.json', qr / 'memory.json', nr / 'eval_qv_STARTED.json', nr / 'eval_qv_COMPLETE.json')},
        'physical_GPU': 1, 'GPU_uuid': pas['GPU_uuid'], 'metric': 'torch.cuda.max_memory_allocated',
        'unit': 'MiB (1048576 bytes)', 'domain_order': DOMAINS, 'dataset_labels': LABELS,
        'raw_IDs': [x['id'] for x in windows[0]],
        'raw_input_sha256': [x['raw_input_sha256'] for x in windows[0]],
        'new_tokens_each': 32, 'samples_per_dataset': 1, 'repeated_runs': 1,
        'series': [dict(label=label, color=color, peak_MiB=row.tolist(),
                        visual_tokens=[x['visual_tokens'] for x in source],
                        prompt_tokens=[x['prompt_tokens'] for x in source])
                   for label, color, row, source in zip(labels, colors, values, windows)],
        'naive_reduction_vs_PAS_percent': (100 * (1-values[0]/values[1])).tolist(),
        'naive_reduction_vs_Qwen_percent': (100 * (1-values[0]/values[2])).tolist(),
        'interpretation': 'Original implementation comparison on five fixed raw-QA windows, including input preparation. Native expanded inputs differ across backbones. Plain-r8 is folded FP16; PAS and Qwen preserve their original mixed-dtype, unfolded representations. Not optimal deployment, full-distribution peak, a domain average, a new quality result, or evidence that ARC succeeds.',
        'template': 'analysis/omnimed_20261009/plot_templates/grouped_bar_template.py',
        'style': 'analysis/omnimed_20261009/plot_templates/style.py',
        'style_sha256': sha(OUT / 'style.py'),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'plot_data.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf8')
    setup_style('bar')
    fig, ax = plt.subplots(figsize=(11.6, 6.4), dpi=300)
    x = np.arange(5); width = .78 / 3
    for index, (label, color, row) in enumerate(zip(labels, colors, values)):
        bars = ax.bar(x+(index-1)*width, row, width, label=label, color=color,
                      edgecolor=darken_color(color, .65), linewidth=1.4, zorder=3)
        ax.bar_label(bars, fmt='%.0f', padding=4, fontsize=10)
    ax.set_xticks(x, LABELS, rotation=12)
    ax.set_ylabel('Peak GPU allocated memory (MiB)')
    ax.set_ylim(0, 5650)
    ax.set_yticks(np.arange(0, 6000, 1000))
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.16), ncol=3,
              frameon=False, columnspacing=1.0, handlelength=1.2)
    polish_axes(ax, y_grid=True, x_grid=False)
    fig.text(.5, .98, 'Five fixed raw-QA windows; 32 generated tokens per window',
             ha='center', va='top', fontsize=15)
    fig.text(.08, .105,
             'Same physical GPU; native input expansion and original dtypes. Includes input preparation.\n'
             'Plain-r8: folded FP16. PAS / Qwen: original, unfolded mixed-dtype graphs.\n'
             'One QA per dataset; no uncertainty estimates. Not an optimal-deployment or full-test peak.',
             ha='left', va='top', fontsize=10, linespacing=1.5)
    fig.subplots_adjust(left=.10, right=.985, bottom=.24, top=.83)
    save_png_pdf(fig, str(OUT / 'fixed_window_memory'), dpi=300)
    plt.close(fig)
    print(json.dumps(dict(output=str(OUT), peak_MiB=values.tolist(), new_models=0,
                          new_scores=0, PNG_sha256=sha(OUT/'fixed_window_memory.png'),
                          PDF_sha256=sha(OUT/'fixed_window_memory.pdf'))))


if __name__ == '__main__':
    main()
