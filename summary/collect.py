import argparse
import csv
import fcntl
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['run', 'protocol', 'evaluation_protocol', 'method', 'order', 'seed', 'epochs', 'status', 'AP', 'F.Rate', 'FWT', 'BWT',
          'config_sha256', 'model_fingerprint']


def collect(result_root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / '.collect.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _collect(result_root, destination)


def _collect(result_root, destination):
    rows = []
    for path in sorted(result_root.rglob('status.json')):
        config_path = path.parent / 'config.json'
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        if config.get('smoke') or 'job' in config:
            continue
        status = json.loads(path.read_text())
        metrics_path = path.parent / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        if status['status'] != 'completed':
            metrics = {}
        row = {'run': str(path.parent.relative_to(result_root)), 'protocol': config['protocol'],
               'evaluation_protocol': metrics.get('evaluation_protocol', config.get('evaluation_protocol', 'legacy_sapt_fwt')),
               'method': config['method'], 'order': config['order'], 'seed': config['seed'],
               'epochs': config['epochs'], 'status': status['status']}
        row.update({key: metrics.get(key) for key in FIELDS if key not in row})
        rows.append(row)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    visible = FIELDS[:12]
    lines = ['# Experiment results', '', 'Each row is a separate attempt, not a multi-seed aggregate. Smoke runs are excluded.',
             'Compare only runs with matching evaluation protocols, experimental settings and model fingerprints. Missing metrics are N/A.', '',
             '| ' + ' | '.join(visible) + ' |', '| ' + ' | '.join(['---'] * len(visible)) + ' |']
    for row in rows:
        values = []
        for key in visible:
            value = row.get(key)
            values.append('N/A' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    if not rows:
        lines.extend(['', 'No experiment attempts recorded yet.'])
    (destination / 'results.md').write_text('\n'.join(lines) + '\n')
    print(f'Collected {len(rows)} attempts into {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-root', type=Path, default=ROOT / 'exp/result')
    parser.add_argument('--output', type=Path, default=ROOT / 'summary')
    args = parser.parse_args()
    collect(args.result_root, args.output)
