"""Pin and audit reference source code without executing external code."""
import hashlib
import json
import subprocess
from pathlib import Path

exp = Path(__file__).resolve().parent
manifest = json.loads((exp / 'baselines.json').read_text())
records = []
for baseline in manifest['added_baselines']:
    target = exp / 'reference_sources' / baseline['method']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(['git', 'clone', '--depth', '1', baseline['repository'], str(target)], check=True)
    revision = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != baseline['revision']:
        subprocess.run(['git', '-C', str(target), 'fetch', 'origin', baseline['revision']], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', baseline['revision']], check=True)
    files = []
    for path in sorted(target.rglob('*')):
        if not path.is_file() or '.git' in path.relative_to(target).parts:
            continue
        files.append({'path': str(path.relative_to(target)), 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    records.append({'method': baseline['method'], 'revision': baseline['revision'], 'files': files,
                    'python_files': sum(p['path'].endswith('.py') for p in files)})
(exp / 'source_audit.json').write_text(json.dumps(records, indent=2))
print(json.dumps([{'method': r['method'], 'revision': r['revision'],
                  'files': len(r['files']), 'python_files': r['python_files']} for r in records], indent=2))
