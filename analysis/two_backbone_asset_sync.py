"""LAN-only asset provisioning; no GPU allocation or training changes."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = '/data2/liyapeng_grp/program/LLMCL'
REPORT = ROOT / 'summary/two_backbone_pct01_20261017'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-i',
       str(ROOT / '.deployment/old_pct01_id_ed25519'), '-p', '8666']
HOST = 'liyapeng_grp@10.254.29.8'
MODEL = 'model/LLaVA-OneVision-Qwen2-0.5B-OV'


def write(name, value):
    p = REPORT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(p)


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def main():
    write('asset_sync_status.json', {'phase': 'copying_pinned_OneVision_model_LAN'})
    expected = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / MODEL).iterdir() if p.is_file()}
    base = ['rsync', '-a', '--partial', '-e', ' '.join(SSH)]
    subprocess.run(base + [str(ROOT / MODEL) + '/', HOST + ':' + OLD + '/' + MODEL + '/'], check=True)
    code = r'''
from pathlib import Path
import json,hashlib
R=Path('/data2/liyapeng_grp/program/LLMCL');expected=EXPECTED
for rel,h in expected.items():
 s=hashlib.sha256()
 with (R/rel).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):s.update(b)
 assert s.hexdigest()==h,rel
print(json.dumps({'passed':True,'files':len(expected),'sha256':expected}))
'''.replace('EXPECTED', repr(expected))
    p = subprocess.run(SSH + [HOST, OLD + '/.vision-env/bin/python -'], input=code,
                       text=True, capture_output=True, check=True)
    write('old_OneVision_model_integrity.json', json.loads(p.stdout))
    write('asset_sync_status.json', {'phase': 'OneVision_model_verified_copying_full_missing_images_LAN'})
    file_list = ROOT / 'summary/five_domain_native_cl_20261016_sparse/full_missing_old_images.txt'
    subprocess.run(base + ['--files-from=' + str(file_list), str(ROOT) + '/', HOST + ':' + OLD + '/'], check=True)
    write('asset_sync_status.json', {'phase': 'copies_complete_model_SHA_verified_full_image_audit_pending',
                                    'model_files': len(expected), 'full_missing_paths_transferred': len(file_list.read_text().splitlines())})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write('asset_sync_status.json', {'phase': 'needs_attention', 'error': repr(exc)})
        raise
