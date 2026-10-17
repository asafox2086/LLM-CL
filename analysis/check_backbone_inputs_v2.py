"""Hash every used image and capture the actual common renderer before GPU IO."""
import hashlib
import json
import sys
from pathlib import Path
import PIL

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'exp/two_backbone_native_cl_20261017_v2'
sys.path.insert(0, str(EXP))
from run import load_protocol
from runtime_utils import input_batch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


class Captured(Exception):
    pass


class CaptureProcessor:
    def apply_chat_template(self, messages, **kwargs):
        self.schema_sha256 = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        return 'capture-only'

    def __call__(self, text, images, **kwargs):
        self.images = [{'size': list(image.size), 'RGB_sha256': hashlib.sha256(image.tobytes()).hexdigest()}
                       for image in images]
        assert all(s['size'][0] * s['size'][1] <= 401408 for s in self.images)
        raise Captured()


def main():
    _, manifest, _, data = load_protocol('pct01')
    images = sorted({name for rows in data.values() for row in rows for name in row['images']})
    hashes = {name: sha(ROOT / name) for name in images}
    raw_map = json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()
    captures = []
    for task in manifest['tasks']:
        row = data[task['task'], 'train'][0]
        capture = CaptureProcessor()
        try:
            input_batch(row, task, capture, 'cpu')
        except Captured:
            pass
        else:
            raise AssertionError('Capture unexpectedly entered model execution')
        captures.append({'task': task['task'], 'id': row['id'], 'images': capture.images,
                         'prompt_schema_sha256': capture.schema_sha256})
    report = ROOT / 'summary/two_backbone_native_cl_20261017_v2'
    report.mkdir(exist_ok=True)
    (report / 'input_images_hashes_runtime.json').write_bytes(raw_map)
    proof = {'used_image_paths': len(images), 'used_image_content_map_sha256': hashlib.sha256(raw_map).hexdigest(),
             'QA': sum(map(len, data.values())), 'Pillow_version': PIL.__version__,
             'actual_common_renderer19': captures,
             'sources_sha256': {p.name: sha(p) for p in EXP.glob('*.py')},
             'model_or_GPU_not_used': True, 'source_images_not_modified': True}
    (report / 'input_CPU_validation.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'used_image_paths': len(images), 'Pillow': PIL.__version__,
                      'content_map_sha256': proof['used_image_content_map_sha256']}))


if __name__ == '__main__':
    main()
