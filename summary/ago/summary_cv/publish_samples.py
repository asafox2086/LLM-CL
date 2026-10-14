"""Copy only report-selected photographs, preserving source bytes and hashes."""
import hashlib
import shutil
from pathlib import Path
from common import ROOT, write_json

def publish_samples(examples):
    folder = ROOT / 'sample'
    folder.mkdir(exist_ok=True)
    manifest = {}
    for example in examples:
        row = example['source']
        if not row.get('image'):
            continue
        source = ROOT / 'CV_data' / row['image']
        name = example['task_id'] + '_' + source.name
        destination = folder / name
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        assert sha == row['image_metadata']['sha256']
        if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != sha:
            shutil.copy2(source,destination)
        example['display_image'] = 'sample/' + name
        entry = manifest.setdefault(name, {'source': 'CV_data/' + row['image'], 'sha256':sha,
            'bytes':destination.stat().st_size, 'examples':[]})
        entry['examples'].append({'title':example['title'], 'instance_id':example['instance_id']})
    write_json(folder/'image_manifest.json', manifest)
    return manifest
