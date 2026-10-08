"""Freeze two medical generation tasks using upstream text, no image downloads."""
import csv
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from common import ROOT, digest_bytes, fingerprint, normalize_input, write_json

PROTOCOL = 'medical_generation2_after_superni7_v1'
TASKS = ['medical_mts_dialog_note', 'medical_iu_xray_impression']
RAW = ROOT / 'data/medical_raw'
OUT = ROOT / 'data/prepared' / PROTOCOL

def ordered(rows):
    return sorted(rows, key=lambda x: fingerprint(['medical_split_seed42', x['instance_id']]))

def example(task, identifier, text, answer, instruction):
    return dict(task_id=task, instance_id=identifier,
                prompt=f'Instruction: {instruction}\n\nInput: {text.strip()}\n\nResponse:\n',
                references=[answer.strip()])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = {}
    audit = {'sources': {}, 'exclusions': {}, 'policy': 'Exact normalized input deduplication; upstream MTS split membership retained; IU report-level hash split after deduplication.'}
    seen = set()
    mts = {}
    # Give official held-out examples priority when removing duplicate conversations.
    for split in ['test', 'dev', 'train']:
        path = RAW / 'mts_dialog' / f'{split}.csv'
        audit['sources'][str(path.relative_to(ROOT))] = digest_bytes(path.read_bytes())
        rows, excluded = [], []
        for row in csv.DictReader(path.open()):
            text, answer = row['dialogue'].strip(), row['section_text'].strip()
            key = normalize_input(text).casefold()
            identifier = f'mts_{split}_{row["ID"]}'
            if not text or not answer or key in seen:
                excluded.append(identifier)
                continue
            seen.add(key)
            rows.append(example(TASKS[0], identifier, text, answer,
                                'Summarize this doctor-patient conversation as a clinical note section. Preserve the stated clinical facts.'))
        audit['exclusions']['mts_'+split] = excluded
        mts[split] = ordered(rows)[:{'train': 1000, 'dev': 100, 'test': 200}[split]]
    datasets[TASKS[0]] = mts
    archive = RAW / 'iu_xray/NLMCXR_reports.tgz'
    audit['sources'][str(archive.relative_to(ROOT))] = digest_bytes(archive.read_bytes())
    rows, seen, excluded = [], set(), []
    with tarfile.open(archive) as handle:
        for member in sorted(handle.getmembers(), key=lambda m: m.name):
            if not member.isfile() or not member.name.endswith('.xml'):
                continue
            tree = ET.fromstring(handle.extractfile(member).read())
            fields = {n.attrib.get('Label'): ''.join(n.itertext()).strip() for n in tree.findall('.//AbstractText')}
            text, answer = fields.get('FINDINGS', ''), fields.get('IMPRESSION', '')
            identifier = tree.find('uId').attrib['id']
            key = normalize_input(text).casefold()
            if not text or not answer or key in seen:
                excluded.append(identifier)
                continue
            seen.add(key)
            rows.append(example(TASKS[1], identifier, text, answer,
                                'Write the impression section of a chest radiology report from its findings. Preserve the stated abnormalities and negations.'))
    rows = ordered(rows)
    if len(rows) < 1300:
        raise ValueError('Not enough distinct IU reports')
    datasets[TASKS[1]] = {'train': rows[:1000], 'dev': rows[1000:1100], 'test': rows[1100:1300]}
    audit['exclusions']['iu'] = excluded
    audit['iu_distinct_valid'] = len(rows)
    entries = []
    for task, splits in datasets.items():
        entry = {'task_id': task, 'splits': {}}
        for split, rows in splits.items():
            path = OUT / f'{task}.{split}.jsonl'
            path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
            entry['splits'][split] = {'path': str(path.relative_to(ROOT)), 'count': len(rows), 'sha256': digest_bytes(path.read_bytes())}
        entries.append(entry)
    write_json(OUT / 'audit.json', audit)
    write_json(ROOT / 'rules/003_medical_manifest.json', {'protocol': PROTOCOL, 'tasks': entries,
        'sources': {'mts_dialog': 'https://github.com/abachaa/MTS-Dialog',
                    'iu_xray': 'https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz'},
        'audit': str((OUT/'audit.json').relative_to(ROOT)),
        'split_note': 'MTS official train/dev/TestSet-1; deterministic training cap. IU custom report-level split with exact findings deduplication; patient identifiers are not available.'})
    print(json.dumps({t: {s: len(r) for s,r in splits.items()} for t,splits in datasets.items()}, indent=2))

if __name__ == '__main__':
    main()
