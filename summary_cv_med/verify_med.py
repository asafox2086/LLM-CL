"""Audit medical dataset isolation, immutable old code, model and current saved scores."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'exp'))
import torch
from common import digest_bytes, fingerprint, write_json
from cv_metrics_med import letter_correct

def verify():
    cache=ROOT/'exp/CV_result_med/shared';manifest=json.loads((cache/'manifest.json').read_text())
    assert digest_bytes((cache/'data.pt').read_bytes())==manifest['data_sha256']
    data=torch.load(cache/'data.pt',weights_only=True,map_location='cpu')
    text_tasks=list(data)[:2];image_tasks=list(data)[2:]
    question_sets=[];image_sets=[];case_sets=[]
    key=lambda r:' '.join(r['question'].lower().split())
    probes=[json.loads(x) for x in (ROOT/'data/knowledge_probe/probes.jsonl').read_text().splitlines()]
    for split in ['train','dev','test']:
        rows=[r for t in text_tasks for r in data[t][split]]
        questions={key(r) for r in rows};assert len(questions)==len(rows)
        assert not questions&{key(r) for r in probes}
        question_sets.append(questions)
        images=[r for t in image_tasks for r in data[t][split]]
        image_sets.append({r['image'] for r in images})
        case_sets.append({r['case_url'] for r in images if r.get('case_url')})
    for sets in [question_sets,image_sets,case_sets]:
        assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    for rows in data['medical_medqa'].values():
        for r in rows:
            if r.get('question_shortened'):
                assert all(choice in r['prompt'] for choice in r['choices'])
                assert r['original_prompt'] != r['prompt']
    for name,sha in manifest['feature_sha256'].items():
        assert digest_bytes((cache/'features'/name).read_bytes())==sha
    assert letter_correct('A','A') and letter_correct('b)','B')
    assert not letter_correct('The','A') and not letter_correct('A because ...','A')
    model=ROOT/'model/Qwen2-VL-2B-Instruct'
    for entry in json.loads((model/'download_manifest.json').read_text())['files']:
        h=hashlib.sha256()
        with (model/entry['Path']).open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==entry['Sha256']
    methods=['seq_lora','migu_lora','olora','sapt_lora']
    old_run=Path(json.loads((ROOT/'exp/CV_result/latest.json').read_text())['run'])
    old_config=json.loads((old_run/'seq_lora/config.json').read_text())
    for path,sha in old_config['implementation'].items():assert digest_bytes((ROOT/path).read_bytes())==sha,path
    latest=json.loads((ROOT/'exp/CV_result_med/latest.json').read_text());run=Path(latest['run'])
    records=0;baseline={};checks={}
    for method in methods:
        recovery=json.loads((cache.parent/f'validation/recovery/{method}.json').read_text());assert recovery['passed']
        preflight=json.loads((cache.parent/f'validation/{method}/two_updates/passed.json').read_text())
        checks[method]={'recovery':True,'preflight':True,'peak_mib':preflight['peak_bytes']/2**20}
        output=run/method;cfg=json.loads((output/'config.json').read_text())
        assert cfg['data_manifest_sha256']==fingerprint(manifest)
        for path,sha in cfg['implementation'].items():assert digest_bytes((ROOT/path).read_bytes())==sha
        baseline[method]={}
        for p in sorted((output/'predictions').glob('stage_*/*.jsonl')):
            stage=int(p.parent.name.split('_')[-1]);task=p.stem
            rows=[json.loads(x) for x in p.read_text().splitlines()]
            assert len(rows)==len(data[task]['test'])
            for r,source in zip(rows,data[task]['test']):
                assert r['instance_id']==source['instance_id'] and r['references']==source['references']
                if task in text_tasks:assert r['scores']['exact_match']==100*letter_correct(r['prediction'],r['references'][0])
            if stage==0:baseline[method][task]=[r['prediction'] for r in rows]
            records+=len(rows)
    for task in data:
        answers=[b[task] for b in baseline.values() if task in b]
        assert all(a==answers[0] for a in answers), f'Base generation differs: {task}'
    checks['sapt_reflection']=json.loads((cache.parent/'validation/sapt_lora/two_updates/reflection_passed.json').read_text())
    result={'passed':True,'run':str(run),'data_sha256':manifest['data_sha256'],'counts':manifest['counts'],
            'checks':['exact QA disjoint across datasets and splits','external probes excluded from training/dev/test QA',
                      'image/case split disjoint','long MCQ preserves all options','all cached features SHA match',
                      'official model file SHA match','old implementation files unchanged','four-method optimizer-step exact recovery',
                      'four-method GPU training/vision preflight','saved predictions and strict letter scores audited',
                      'shared available base predictions identical'],
            'methods':checks,'current_saved_test_records':records,
            'scope':'startup/interim audit; does not assert experiments have finished'}
    write_json(ROOT/'summary_cv_med/verification.json',result);print(json.dumps({'passed':True,'records':records,'counts':manifest['counts']}))

if __name__=='__main__':verify()
