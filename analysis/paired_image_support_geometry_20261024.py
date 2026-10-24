"""Freeze label-blind same-task image derangements; no processor or model."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import sys
import time

SEED='20261024:paired-image-support-v1'
REPORT='summary/final_weight_reduction_20261024/paired_image_support_diagnostic_v1'
SCREEN='summary/edge_deployment_20261019/osft_increment_screen/v1'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def order(value):
    return hashlib.sha256((SEED+':'+value).encode()).digest()

def match(ids,records):
    assigned={}
    def augment(target,seen):
        a=records[target]
        for donor in sorted(ids,key=lambda d:order(target+'->'+d)):
            b=records[donor]
            if donor in seen or a['group']==b['group'] or set(a['image_sha256']) & set(b['image_sha256']):continue
            seen.add(donor)
            if donor not in assigned or augment(assigned[donor],seen):
                assigned[donor]=target
                return True
        return False
    for target in sorted(ids,key=order):
        if not augment(target,set()):return None
    result={target:donor for donor,target in assigned.items()}
    assert set(result)==set(result.values())==set(ids)
    return result

def run(root):
    from PIL import Image, __version__ as pillow_version
    begin=time.perf_counter();report=root/REPORT
    assert 'torch' not in sys.modules and not (report/'PAIRING.json').exists()
    admission=json.loads((report/'ADMISSION.json').read_bytes())
    assert admission['decision']=='accept_single_label_blind_geometry_pairing_only'
    assert sha(Path(__file__))==admission['consumer_sha256']
    assert sha(report/'PROTOCOL.md')==admission['protocol_sha256']
    ready=json.loads((root/SCREEN/'READY.json').read_bytes())
    split_path=root/SCREEN/'SPLIT.json';assert sha(split_path)==ready['split_sha256']
    split=json.loads(split_path.read_bytes());groups={}
    for gid,group in split['groups'].items():
        if group['dev']:
            for qid in group['ids']:assert qid not in groups;groups[qid]=gid
    policy=json.loads((root/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json').read_bytes())
    info=policy['conditions']['pct01'];manifest_path=root/info['manifest']
    assert sha(manifest_path)==info['manifest_sha256']==ready['manifest_sha256']
    manifest=json.loads(manifest_path.read_bytes())
    images={};records={};buckets=defaultdict(list);source_files=[]
    for task in manifest['tasks']:
        wanted=set(split['dev_ids'][str(task['task'])])
        if not wanted:continue
        source=task['files']['train'];path=root/source['path'];assert sha(path)==source['sha256']
        source_files.append(dict(path=source['path'],sha256=source['sha256']))
        found=set()
        for line in path.read_bytes().splitlines():
            row=json.loads(line);qid=row['id']
            if qid not in wanted:continue
            assert qid not in found;found.add(qid)
            rels=row.get('images',[]);assert rels
            for rel in rels:
                if rel in images:continue
                relative=Path(rel);assert not relative.is_absolute() and '..' not in relative.parts
                path_image=root/rel;digest=sha(path_image);assert digest==split['image_sha256'][rel]
                with Image.open(path_image) as image:
                    width,height=image.size
                assert width>0 and height>0
                common=(width,height)
                if width*height>401408:
                    ratio=math.sqrt(401408/(width*height))
                    common=(max(1,int(width*ratio)),max(1,int(height*ratio)))
                assert common[0]*common[1]<=401408
                images[rel]=dict(sha256=digest,original_size=[width,height],common_size=list(common),bytes=path_image.stat().st_size)
            record=dict(id=qid,task=task['task'],domain=task['domain'],group=groups[qid],
                image_sha256=[images[p]['sha256'] for p in rels],
                common_sizes=[images[p]['common_size'] for p in rels])
            records[qid]=record
            signature=(task['task'],len(rels),tuple(tuple(images[p]['common_size']) for p in rels))
            buckets[signature].append(qid)
        assert found==wanted
    assert len(records)==100 and set(records)==set(groups)
    pairing={};excluded=[]
    for signature,ids in sorted(buckets.items()):
        mapping=match(ids,records) if len(ids)>1 else None
        if mapping is None:
            excluded.append(dict(task=signature[0],common_sizes=signature[2],ids=sorted(ids),reason='no_complete_derangement_in_fixed_geometry_bucket'))
        else:pairing.update(mapping)
    assert Counter(pairing.keys())==Counter(pairing.values())
    coverage={}
    for domain in split['domains_dev_QA']:
        ids=[qid for qid in pairing if records[qid]['domain']==domain]
        coverage[domain]=dict(QA=len(ids),known_groups=len({records[qid]['group'] for qid in ids}),tasks=sorted({records[qid]['task'] for qid in ids}))
    gate=(coverage['medical']['QA']>=8 and len(coverage['medical']['tasks'])>=2 and
          sum(v['QA']>=8 for d,v in coverage.items() if d!='medical')>=2)
    result=dict(scope='One label-blind fixed-DEV geometry pairing; not actual packed-layout/model verification.',
        seed=SEED,consumer_sha256=sha(Path(__file__)),protocol_sha256=admission['protocol_sha256'],
        admission_sha256=sha(report/'ADMISSION.json'),manifest_sha256=info['manifest_sha256'],
        split_sha256=ready['split_sha256'],source_files=source_files,
        input_batch_source_sha256=sha(root/'exp/two_backbone_native_cl_20261017_v2/runtime_utils.py'),
        interpreter=sys.executable,pillow_version=pillow_version,
        QA_total=100,paired_QA=len(pairing),coverage=coverage,coverage_investment_gate_passed=gate,
        rows=[dict(records[qid],donor_id=pairing.get(qid)) for qid in sorted(records)],excluded_buckets=excluded,
        unique_image_headers_read=len(images),image_file_bytes_hashed=sum(v['bytes'] for v in images.values()),
        RGB_decodes=0,processor_calls=0,actual_packed_layout_equality_verified=False,
        references_or_model_outputs_used_for_selection=False,QA_forward_exposures=0,
        TEST_rows_opened=0,model_requests=0,new_scores=0,training_updates=0,model_admitted=False,
        seconds=time.perf_counter()-begin)
    assert 'torch' not in sys.modules
    with (report/'PAIRING.json').open('x',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':run(Path(sys.argv[1]))
