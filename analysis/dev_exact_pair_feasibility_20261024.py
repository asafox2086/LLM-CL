"""Check whether fixed DEV permits a narrow input/answer discrimination probe.

No model, image payload, predictions, score calculation, or training statistics.
Reference string disjointness is not semantic mutual exclusion.
"""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib
import json
import sys
import time


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def run(root, output, private):
    started = time.perf_counter()
    assert not output.exists() and not private.exists()
    screen = root/'summary/edge_deployment_20261019/osft_increment_screen/v1'
    ready = json.loads((screen/'READY.json').read_bytes())
    split_raw = (screen/'SPLIT.json').read_bytes()
    assert hashlib.sha256(split_raw).hexdigest() == ready['split_sha256']
    split = json.loads(split_raw)
    policy = json.loads((root/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json').read_bytes())
    anchor = policy['conditions']['pct01']
    manifest_path = root/anchor['manifest']
    manifest_raw = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == anchor['manifest_sha256'] == ready['manifest_sha256']
    manifest = json.loads(manifest_raw)
    assert len(manifest['tasks']) == 19 and sum(map(len, split['dev_ids'].values())) == 100
    membership = {}
    for gid, group in split['groups'].items():
        if group['dev']:
            for qid in group['ids']:
                assert qid not in membership
                membership[qid] = gid
    assert len(membership) == 100
    task_results, candidates, input_files = [], [], []
    for task in manifest['tasks']:
        tid = str(task['task'])
        wanted = set(split['dev_ids'][tid])
        if not wanted:
            task_results.append(dict(task=int(tid), domain=task['domain'], DEV_QA=0, candidate_pairs=0, covered_QA=0))
            continue
        info = task['files']['train']
        raw = (root/info['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == info['sha256']
        input_files.append(dict(path=info['path'], sha256=info['sha256']))
        rows = {}
        # TRAIN containers include DEV: only the selected IDs' content is used.
        for line in raw.splitlines():
            item = json.loads(line)
            if item['id'] in wanted:
                assert item['id'] not in rows
                rows[item['id']] = item
        assert set(rows) == wanted
        templates = defaultdict(list)
        excluded = Counter()
        for qid in sorted(rows):
            r = rows[qid]
            if not r.get('images') or any(p not in split['image_sha256'] for p in r['images']):
                excluded['missing_image_or_SHA'] += 1
                continue
            refs = r.get('answers')
            if not isinstance(refs, list) or not refs or any(not isinstance(s, str) or not s.strip() for s in refs):
                excluded['missing_or_empty_references'] += 1
                continue
            # runtime_utils.input_batch uses task instruction, context, question,
            # and one image marker per image; row metadata options are not added.
            templates[(task['instruction'], r['dataset'], r['question'], r.get('context', ''), len(r['images']))].append(qid)
        covered = set()
        task_pairs = []
        for template, ids in templates.items():
            for first, second in combinations(ids, 2):
                a, b = rows[first], rows[second]
                refs = [r['answers'] for r in (a, b)]
                assert all(isinstance(v, list) and all(isinstance(s, str) for s in v) for v in refs)
                if any(not v or any(not s.strip() for s in v) for v in refs):
                    continue
                image_sets = [{split['image_sha256'][p] for p in r['images']} for r in (a, b)]
                if membership[first] == membership[second] or not all(image_sets) or image_sets[0] & image_sets[1]:
                    continue
                if set(refs[0]) & set(refs[1]):
                    continue
                record = dict(task=int(tid), domain=task['domain'], ids=[first, second],
                    groups=[membership[first], membership[second]], template_sha256=digest(template),
                    references_sha256=[digest(v) for v in refs], image_sha256=[sorted(v) for v in image_sets],
                    semantic_mutual_exclusion_verified=False)
                task_pairs.append(record)
                candidates.append(dict(public=record, question=a['question'], context=a.get('context', ''),
                    dataset=a['dataset'], references=refs, metadata=[r.get('metadata', {}) for r in (a, b)]))
                covered.update((first, second))
        task_results.append(dict(task=int(tid), domain=task['domain'], DEV_QA=len(rows),
            exact_template_groups=len(templates), repeated_template_groups=sum(len(v)>1 for v in templates.values()),
            candidate_pairs=len(task_pairs), covered_QA=len(covered), excluded_QA=dict(excluded)))
    assert sum(t['DEV_QA'] for t in task_results) == 100
    assert 'torch' not in sys.modules
    private.parent.mkdir(parents=True, exist_ok=True)
    with private.open('x', encoding='utf8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
        f.write('\n')
    result = dict(scope='One fixed 100-DEV exact-pair feasibility check; no model admission.',
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        screen_READY_sha256=hashlib.sha256((screen/'READY.json').read_bytes()).hexdigest(),
        current_input_batch_source_sha256=hashlib.sha256((root/'exp/two_backbone_native_cl_20261017_v2/runtime_utils.py').read_bytes()).hexdigest(),
        split_sha256=ready['split_sha256'], manifest_sha256=anchor['manifest_sha256'], input_files=input_files,
        tasks=task_results, candidate_pairs=[r['public'] for r in candidates],
        total_candidate_pairs=len(candidates),
        covered_DEV_QA=len({qid for r in candidates for qid in r['public']['ids']}),
        private_candidate_file_sha256=hashlib.sha256(private.read_bytes()).hexdigest(),
        semantic_mutual_exclusion_verified=False, model_admitted=False,
        label_used_for_diagnostic_pair_construction=True, pairs_are_not_independent=True,
        historical_processor_or_prompt_token_equality_verified=False,
        raw_QA_public=False, TEST_rows_opened=0, images_opened=0, model_requests=0,
        training_updates=0, predictions_read=0, scores_recomputed=0,
        seconds=time.perf_counter()-started)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
