"""Derive Qwen pct01 CL-MoE versus Seq acquisition from sealed aggregates.

CLI: --root /data1/liyapeng/program/LLMCL
Writes only a new summary/qwen_cl_completed_20261028/ACQUISITION.json.
No scorer/model imports, QA/prediction/checkpoint reads, or server operations.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import time


DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
FRAME = 'exp/two_backbone_native_cl_20261017_v2'
MANIFEST = 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
COLLECTOR_SHA = 'd7aac920f738c6ee8fde9a636560f2d2d6ee7ea1ba9de9e030e6548b4b4cad0a'
PROOF_DIR = 'summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/completed'
SPECS = {
    'CL_MoE': dict(file='ebf69e0412424b46f33e.json',
        sha256='b4de7085d6c0b0f5cb73a8720928b1ceee63c41eb586a8ae0c40d9f0f3214a72',
        run=FRAME+'/runs/qwen/pct01/CL_MoE/training_20261022_162837'),
    'SeqFT_LoRA': dict(file='2a0b6d43f89c3cec6cb9.json',
        sha256='ff979a21b171372834940e3b68bdcaa84c7db4cdf9eaccb8939fa4b3c84bd59a',
        run=FRAME+'/runs/qwen/pct01/SeqFT_LoRA/training_20261019_verified_resume_branch_v2'),
}
SCORE_KEYS = {'rougeL', 'token_f1', 'exact_match', 'count', 'exact_match_correct'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def scalar(value):
    check(isinstance(value, (int, float)) and not isinstance(value, bool)
          and math.isfinite(value) and 0 <= value <= 100, 'invalid original F1 scalar')
    return value


def count(value):
    check(isinstance(value, int) and not isinstance(value, bool) and value > 0,
          'invalid original QA count')
    return value


def close(left, right, label):
    check(math.isclose(left, right, rel_tol=0, abs_tol=1e-10), label)


class Metadata:
    def __init__(self, root):
        self.root, self.ledger, self.stamps = root, {}, {}

    def read(self, relative, expected):
        rel = PurePosixPath(relative)
        check(not rel.is_absolute() and '..' not in rel.parts and '\\' not in relative,
              'invalid metadata relative path')
        path = self.root/relative
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        stamp = (before.st_size, before.st_mtime_ns)
        check(stamp == (after.st_size, after.st_mtime_ns), 'metadata changed during read')
        check(digest(raw) == expected, 'sealed metadata SHA mismatch: '+relative)
        check(relative not in self.stamps or self.stamps[relative] == stamp,
              'metadata changed between repeated references')
        self.stamps[relative] = stamp
        self.ledger[relative] = dict(sha256=expected, bytes=len(raw), mtime_ns=before.st_mtime_ns)
        return json.loads(raw)

    def unchanged(self):
        for relative, stamp in self.stamps.items():
            current = (self.root/relative).stat()
            check((current.st_size, current.st_mtime_ns) == stamp,
                  'metadata changed before output: '+relative)


def signs(values):
    return dict(higher=sum(x > 0 for x in values), lower=sum(x < 0 for x in values),
                tie=sum(x == 0 for x in values), denominator=len(values))


def read_method(io, method, manifest):
    spec = SPECS[method]
    proof_path = PROOF_DIR+'/'+spec['file']
    proof = io.read(proof_path, spec['sha256'])
    c = proof['candidate']
    check(c['host'] == 'new3090' and c['backbone'] == 'qwen' and c['method'] == method
          and c['run'] == spec['run'] and c['collector_sha256'] == COLLECTOR_SHA,
          'fixed formal candidate identity')
    check(proof['CPU_only'] and proof['primary_metric'] == 'existing raw Answer F1'
          and proof['rescored_cells'] == 74 and proof['final_QA'] == 21454
          and proof['positive_finite_training_stages'] == 19,
          'original formal completeness')
    sources = proof['source_sha256']
    check(sources[MANIFEST] == MANIFEST_SHA, 'same sealed manifest')
    config_path = spec['run']+'/config.json'
    check(sources[config_path] == proof['config_sha256'], 'config proof linkage')
    config = io.read(config_path, proof['config_sha256'])
    check(config['experiment_id'] == FRAME.split('/')[-1]
          and config['backbone'] == 'qwen' and config['method'] == method
          and config['condition'] == 'pct01' and config['manifest_sha256'] == MANIFEST_SHA
          and config['seed'] == 42 and config['epochs'] == 1 and config['batch_size'] == 16
          and config['micro_batch_size'] == 1 and config['train_fraction'] == .01
          and config['test_fraction'] == 1 and not config['preflight_only'],
          'shared formal data/training configuration')
    check(len(proof['final_native19']) == 19 and set(proof['domains']) == set(DOMAINS),
          'full final scalar arrays')
    rows = []
    for task in manifest['tasks']:
        tid = task['task']; test = task['files']['test']; train = task['files']['train']
        check(sources[test['path']] == test['sha256'] and sources[train['path']] == train['sha256'],
              'original paired TRAIN/TEST file identities')
        check(config['task_counts'][task['task_id']]['test'] == test['count']
              and config['task_counts'][task['task_id']]['train'] == train['count'], 'config task counts')
        prediction_rel = f'evaluations/stage_{tid:02d}/T{tid:02d}_test_predictions.jsonl'
        prediction_path = spec['run']+'/'+prediction_rel
        check(sources[prediction_path] == proof['prediction_sha256'][prediction_rel],
              'paired original prediction SHA declarations')
        score_path = prediction_path.replace('.jsonl', '.scores.json')
        saved = io.read(score_path, sources[score_path])
        check(set(saved) == SCORE_KEYS and count(saved['count']) == test['count'],
              'actual sealed aggregate score schema/count')
        check(isinstance(saved['exact_match_correct'], int)
              and 0 <= saved['exact_match_correct'] <= saved['count'], 'existing exact count')
        for field in ('token_f1', 'rougeL', 'exact_match'):
            scalar(saved[field])
        acquired = saved['token_f1']
        final = scalar(proof['final_native19'][tid-1])
        if tid == 19:
            close(acquired, final, 'T19 acquired is original final-stage scalar')
        rows.append(dict(task=tid, task_id=task['task_id'], domain=task['domain'],
            native_label_field=task['native_label_field'], native_label=task['native_label'],
            TEST_QA=test['count'], TRAIN_QA=train['count'], acquired_stage=tid,
            acquired_F1=acquired, final_F1=final, aggregate_score_path=score_path,
            aggregate_score_sha256=sources[score_path], paired_prediction_sha256=sources[prediction_path],
            TEST_file_sha256=test['sha256'], original_ID_reference_pairing_inherited_from_formal_proof=True))
    domains = {}
    for domain in DOMAINS:
        selected = [r for r in rows if r['domain'] == domain]
        n = sum(r['TEST_QA'] for r in selected)
        acquired = sum(r['acquired_F1']*r['TEST_QA'] for r in selected)/n
        final = scalar(proof['domains'][domain]['F1'])
        check(proof['domains'][domain]['QA'] == n, 'original final domain QA')
        close(sum(r['final_F1']*r['TEST_QA'] for r in selected)/n, final,
              'copied final native19/domain consistency')
        domains[domain] = dict(TEST_QA=n, tasks=[r['task'] for r in selected],
                               acquired_F1=acquired, final_F1=final)
    acquired_macro = sum(domains[d]['acquired_F1'] for d in DOMAINS)/5
    final_macro = scalar(proof['domain_macro_final'])
    close(sum(domains[d]['final_F1'] for d in DOMAINS)/5, final_macro, 'final five-domain macro')
    final_ap = scalar(proof['aggregate']['AP'])
    close(sum(r['final_F1'] for r in rows)/19, final_ap, 'original final AP19')
    return dict(proof_file=proof_path, proof_sha256=spec['sha256'], candidate=c,
        checkpoint_sha256_from_formal_proof=proof['checkpoint_sha256'],
        config_sha256=proof['config_sha256'], tasks=rows, domains=domains,
        acquired_domain_macro_F1=acquired_macro, acquired_AP19=sum(r['acquired_F1'] for r in rows)/19,
        final_domain_macro_F1=final_macro, final_AP19=final_ap,
        final_scalars_copied_from_cached_formal_proof=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    check(args.root.is_absolute() and args.root.is_dir(), 'absolute existing NEW root required')
    root = args.root.resolve()
    output = root/'summary/qwen_cl_completed_20261028/ACQUISITION.json'
    check(not output.exists(), 'exclusive derivation output already exists')
    tick, unix = time.monotonic(), time.time()
    io = Metadata(root)
    manifest = io.read(MANIFEST, MANIFEST_SHA)
    check(manifest['condition'] == 'pct01' and manifest['seed'] == 42
          and manifest['formal_ready'] and not manifest['assets_pending']
          and [t['task'] for t in manifest['tasks']] == list(range(1, 20)), 'original manifest')
    check(sum(count(t['files']['train']['count']) for t in manifest['tasks']) == 1313
          and sum(count(t['files']['test']['count']) for t in manifest['tasks']) == 21454
          and set(t['domain'] for t in manifest['tasks']) == set(DOMAINS), 'complete original data counts')
    methods = {m: read_method(io, m, manifest) for m in SPECS}
    cl, seq = methods['CL_MoE'], methods['SeqFT_LoRA']
    task_differences = []
    for a, b in zip(cl['tasks'], seq['tasks']):
        for field in ('task', 'task_id', 'domain', 'TEST_QA', 'TEST_file_sha256'):
            check(a[field] == b[field], 'paired task/TEST identity')
        task_differences.append(dict(task=a['task'], domain=a['domain'], TEST_QA=a['TEST_QA'],
            native_label=a['native_label'], CL_acquired_F1=a['acquired_F1'], Seq_acquired_F1=b['acquired_F1'],
            CL_minus_Seq_acquired_pp=a['acquired_F1']-b['acquired_F1'],
            CL_final_F1=a['final_F1'], Seq_final_F1=b['final_F1'],
            CL_minus_Seq_final_pp=a['final_F1']-b['final_F1']))
    domain_differences = {d:dict(TEST_QA=cl['domains'][d]['TEST_QA'],
        CL_minus_Seq_acquired_pp=cl['domains'][d]['acquired_F1']-seq['domains'][d]['acquired_F1'],
        CL_minus_Seq_final_pp=cl['domains'][d]['final_F1']-seq['domains'][d]['final_F1']) for d in DOMAINS}
    result = dict(schema='qwen-cl-versus-seq-acquisition-v1',
        status='COMPLETE_SEALED_AGGREGATE_DERIVATION', backbone='qwen', condition='pct01',
        methods=methods, CL_minus_Seq=dict(tasks=task_differences, domains=domain_differences,
            acquired_domain_macro_pp=cl['acquired_domain_macro_F1']-seq['acquired_domain_macro_F1'],
            acquired_AP19_pp=cl['acquired_AP19']-seq['acquired_AP19'],
            final_domain_macro_pp=cl['final_domain_macro_F1']-seq['final_domain_macro_F1'],
            final_AP19_pp=cl['final_AP19']-seq['final_AP19'],
            task_acquired_counts=signs([r['CL_minus_Seq_acquired_pp'] for r in task_differences]),
            domain_acquired_counts=signs([r['CL_minus_Seq_acquired_pp'] for r in domain_differences.values()]),
            task_final_counts=signs([r['CL_minus_Seq_final_pp'] for r in task_differences]),
            domain_final_counts=signs([r['CL_minus_Seq_final_pp'] for r in domain_differences.values()])),
        source_metadata=io.ledger, source_script_sha256=digest(Path(__file__).read_bytes()),
        weights=dict(within_domain='Original complete TEST QA counts', domain_macro='Five domains equally weighted',
                     acquired_AP19='19 task-specific acquired states equally weighted'),
        limits=['Descriptive comparison of two completed formal methods, not a causal intervention.',
                'Acquired is R[t,t]; domain summaries combine task-specific times, not one model at one time.',
                'Acquired differences include earlier-task transfer and current learning; no pure current-stage attribution.',
                'No retention/forgetting decomposition or significance/equivalence test.',
                'No claim that all tasks/domains must win, and no new method/parameter or GPU measurement.'],
        activity=dict(cached_proofs=2, existing_aggregate_score_files=38,
            new_scores=0, new_models=0, new_generations=0, QA_rows_read=0, prediction_rows_read=0,
            checkpoint_payload_reads=0, paired_prediction_files_hashed_or_opened=0,
            TEST_files_hashed_or_opened=0, pairing_source='Original formal proof SHA declarations'),
        started_unix=unix, seconds_before_output=time.monotonic()-tick)
    io.unchanged()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output), SHA256=digest(output.read_bytes()),
                         new_scores=0, new_models=0)))


if __name__ == '__main__':
    main()
