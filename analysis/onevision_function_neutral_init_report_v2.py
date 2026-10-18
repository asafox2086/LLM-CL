"""Read-only sealed diagnostics; all three conditions, no TEST-based selection."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import onevision_initialization_source_report as original
from answer_f1 import answer_f1

ROOT = original.ROOT
REPORT = ROOT/'summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v2'
FAMILY = ROOT/'exp/onevision_function_neutral_init_20261018/v2'
CASES = ['cold_compensated','within_compensated','cross_compensated']


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def complete(index,spec):
    p=REPORT/(CASES[index]+'_complete.json')
    if not p.exists():return False
    proof=read(p)
    assert proof['complete'] and proof['spec']==spec and proof['target_full_TEST_QA']==507
    run=ROOT/proof['run'];assert run.resolve().is_relative_to(FAMILY.resolve())
    for rel,s in proof['artifact_sha256'].items():
        p=run/rel;assert p.resolve().is_relative_to(run.resolve()) and sha(p)==s
    assert read(run/'actual_function_and_update_gate.json')['passed']
    return True


def main():
    spec=read(REPORT/'CPU_PLAN_CHECK.json')['spec']
    for n,s in spec['source_sha256'].items():assert sha(ROOT/n)==s
    proofs={p.name:sha(p) for p in REPORT.glob('*_complete.json')}
    dest=REPORT/'VERIFIED_EVIDENCE.json'
    if dest.exists() and read(dest)['proof_sha256']==proofs:return
    manifest=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest)==spec['parent_spec']['manifest_sha256']
    f=read(manifest)['tasks'][3]['files']['test'];p=ROOT/f['path'];assert sha(p)==f['sha256']
    gold=[json.loads(s) for s in p.read_text().splitlines()];assert len(gold)==507
    rows={};scores={}
    for name in ['cold','within_full','cross_full']+CASES:
        folder=original.REPORT if name not in CASES else REPORT
        pointer=folder/(name+'_complete.json')
        if not pointer.exists():continue
        if name in CASES:
            assert complete(CASES.index(name),spec)
        proof=read(pointer);run=ROOT/proof['run']
        assert run.resolve().is_relative_to((ROOT/'exp').resolve())
        for rel,s in proof['artifact_sha256'].items():
            p=run/rel;assert p.resolve().is_relative_to(run.resolve()) and sha(p)==s
        actual=[json.loads(s) for s in (run/'evaluations/target_full507/medical_image.jsonl').read_text().splitlines()]
        assert [r['id'] for r in actual]==[r['id'] for r in gold]
        for r,g in zip(actual,gold):
            assert r['references']==g['answers'] and math.isfinite(r['nll'])
            assert r['scores']['token_f1']==answer_f1(r['prediction'],r['references'])
        scores[name]=[r['scores']['token_f1'] for r in actual]
        rows[name]=dict(F1=sum(scores[name])/507,NLL=sum(r['nll'] for r in actual)/507,
            proof_sha256=sha(pointer),run=proof['run'])
    contrasts=[]
    names=list(rows)
    for i,target in enumerate(names):
        for reference in names[:i]:
            blocks=defaultdict(list)
            for j,g in enumerate(gold):blocks[tuple(g['images'])].append(scores[target][j]-scores[reference][j])
            groups=list(blocks.values());rng=random.Random(42);draws=[]
            for _ in range(10000):
                sampled=[groups[rng.randrange(len(groups))] for _ in groups]
                draws.append(sum(map(sum,sampled))/sum(map(len,sampled)))
            draws.sort();contrasts.append(dict(target=target,reference=reference,
                delta_F1=sum(map(sum,groups))/507,image_95CI=[draws[249],draws[9749]],uncorrected=True))
    out=dict(CPU_only=True,proof_sha256=proofs,source_sha256=sha(Path(__file__)),rows=rows,
        all_available_pairwise_contrasts=contrasts,pending=[n for n in CASES if n not in rows],
        TEST_selection=False, limits=['One target/two fixed sources/one seed, extra source12exposures/2updates charged.',
        'Nonzero compensation removes initial output, not prior parameter information; effective rank may be16.',
        'Paired image CIs are uncorrected fixed-model estimates, not multiseed or universal domain effects.',
        'Not a full five-domain CL method, no full AP/FWT/BWT or memory superiority inferred.'])
    dest.write_text(json.dumps(out,indent=2)+'\n')
    lines=['# 保留源参数、抵消初始旧输出：完整507题控制','',
        '| 条件 | 最终507 Answer F1 | 最终507 NLL |','| --- | ---: | ---: |']
    for n in ['cold','within_full','cross_full']+CASES:
        r=rows.get(n);lines.append(f"| {n} | {r['F1']:.3f} | {r['NLL']:.4f} |" if r else f'| {n} | — | — |')
    lines += ['','三补偿条件必须全部测，不用测试挑选赢家。初始32精确函数门槛、真实梯度／更新门槛、冷组两步和507逐项等价均绑定于各完成证明。尚待项不是零收益。',
        '','[预定协议](PROTOCOL.md) · [所有配对／SHA／边界](VERIFIED_EVIDENCE.json)']
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(accepted=[n for n in CASES if n in rows],pending=out['pending'])))


if __name__=='__main__':main()
