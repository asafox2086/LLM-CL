"""Publish independently verified FOREVER alongside the immutable seven scope.

CPU documentation only. Neither active collector source nor its 14-pair index
is changed. No training, generation, checkpoint loading or result selection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import v2_readme_edge_table_20261019 as edge

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/two_backbone_pct01_20261017/forever_verified_20261020'
PROOF = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v3/formal_verification/dependency_restore_v1/onevision_COMPLETE.json'
VERIFIER = ROOT/'analysis/forever_multimodal_formal_verify_v3_20261019.py'
VERIFIER_SHA = '5f51ab1cbfc871ed1c9492a042235c71d4830517c98f3b34da6e4180bd4bc55a'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def forever():
    proof = json.loads(PROOF.read_text())
    transfer = json.loads((REPORT/'SOURCE_TRANSFER.json').read_text())
    assert sha(PROOF)==transfer['files'][str(PROOF.relative_to(ROOT))]['sha256']
    assert sha(VERIFIER)==proof['wrapper_sha256']==VERIFIER_SHA
    assert proof['status']=='FOREVER_FORMAL_COMPLETE_VERIFIED'
    assert proof['GPU_or_training_launched'] is False and proof['formal_publisher_not_modified']
    base = proof['shared_V2_proof']; c=base['candidate']
    assert c['method']=='FOREVER' and c['backbone']=='onevision'
    assert c['collector_sha256']==edge.formal.COLLECTOR_SHA
    assert base['CPU_only'] and base['rescored_cells']==74 and base['final_QA']==21454
    assert base['positive_finite_training_stages']==19 and base['primary_metric']=='existing raw Answer F1'
    assert len(base['checkpoint_sha256'])==64 and len(base['prediction_sha256'])==74
    assert len(proof['replay_stages'])==18
    assert sum(x['actual_QA'] for x in proof['replay_stages'])==proof['extra_replay_QA_exposures']
    assert sum(x['actual_updates'] for x in proof['replay_stages'])==proof['extra_replay_optimizer_updates']
    assert sum(x['positive_LR'] for x in proof['replay_stages'])==proof['extra_replay_positive_LR_updates']
    assert sum(x['author_zero_warmup'] for x in proof['replay_stages'])==proof['author_replay_zero_warmup_updates']
    assert proof['extra_replay_optimizer_updates']==proof['extra_replay_positive_LR_updates']+proof['author_replay_zero_warmup_updates']
    values = [base['domains'][d]['F1'] for d in edge.formal.DOMAINS]
    values += [base['aggregate'][m] for m in ['AP','FWT','BWT']]
    assert all(math.isfinite(x) for x in values)
    return proof,values


def internal(text,values):
    lines=[];backbone=None;found=0
    for line in text.splitlines():
        if line.startswith('| **'):
            backbone=next(k for k,v in edge.formal.BACKBONES if v in line)
        if line.startswith('| FOREVER'):
            cells=[s.strip() for s in line.split('|')[1:-1]];assert len(cells)==9
            if backbone=='onevision':
                expected=[f'{x:.3f}' for x in values]
                assert cells[1:]==['']*8 or cells[1:]==expected
                cells=['FOREVER']+expected
            else:
                assert cells[1:]==['']*8
                cells[0]='FOREVER'
            line='| '+' | '.join(cells)+' |';found+=1
        lines.append(line)
    assert found==2
    out='\n'.join(lines)+'\n'
    out=out.replace('FOREVER为第8项计划比较，未准入的占位不计完成。',
        'FOREVER已作为第8个方法独立准入；OneVision完整结果由v3专用验证器核验，Qwen尚待运行。原七方法发布器范围14组不变。')
    note='[FOREVER独立完整核验与额外回放成本](forever_verified_20261020/README.md)。'
    if note not in out:out+='\n'+note+'\n'
    return out


def main(write_files):
    index_path=edge.formal.INDEX;index_bytes=index_path.read_bytes();index=json.loads(index_bytes)
    base_rows=edge.formal.verified_rows(index);proof,values=forever();key=('onevision','FOREVER')
    assert key not in base_rows
    original_verify=edge.formal.verified_rows
    def combined(idx):
        rows=original_verify(idx);assert key not in rows;rows[key]=values;return rows
    edge.formal.verified_rows=combined
    try:
        root=ROOT/'README.md';before=root.read_text();after=edge.render(before)
        assert edge.render(after)==after
    finally:edge.formal.verified_rows=original_verify
    # Scientific cells of the seven-method scope stay exactly the sealed values.
    backbone=None;got={}
    for line in after.splitlines():
        if not line.startswith('|'):continue
        cells=[s.strip() for s in line.split('|')[1:-1]]
        if cells[0].startswith('**') and len(cells)==17:
            backbone=next(k for k,v in edge.formal.BACKBONES if v in cells[0])
        elif len(cells)==17 and cells[0] in dict((label,m) for m,label in edge.formal.METHODS):
            method=dict((label,m) for m,label in edge.formal.METHODS)[cells[0]]
            got[backbone,method]=cells[3:11]
    for k,v in base_rows.items():assert got[k]==[f'{x:.3f}' for x in v]
    assert got[key]==[f'{x:.3f}' for x in values]
    assert got['qwen','FOREVER']==['']*8
    count=len(base_rows)+1
    after=after.replace('OneVision在旧GPU0进行正式S0评价，Qwen等待原基线后的空闲GPU2',
        'OneVision已通过完整19阶段／74格／21454题及回放成本核验，Qwen仍等待原基线后的空闲GPU2')
    after=after.replace('共16个目标中4组完整核验',f'共16个目标中{count}组完整核验')
    marker='<!-- FOREVER_V3_PUBLICATION_20261020 -->'
    note=marker+'\n**正式基线新结果：** FOREVER／OneVision完整核验AP45.245、FWT17.748、BWT5.642；正式完成'+str(count)+'/16组。额外13,952次历史QA曝光／872次回放更新另计，不当作相同训练计算量。[完整证明与回放成本](summary/two_backbone_pct01_20261017/forever_verified_20261020/README.md)。\n\n'
    if marker in after:
        left,tail=after.split(marker,1);_,right=tail.split('\n\n',1);after=left+note+right
    else:after=note+after
    source_link='[FOREVER额外完整核验](summary/two_backbone_pct01_20261017/forever_verified_20261020/README.md)'
    if source_link not in after:
        anchor='[逐题核验索引](summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/VERIFIED_INDEX.json)'
        assert anchor in after;after=after.replace(anchor,anchor+' · '+source_link,1)
    table=ROOT/'summary/two_backbone_pct01_20261017/MAIN_TABLE.md';table_before=table.read_text();table_after=internal(table_before,values)
    assert internal(table_after,values)==table_after
    assert index_path.read_bytes()==index_bytes
    merged=dict(base_rows);merged[key]=values
    publication=dict(CPU_only=True,original_seven_scope_index_unchanged=True,
        original_seven_scope_pairs=index['required_pairs'],total_target_pairs=16,
        complete_pairs=count,original_index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        FOREVER_proof=str(PROOF.relative_to(ROOT)),FOREVER_proof_sha256=sha(PROOF),
        FOREVER_verifier_sha256=VERIFIER_SHA,publication_source_sha256=sha(__file__),
        raw_AnswerF1_only=True,main_table_rows=[dict(backbone=b,method=m,
            domains=dict(zip(edge.formal.DOMAINS,v[:5])),domain_macro_F1=sum(v[:5])/5,
            AP=v[5],FWT=v[6],BWT=v[7]) for (b,m),v in merged.items()],
        replay_QA_exposures=proof['extra_replay_QA_exposures'],replay_updates=proof['extra_replay_optimizer_updates'],
        replay_positive_LR=proof['extra_replay_positive_LR_updates'],author_replay_warmup_zero=proof['author_replay_zero_warmup_updates'],
        deployment_columns_NOT_measured=True,not_a_research_method_gain=True)
    if write_files:
        assert root.read_text()==before and table.read_text()==table_before
        root.write_text(after);table.write_text(table_after)
        assert index_path.read_bytes()==index_bytes
        write(REPORT/'PUBLICATION.json',publication)
    print(json.dumps(dict(published=write_files,complete_pairs=count,FOREVER=proof['shared_V2_proof']['aggregate'],
                          original_seven_scope_unchanged=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args();main(a.write)
