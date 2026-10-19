"""Independent stdlib reconstruction of census aggregates from frozen TRAIN."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/train_view_census/v1'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    v=json.loads((REPORT/'COMPLETE.json').read_text())
    assert v['complete'] and v['TRAIN_QA']==1313 and v['tasks']==19
    assert v['MODEL_FORWARDS']==v['TEST_QA']==0 and not v['CUDA_initialized']
    assert v['source_sha256']==sha(ROOT/'analysis/edge_train_view_census_20261019.py')
    manifest=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest)==v['manifest_sha256']
    data=json.loads(manifest.read_text());images={r['path']:r for r in v['images']}
    assert len(images)==len(v['images'])==v['unique_source_images']
    for r in images.values():
        assert r['source_sha256']==sha(ROOT/r['path'])
        hashes=r['view_sha256'];assert r['views']==len(hashes)==len(r['inverse'])
        assert len(set(hashes))==r['unique_views']==len(r['first_indices'])
        unique=[hashes[i] for i in r['first_indices']]
        assert len(set(unique))==len(unique) and [unique[i] for i in r['inverse']]==hashes
    domains={};tasks={};used=set()
    for task in data['tasks']:
        info=task['files']['train'];p=ROOT/info['path'];assert sha(p)==info['sha256']
        for line in p.read_text().splitlines():
            row=json.loads(line);items=[images[n] for n in row['images']];used.update(row['images'])
            raw=sum(i['views'] for i in items);unique=sum(i['unique_views'] for i in items)
            for table,key in [(domains,task['domain']),(tasks,task['task_id'])]:
                a=table.setdefault(key,dict(QA=0,image_references=0,raw_views=0,exact_unique_within_each_image_views=0,QA_with_duplicates=0,paths=set()))
                a['QA']+=1;a['image_references']+=len(items);a['raw_views']+=raw
                a['exact_unique_within_each_image_views']+=unique;a['QA_with_duplicates']+=int(raw>unique);a['paths'].update(row['images'])
    assert used==set(images)
    for table,reference in [(domains,v['domains']),(tasks,v['native_tasks'])]:
        for k,a in table.items():
            paths=a.pop('paths');source=[images[n] for n in paths]
            a.update(unique_source_images=len(source),unique_source_raw_views=sum(r['views'] for r in source),
                unique_source_reused_views=sum(r['unique_views'] for r in source),
                unique_source_images_with_duplicates=sum(r['views']>r['unique_views'] for r in source))
            assert a==reference[k]
    raw=sum(x['raw_views'] for x in domains.values());unique=sum(x['exact_unique_within_each_image_views'] for x in domains.values())
    total=dict(QA=sum(x['QA'] for x in domains.values()),QA_with_duplicates=sum(x['QA_with_duplicates'] for x in domains.values()),raw_views=raw,exact_unique_within_image_views=unique,view_count_reduction_fraction=(raw-unique)/raw)
    out=dict(complete=True,verified_domains=domains,verified_total=total,
        census_complete_sha256=sha(REPORT/'COMPLETE.json'),verifier_sha256=sha(__file__),
        source_image_SHA_rechecked_on_new_host=len(images),TRAIN_only=True,TEST_QA=0,
        model_forwards=0,no_latency_or_F1_claim=True)
    (REPORT/'VERIFIED_RESULTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    lines=['# 冻结1313TRAIN的精确重复视图：完整统计','',
        '完整19任务、五域，独立读取冻结TRAIN重新汇总，并在新服务器逐一复算原图SHA。原预处理在五条预定TRAIN上与冻结公共函数的像素SHA完全一致。零模型前向、零TEST；没有改动数据。','',
        '| 领域 | TRAIN QA | 含重复视图QA | 原视图数 | 同一图内精确唯一视图数 | 可省视图编码占比 |',
        '|---|---:|---:|---:|---:|---:|']
    for k,a in domains.items():
        reduction=(a['raw_views']-a['exact_unique_within_each_image_views'])/a['raw_views']
        lines.append(f"| {k} | {a['QA']} | {a['QA_with_duplicates']} | {a['raw_views']} | {a['exact_unique_within_each_image_views']} | {100*reduction:.2f}% |")
    lines+=['',f"合计：{total['QA_with_duplicates']}/{total['QA']}条TRAIN包含同图精确重复；{raw}个视图可复用为{unique}次唯一视图编码，视图计数减少{100*total['view_count_reduction_fraction']:.2f}%。按唯一源图像和原生任务的另一套完整计数保留在JSON，不将重复QA当作独立图片。",'',
        '结论：遥感的重复不是五题偶然现象，但其他三域没有精确重复，医学也只有部分。因此单靠精确视图去重无法解释全部部署成本；它是保持原token布局的工程控制，不能作为已获得广泛端侧优势的新算法。', '',
        '视图次数减少不是实测延迟／显存改善：改变批量会影响内核、浮点舍入和利用率，GPU数值与成本仍待。这里只覆盖冻结训练池，不推广测试集、全部数据或其他骨干；也不把预处理冗余说成1%少数据造成的性质。GPU预定四条件与50%预算不因本统计改变。']
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(total))


if __name__=='__main__':main()
