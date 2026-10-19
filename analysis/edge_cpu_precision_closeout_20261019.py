"""Close the already-finished CPU engineering controls, without new inference.

These five TRAIN requests are not a CL experiment or evidence of a useful
compression method. Do not use this closure to choose the architecture trial.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'summary/edge_deployment_20261019/cpu_precision/v2'
REPORT=STUDY/'closeout'
VIEWS=['fp32_dense','decoder_int8','all_linear_int8']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def main():
    assert read(STUDY/'ALL_COMPLETE.json')['complete']
    assert read(STUDY/'fresh_reload/ALL_COMPLETE.json')['all_fresh_reload_gates_passed']
    port=ROOT/'exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01/OSFT/training_20261018_044103'
    source=port/'runner_source/runtime_utils.py'
    assert sha(source)==read(port/'config.json')['execution_source_sha256']['runtime_utils.py']
    defs=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['tokens','score']]
    ns=dict(re=re,Counter=Counter);assert len(defs)==2
    exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),str(source),'exec'),ns)
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest=read(manifest_path);selected=[]
    for tid in [1,2,12,16,17]:
        task=manifest['tasks'][tid-1];meta=task['files']['train'];p=ROOT/meta['path'];assert sha(p)==meta['sha256']
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        row=min(rows,key=lambda r:hashlib.sha256(('edge-profile42:'+r['id']).encode()).digest())
        selected.append(row)
    results={}
    for view in VIEWS:
        p=STUDY/(view+'_COMPLETE.json');v=read(p)
        assert v['complete'] and v['TEST_QA']==0 and v['TRAIN_unique_requests']==5 and not v['cuda_initialized']
        assert v['source_sha256']==sha(ROOT/'analysis/edge_cpu_precision_v2_20261019.py')
        assert v['ready_sha256']==sha(STUDY/'READY.json') and v['manifest_sha256']==sha(manifest_path)
        rp=STUDY/'fresh_reload'/(view+'_COMPLETE.json');fresh=read(rp)
        assert fresh['complete'] and fresh['fresh_process_state_and_tokens_exact'] and fresh['strict_all_state_entries_exact']
        assert fresh['source_sha256']==sha(ROOT/'analysis/edge_cpu_precision_reload_20261019.py')
        assert fresh['original_complete_sha256']==sha(p) and fresh['package_file_hashes_rechecked']
        assert len(fresh['gates'])==5 and all(g['generation_tokens_exact'] for g in fresh['gates'])
        assert fresh['reload_process_pid']!=fresh['original_model_process_pid']
        assert len(v['requests'])==20
        scores=[]
        for row in selected:
            quality=[r for r in v['requests'] if r['id']==row['id'] and not r['speed_only']]
            assert len(quality)==1
            q=quality[0];value=ns['score'](q['prediction'],row['answers'])['token_f1']
            assert value==q['TRAIN_AnswerF1']
            ref=next(r for r in v['requests'] if r['id']==row['id'] and r['speed_only'] and r['repeat']==0)
            gate=next(g for g in fresh['gates'] if g['id']==row['id'])
            assert gate['new_request']['output_tokens']==ref['output_tokens']
            scores.append(dict(id=row['id'],AnswerF1=value))
        sizes={name:(ROOT/v['package']/name).stat().st_size for name in v['package_files']}
        assert all(sizes[name]==meta['bytes'] for name,meta in v['package_files'].items())
        assert sum(sizes.values())==v['actual_package_bytes']
        results[view]=dict(scores=scores,package_bytes=v['actual_package_bytes'],quantized_count=v['quantized_count'],
            original_process=fresh['original_model_process_pid'],reload_process=fresh['reload_process_pid'],
            original_complete_sha256=sha(p),reload_complete_sha256=sha(rp),
            complete_fresh_reload_proved=True,not_full_TEST_quality=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    out=dict(complete=True,engineering_only=True,new_model_requests=0,TEST_QA=0,
        scorer_source_sha256=sha(source),collector_source_sha256=sha(__file__),
        views=results,no_main_table_publication=True,no_architecture_trial_selection=True)
    (REPORT/'VERIFIED_RESULTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    lines=['# 既有CPU三精度控制：工程封存','',
        '三组均已完成，并在不同新进程严格恢复全部状态；五题短生成逐值相同。已按全部原参考独立复算自然停Answer F1。本次只收集既有结果，0新增模型请求。',
        '', '| 固定TRAIN请求 | fresh FP32 | decoder INT8 | all-linear INT8 |','|---|---:|---:|---:|']
    for i,row in enumerate(selected):
        lines.append('| '+row['id']+' | '+' | '.join(f"{results[x]['scores'][i]['AnswerF1']:.3f}" for x in VIEWS)+' |')
    lines += ['', '这是五个训练样例，不是五域测试均值／AP，也不是学术方法效果。逐通道量化的已有独立负例另存原目录，不据此推出全部量化无效或某一模块有错。',
        '', '保存的FP32运行镜像不是最小公平存储基线：原FP16文件可以加载为FP32运行。共享旧Xeon上的时间也不作为正式CPU／GPU／手机优势。这里不扩展卡常路线，不用结果选择架构容量配置；最终目标尚未达到。']
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({view:[s['AnswerF1'] for s in v['scores']] for view,v in results.items()}))


if __name__=='__main__':main()
