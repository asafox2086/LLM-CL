"""One stdlib consumption of first pre-update NLL; no tensors, QA or scoring."""
from pathlib import Path
import json,hashlib,time,datetime,ast
R=Path('/data1/liyapeng/program/LLMCL')
P=R/'summary/final_weight_reduction_20261023/initial_loss_evidence_20261023/v1'
OLD=R/'summary/edge_deployment_20261019/osft_increment_screen/v1'
NEW=R/'summary/final_weight_reduction_20261023/answer_guided_pilot_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    tick=time.monotonic();assert not (P/'EVIDENCE.json').exists()
    old=read(OLD/'READY.json');new=read(NEW/'READY.json')
    oldproof=read(OLD/'sealed_case_reports_20261022/LoRA_qv_VERIFIED.json')
    assert sha(OLD/'READY.json')==oldproof['ready_sha256']
    assert sha(NEW/'READY.json')=='bb00ee0d5c50b9854f01fdb6e5a6f8efd50f3b547409e04487ae25d4b25bec06'
    assert old['manifest_sha256']==new['manifest_sha256'] and old['split_sha256']==new['split_sha256']
    assert old['train_AST_sha256']==new['training_ast_sha256']
    assert set(old['base_files'])<=set(new['base_files'])
    assert all(new['base_files'][n]==h for n,h in old['base_files'].items())
    extra_files=sorted(set(new['base_files'])-set(old['base_files']))
    assert len(extra_files)==14
    common=set(old['sources'])&set(new['sources'])
    assert common and all(old['sources'][n]==new['sources'][n] for n in common)
    port='exp/two_backbone_native_cl_20261017_v2/'
    required=[port+n for n in ('run.py','runtime_utils.py','onevision_bridge.py','frozen_vision.py')]
    assert all(n in common and sha(R/n)==old['sources'][n] for n in required)
    out=dict(question_sha256=sha(P/'QUESTION.md'),analysis_source_sha256=sha(Path(__file__)),
        observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),records={},
        shared_manifest_sha256=old['manifest_sha256'],shared_split_sha256=old['split_sha256'],
        shared_training_AST_sha256=old['train_AST_sha256'],same_common_base_file_hash_declarations=True,
        historical_preprocessing_hash_match=False,strict_input_and_preprocess_match=False,controlled_causality=False,
        additional_files_only_frozen_in_new_run=extra_files,
        common_source_count=len(common),input_and_training_sources={n:old['sources'][n] for n in required},
        provenance_sha256={str(p.relative_to(R)):sha(p) for p in [OLD/'READY.json',NEW/'READY.json',OLD/'sealed_case_reports_20261022/LoRA_qv_VERIFIED.json']},
        no_model_tensor_QA_generation_optimizer_GPU_score_bootstrap=True,
        batch_input_digest_available=False,no_final_F1_causality_or_unique_SVD_attribution=True)
    for case in ('LoRA_qv','Spectral','Answer'):
        if case=='LoRA_qv':
            root=R/'exp/osft_increment_screen_20261019/v1/private'/case
            complete=OLD/'LoRA_qv_COMPLETE.json';c=read(complete)
            assert sha(complete)==oldproof['complete_marker_sha256']
            expected=oldproof['training_events_sha256']
        else:
            root=R/'exp/answer_guided_rank_renewal_20261023/v1/private'/case
            complete=NEW/'cases'/case/'COMPLETE.json';c=read(complete);expected=c['events_sha256']
            assert sha(complete)==read(NEW/'VERIFIED.json')['cases'][case]['complete_sha256']
        p=root/'logs/events.jsonl';raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
        lines=raw.decode().splitlines();assert len(lines)==87
        e=json.loads(lines[0]);assert e['event']=='train_update' and (e['stage'],e['step'],e['qa_seen'])==(1,1,16)
        selected={k:e[k] for k in ('stage','step','qa_seen','total_qa','loss','prompt_tokens','target_tokens','learning_rate_used')}
        out['records'][case]=dict(first_pre_update_batch=selected,event_log_sha256=expected,complete_sha256=sha(complete),
            complete_path=complete.relative_to(R).as_posix(),event_rows=87)
    ref=out['records']['LoRA_qv']['first_pre_update_batch']
    for case in ('Spectral','Answer'):
        e=out['records'][case]['first_pre_update_batch']
        assert all(e[k]==ref[k] for k in ('stage','step','qa_seen','total_qa','prompt_tokens','target_tokens'))
        out['records'][case]['NLL_difference_from_qv']=e['loss']-ref['loss']
    out['interpretation_scope']='Same declared FIT recipe, ordering source and batch token counts, not input-tensor identity or a new controlled model run. The recorded loss precedes the first optimizer update, so any difference is already present before stage-end consolidation. One batch is not a five-domain or final F1 diagnosis.'
    out['consumer_seconds_before_write']=time.monotonic()-tick
    assert out['consumer_seconds_before_write']<10
    (P/'EVIDENCE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(evidence_sha256=sha(P/'EVIDENCE.json'),records=out['records'],seconds=out['consumer_seconds_before_write'])))
if __name__=='__main__':main()
