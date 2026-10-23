"""Only the new tiny visual-graph/gradient/serialization admission; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy, importlib, json, pickle, sys, time, traceback
from pathlib import Path
from core import *

def cpu_gate():
    import torch,cloudpickle
    from transformers import LlavaOnevisionConfig,LlavaOnevisionForConditionalGeneration,SiglipVisionConfig,Qwen2Config
    sys.path.insert(0,str(PORT));import onevision_bridge
    from host_adam import HostAdamW
    torch.set_num_threads(2);torch.manual_seed(42)
    vc=SiglipVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=26,num_attention_heads=2,
        image_size=28,patch_size=14,vision_use_head=False)
    tc=Qwen2Config(hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,
        num_key_value_heads=1,vocab_size=64,max_position_embeddings=128)
    cfg=LlavaOnevisionConfig(vision_config=vc,text_config=tc,image_token_index=63,
        image_grid_pinpoints=[[28,28]],vision_feature_layer=-1,vision_feature_select_strategy='full')
    model=onevision_bridge.OneVisionBridge(LlavaOnevisionForConditionalGeneration(cfg)).eval()
    student=make_student(model);model._llmcl_visual_original=model.visual.forward
    original=teacher_layers(model);assert len(original)==26
    old=digest(model.parameters());new0=digest(student.parameters())
    pixels=torch.randn(1,2,3,28,28);sizes=torch.tensor([[28,28]])
    with torch.no_grad():target=model.visual(pixels,sizes)
    assert target.ndim==2 and target.shape[1]==16 and torch.isfinite(target).all()
    before=extract_student(model,student,pixels,sizes)
    assert teacher_layers(model) is original and len(original)==26 and before.shape==target.shape
    for p in student.parameters():p.requires_grad_(True)
    optimizer=HostAdamW(student.parameters(),lr=1e-4,weight_decay=0.,eps=1e-8,betas=(.9,.999))
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        predicted=extract_student(model,student,pixels,sizes)
        loss=normalized_error(predicted,target)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
        assert all(p.grad is None for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(student.parameters(),1.);optimizer.step()
        value=dict(step=step+1,loss=float(loss),keys=[n for n,p in student.named_parameters() if p.grad is not None])
        path=Path(__file__).parent/'.cpu_admission'/'missing_parent'/f'{step+1}.json';write(path,value)
        assert json.loads(path.read_text())==value
    assert digest(model.parameters())==old and digest(student.parameters())!=new0
    assert teacher_layers(model) is original
    model._llmcl_visual_cache={};model._llmcl_visual_cache_bytes=0
    cleanup_compact(model,student)
    final=digest(model.parameters());path=Path(__file__).parent/'.cpu_admission'/'compact.pt'
    torch.save(model,path,pickle_module=cloudpickle)
    loaded=torch.load(path,pickle_module=pickle,weights_only=False)
    assert len(teacher_layers(loaded))==14 and digest(loaded.parameters())==final
    assert count(loaded)==count(model)
    return dict(passed=True,CPU_FP32_only_not_actual_CUDA_FP16_quality=True,
        actual_tiny_Siglip_26_to_14=True,AnyRes_packed_projection_gradient=True,
        student_not_teacher_parameter_alias=True,parent_unchanged_after_two_updates=True,
        short_tower_restore=True,missing_directory_atomic_JSON_reload=True,
        compact_full_graph_reload=True,teacher_tail_not_registered_in_compact_graph=True,
        before_feature_shape=list(target.shape),CUDA_initialized=torch.cuda.is_initialized())

def main():
    assert not any((REPORT/n).exists() for n in ('READY.json','CPU_GATE.json','CPU_GATE_FAILURE.json'))
    review=json.loads((REPORT/'SOURCE_REVIEW.json').read_text());assert review['decision']=='accept'
    for n,h in review['files_sha256'].items():assert sha(ROOT/n)==h,n
    _,_,_,ledger=data()
    assert len(ledger['calibration'])==260 and len(ledger['diagnostic'])==65
    oldpath=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/READY.json'
    old=json.loads(oldpath.read_text())
    assert ledger['old_SPLIT_sha256']==old['split_sha256']
    assert ledger['manifest_sha256']==old['manifest_sha256']
    for n,h in ledger['selected_image_sha256'].items():assert sha(ROOT/n)==h,n
    vc=json.loads((ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/config.json').read_text())['vision_config']
    assert (vc['hidden_size'],vc['intermediate_size'],vc['num_hidden_layers'],vc['num_attention_heads'])==(1152,4304,26,16)
    for n,h in old['sources'].items():assert sha(ROOT/n)==h,n
    import torch
    assert not torch.cuda.is_initialized();start=time.monotonic();gate=cpu_gate();gate['seconds']=time.monotonic()-start
    assert gate['passed'] and not torch.cuda.is_initialized()
    write(REPORT/'CPU_GATE.json',gate)
    dependencies=dict(old['sources']);dependencies[oldpath.relative_to(ROOT).as_posix()]=sha(oldpath)
    dependencies.update(ledger['selected_image_sha256'])
    for n,h in dependencies.items():assert sha(ROOT/n)==h,n
    for name in ('run.py','runtime_utils.py','onevision_bridge.py','frozen_vision.py','checkpoints.py','host_adam.py','resource_monitor.py','training_validity.py'):
        p=PORT/name;dependencies[p.relative_to(ROOT).as_posix()]=sha(p)
    runtime_sources={}
    for name in ('siglip.modeling_siglip','llava_onevision.modeling_llava_onevision','qwen2.modeling_qwen2',
                 'llava_onevision.processing_llava_onevision','llava_onevision.image_processing_llava_onevision'):
        module='transformers.models.'+name;p=Path(importlib.import_module(module).__file__).resolve()
        runtime_sources[module]=str(p)
        dependencies[p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)]=sha(p)
    for name in ('analysis/onevision_research_device_entry.py','exp/osft_increment_screen_20261019/v1/study.py'):
        dependencies[name]=sha(ROOT/name)
    base_files=dict(old['base_files'])
    for p in (ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV').iterdir():
        if p.is_file() and p.suffix in ('.json','.txt','.model'):base_files[p.relative_to(ROOT).as_posix()]=sha(p)
    evidence={p.relative_to(ROOT).as_posix():sha(p) for p in [REPORT/'PROTOCOL.md',REPORT/'NECESSITY_REVIEW.json',REPORT/'SOURCE_REVIEW.json',REPORT/'CPU_GATE.json']}
    write(REPORT/'READY.json',dict(sources={p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')},
        dependencies=dependencies,base_files=base_files,evidence=evidence,data=ledger,
        runtime_source_paths=runtime_sources,runtime_interpreter=str(Path(sys.executable).resolve()),
        prefix_blocks=13,student_tail_blocks=1,teacher_tail_blocks=13,student_training_coordinates=15239504,
        model_budget_seconds=BUDGET,train_only_S1=True,not_Answer_F1_or_new_framework=True,
        automatic_continual_training=False,scientific_sources_FROZEN=True))
    print(json.dumps(dict(ready_sha256=sha(REPORT/'READY.json'),gate_seconds=gate['seconds'],no_actual_model_started=True)))

if __name__=='__main__':
    try:main()
    except Exception:
        write(REPORT/'CPU_GATE_FAILURE.json',dict(traceback=traceback.format_exc(),no_blind_repeated_gate=True));raise
