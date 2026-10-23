"""One S1-only function-substitution diagnostic, not a claimed new framework."""
import contextlib, copy, hashlib, json, math, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/final_weight_reduction_20261023/visual_tail_function_probe_v1'
WORK=Path(__file__).parent/'private'
PREFIX=13
BUDGET=1800
BASE=893675552
sys.path.insert(0,str(ROOT/'exp/osft_increment_screen_20261019/v1'))
from study import sha

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def htext(s):return hashlib.sha256(s.encode()).hexdigest()

def data():
    policy=json.loads((PORT/'execution_policy.json').read_text())['conditions']['pct01']
    mp=ROOT/policy['manifest'];assert sha(mp)==policy['manifest_sha256']
    manifest=json.loads(mp.read_text());task=manifest['tasks'][0];assert task['task']==1
    fp=ROOT/task['files']['train']['path'];assert sha(fp)==task['files']['train']['sha256']
    rows=[json.loads(s) for s in fp.read_text().splitlines()]
    sp=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
    split=json.loads(sp.read_text());fit=set(split['fit_ids']['1'])
    byid={r['id']:r for r in rows if r['id'] in fit};assert len(byid)==327
    groups=[]
    for gid,g in split['groups'].items():
        ii=[i for i in g['ids'] if i in byid]
        if ii:
            assert not g['dev']
            representative=min(ii,key=lambda i:htext('42:visual-function-representative:'+i))
            groups.append((gid,representative,ii))
    groups.sort(key=lambda g:htext('42:visual-function-s1:'+g[0]))
    assert len(groups)==325 and sum(len(x[2]) for x in groups)==327
    diagnostic=groups[:65];calibration=groups[65:]
    ledger=dict(manifest_sha256=sha(mp),TRAIN_S1_sha256=sha(fp),old_SPLIT_sha256=sha(sp),
        calibration=[dict(group=g,id=i,group_QA_ids=ii) for g,i,ii in calibration],
        diagnostic=[dict(group=g,id=i,group_QA_ids=ii) for g,i,ii in diagnostic],
        original_S1_FIT_QA=327,known_groups=325,calibration_QA=260,diagnostic_QA=65,
        updates=17,official_TEST_QA=0,original_DEV_QA=0,future_TRAIN_files_opened=0,
        not_independent_images_or_cross_domain_validation=True)
    selected=[byid[i] for _,i,_ in groups]
    ledger['selected_image_sha256']={image:split['image_sha256'][image] for row in selected for image in row['images']}
    assert not {g for g,_,_ in calibration}&{g for g,_,_ in diagnostic}
    return task,[byid[i] for _,i,_ in calibration],[byid[i] for _,i,_ in diagnostic],ledger

def teacher_layers(model):return model.core.vision_tower.vision_model.encoder.layers

def make_student(model):
    import torch
    for p in model.parameters():p.requires_grad_(False)
    layers=teacher_layers(model);assert len(layers)==26
    student=copy.deepcopy(layers[PREFIX])
    assert not {id(p) for p in student.parameters()}&{id(p) for p in model.parameters()}
    return student

@contextlib.contextmanager
def shallow(model,student):
    import torch
    encoder=model.core.vision_tower.vision_model.encoder
    original=encoder.layers
    encoder.layers=torch.nn.ModuleList(list(original[:PREFIX])+[student])
    try:yield
    finally:encoder.layers=original

def extract_student(model,student,raw_pixels,sizes):
    # Bypass the frozen visual-cache wrapper: gradients must reach the small tail.
    with shallow(model,student):
        return model._llmcl_visual_original(raw_pixels,grid_thw=sizes)

def normalized_error(student,teacher):
    # Fixed per-QA normalization; each known group contributes one representative.
    return (student.float()-teacher.float()).square().mean()/teacher.float().square().mean().clamp_min(1e-8)

def digest(parameters):
    h=hashlib.sha256()
    for p in parameters:
        a=p.detach().cpu().contiguous();h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode())
        h.update(a.view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()

def tensor_digest(mapping):
    return {k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=digest([v]))
            for k,v in mapping.items() if hasattr(v,'detach')}

def count(model):
    return dict(coefficients=sum(p.numel() for p in model.parameters()),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))

def cleanup_compact(model,student):
    import torch
    vm=model.core.vision_tower.vision_model
    vm.encoder.layers=torch.nn.ModuleList(list(vm.encoder.layers[:PREFIX])+[student])
    model.core.vision_tower.config.num_hidden_layers=14
    vm.config.num_hidden_layers=14
    model.core.config.vision_config.num_hidden_layers=14
    for p in model.parameters():p.requires_grad_(False)
    model._llmcl_visual_cache.clear();model._llmcl_visual_cache_bytes=0
    assert len(teacher_layers(model))==14
    assert not any(int(n.split('encoder.layers.')[1].split('.')[0])>=14
                   for n,_ in model.named_parameters() if 'encoder.layers.' in n)

def outcome(before,after,layout):
    assert [r['id'] for r in before]==[r['id'] for r in after]
    mean=lambda xs:math.fsum(xs)/len(xs)
    parent=mean([r['parent_NLL'] for r in before])
    b=mean([r['student_NLL'] for r in before]);a=mean([r['student_NLL'] for r in after])
    g0=b-parent;g1=a-parent
    e0=mean([r['feature_error'] for r in before]);e1=mean([r['feature_error'] for r in after])
    passed=layout['coefficients']<=.8*BASE and g0>0 and g1<=.5*g0 and e1<e0
    return dict(parent_mean_NLL=parent,uncorrected_mean_NLL=b,corrected_mean_NLL=a,
        initial_gap=g0,remaining_gap=g1,closed_fraction=(g0-g1)/g0 if g0>0 else None,
        before_mean_normalized_feature_MSE=e0,after_mean_normalized_feature_MSE=e1,
        actual_final_layout=layout,point_gate_passed=passed,
        next_action='necessary_review_only' if passed else 'close_this_fixed_probe',
        threshold_half_gap_is_investment_gate_not_quality_tolerance=True,
        no_Answer_F1_or_generalization_claim=True,automatic_full_training=False)
