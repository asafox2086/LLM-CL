"""TaskGradReuse: task-gradient SVD A initialization, B0, native SeqLoRA.

Only stage-boundary initialization and the S19 dense merge are new. Native
answer loss, optimizer, minibatch order, schedule and evaluation are untouched.
The default CUDA SVD output is used directly; exact orthogonality is not claimed.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from methods import Method

RANK = 8
ALPHA = 32
SCALE = 4
DROPOUT = .1
KAPPA = 1 / math.sqrt(3)
COORDINATES = 540672


def _configure_math():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def _digest(named):
    """CPU byte I/O only; no CPU model arithmetic or retained tensor copies."""
    h = hashlib.sha256()
    for name, value in named:
        h.update(str((name,tuple(value.shape),str(value.dtype))).encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().reshape(-1)
        for start in range(0,raw.size,8*1024**2):
            h.update(raw[start:start+8*1024**2].tobytes())
        del raw
    return h.hexdigest()


def _write_new(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


class TaskGradReuseMethod(Method):
    def __init__(self, model, name, instructions, embedding=None):
        assert name == 'TaskGradReuse'
        super().__init__(model,'SeqLoRA',instructions,embedding)
        self.name = name
        self.original_revision = 'task-current-default-CUDA-SVD-right-A-r8-B0-native-v1'
        self.learning_rate = 1e-4
        self.host_optimizer = False
        self.offload_activations = False
        self.vision_original = None
        self.rank_plan = {f'model.layers.{i}.self_attn.{p}_proj':8
                          for i in range(24) for p in ('q','v')}
        self.completed_stage = 0
        self.initialized_stage = 0
        self.active_stage = None
        self.deployed = False
        self.state_root = None
        self._processor = None
        self._manifest = None
        self._train_rows = {}
        self._preflight = None
        self._history = []  # JSON scalars/digests only; never G, factors or model copies.
        _configure_math()
        self._check_AB()

    def _check_AB(self):
        assert not self.deployed and len(self.layers) == 48
        expected = {n+s for n in self.rank_plan for s in ('.A.weight','.B.weight')}
        trainable = {n:p for n,p in self.model.named_parameters() if p.requires_grad}
        assert not trainable or set(trainable) == expected
        assert sum(self.model.get_submodule(n).A.weight.numel()+
                   self.model.get_submodule(n).B.weight.numel() for n in self.rank_plan) == COORDINATES
        for name in self.rank_plan:
            layer = self.model.get_submodule(name)
            assert layer.weight.is_cuda and layer.weight.dtype == torch.float16
            assert layer.weight.shape == (896 if name.endswith('q_proj') else 128,896)
            assert layer.A.weight.shape == (8,896)
            assert layer.B.weight.shape == (layer.weight.shape[0],8)
            assert layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
            assert layer.A.weight.device == layer.B.weight.device == layer.weight.device
            assert layer.scale == SCALE and layer.dropout.p == DROPOUT

    def bind_context(self, processor, manifest, data, run, preflight=False):
        """TRAIN context only. Full method checkpoints preserve cross-stage use.

        The native loader already rebinds state_root after relocating a run.
        A caller may also rebind this context explicitly without resetting stage.
        """
        assert manifest['condition'] == 'pct01' and manifest['seed'] == 42
        assert [t['task'] for t in manifest['tasks']] == list(range(1,20))
        assert sum(t['files']['train']['count'] for t in manifest['tasks']) == 1313
        self.state_root = Path(run)/'method_state'
        self._processor = processor
        self._manifest = manifest
        self._preflight = bool(preflight)
        self._train_rows = {} if self.deployed else {
            stage:list(data[stage,'train']) for stage in range(1,20)}
        for stage,rows in self._train_rows.items():
            assert len(rows) == manifest['tasks'][stage-1]['files']['train']['count']
            assert len({r['id'] for r in rows}) == len(rows)

    def _state_digest(self):
        core = getattr(self.model,'core',self.model)
        return dict(parameters=_digest(core.named_parameters(remove_duplicate=False)),
                    buffers=_digest(core.named_buffers(remove_duplicate=False)))

    def _fold(self, final=False):
        self._check_AB();_configure_math()
        records = []
        for name in self.rank_plan:
            layer = self.model.get_submodule(name)
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                a,b = layer.A.weight.detach(),layer.B.weight.detach()
                assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(layer.weight).all()
                a_sha,b_sha = _digest([('A',a)]),_digest([('B',b)])
                before = _digest([('weight',layer.weight)])
                merged = layer.weight.detach().float()+SCALE*(b.float()@a.float())
                weight = merged.to(layer.weight.dtype)
                assert torch.isfinite(merged).all() and torch.isfinite(weight).all()
                error = weight.float()-merged
                records.append(dict(module=name,A_digest=a_sha,B_digest=b_sha,
                    base_before_digest=before,base_after_digest=_digest([('weight',weight)]),
                    merge_dtype='torch.float32',output_dtype=str(weight.dtype),device=str(weight.device),
                    scale=SCALE,autocast_enabled=False,matmul_allow_tf32=False,
                    rounding_max_abs=float(error.abs().max()),
                    rounding_rms=float(error.square().mean().sqrt())))
                if final:
                    # Meta allocation consumes no RNG and no discarded dense GPU matrix.
                    with torch.device('meta'):
                        replacement = torch.nn.Linear(weight.shape[1],weight.shape[0],
                            bias=layer.bias is not None,dtype=weight.dtype)
                    replacement.weight = torch.nn.Parameter(weight,requires_grad=False)
                    if layer.bias is not None:replacement.bias = layer.bias
                    parent,child = name.rsplit('.',1)
                    setattr(self.model.get_submodule(parent),child,replacement)
                else:
                    layer.weight.copy_(weight)
                    layer.B.weight.zero_()
                    layer.A.weight.grad = None;layer.B.weight.grad = None
            del a,b,merged,weight,error
        if final:
            self.layers.clear()
            self.deployed = True
            self.model.requires_grad_(False).eval()
            assert not any(n.endswith(('.A.weight','.B.weight')) for n,_ in self.model.named_parameters())
        return records

    def _estimate(self, stage):
        from runtime_utils import input_batch, answer_loss, rng, restore_rng
        from frozen_vision import prepare
        import onevision_bridge
        assert onevision_bridge.ACTIVE is self.model
        rows = self._train_rows[stage][:min(16,len(self._train_rows[stage]))]
        assert rows and self._processor is not None
        task = self._manifest['tasks'][stage-1]
        parameters = list(self.model.parameters())
        masks = [(p,p.requires_grad) for p in parameters]
        core = getattr(self.model,'core',self.model)
        modules = {id(m):m for m in list(self.model.modules())+list(core.modules())}
        modes = [(m,m.training) for m in modules.values()]
        saved_rng = rng();state_before = self._state_digest()
        targets = {n:self.model.get_submodule(n).weight for n in self.rank_plan}
        assert len({id(p) for p in targets.values()}) == 48
        cache = self.model._llmcl_visual_cache
        old_cache = list(cache.items())  # CPU feature I/O references, no activation arithmetic.
        cache_fields = ('_llmcl_visual_cache_bytes','_llmcl_visual_cache_hits','_llmcl_visual_cache_misses')
        cache_state = {n:getattr(self.model,n) for n in cache_fields}
        native_last = onevision_bridge.LAST_NATIVE
        started = time.monotonic();losses = [];gradients = None
        try:
            self.model.requires_grad_(False).eval();self.model.zero_grad(set_to_none=True)
            for p in targets.values():
                assert p.is_cuda and p.dtype == torch.float16
                p.data = p.data.float();p.requires_grad_(True)
            for row in rows:
                batch,_ = input_batch(row,task,self._processor,self.model.device)
                prepare(batch,row,self.model,self.vision_original)
                with torch.autocast('cuda',dtype=torch.float16):
                    loss,tokens = answer_loss(self.model,batch,row['answer'],self._processor)
                assert torch.isfinite(loss) and tokens > 0
                losses.append(float(loss.detach()))
                (loss/len(rows)).backward()
                del loss,batch
            gradients = {}
            for name,p in targets.items():
                assert p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
                gradients[name] = p.grad.detach().clone()
        finally:
            self.model.zero_grad(set_to_none=True)
            for p in targets.values():p.data = p.data.half()
            for p,flag in masks:p.requires_grad_(flag)
            for module,flag in modes:module.training = flag
            cache.clear();cache.update(old_cache)
            for n,value in cache_state.items():setattr(self.model,n,value)
            onevision_bridge.LAST_NATIVE = native_last
            restore_rng(saved_rng)
        assert self._state_digest() == state_before
        assert all(p.grad is None for p in parameters)
        return gradients,dict(QA=len(rows),backwards=len(rows),optimizer_updates=0,
            ordered_IDs_sha256=hashlib.sha256(''.join(r['id']+'\n' for r in rows).encode()).hexdigest(),
            answer_losses=losses,QA_equal_weight=True,current_task_TRAIN_only=True,
            parameter_and_all_buffer_state_restored=True,modes_masks_dtype_RNG_restored=True,
            native_feature_cache_restored=True,gradient_dtype='torch.float32',forward_autocast='torch.float16',
            dense_gradient_coordinates=sum(g.numel() for g in gradients.values()),
            gradient_bytes=sum(g.numel()*g.element_size() for g in gradients.values()),
            seconds=time.monotonic()-started)

    def _initialize(self, gradients):
        self._check_AB();assert set(gradients) == set(self.rank_plan)
        records = []
        for name in self.rank_plan:
            layer = self.model.get_submodule(name);g = gradients.pop(name)
            assert g.is_cuda and g.device == layer.weight.device and g.shape == layer.weight.shape
            assert g.dtype == torch.float32 and not g.requires_grad and torch.isfinite(g).all()
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                # No driver override, QR, row renormalization or numerical-error cutoff.
                u,s,vh = torch.linalg.svd(g,full_matrices=False)
                assert torch.isfinite(s).all() and torch.isfinite(vh).all()
                basis = vh[:RANK];a = (KAPPA*basis).contiguous()
                assert a.shape == layer.A.weight.shape and torch.isfinite(a).all()
                layer.A.weight.copy_(a);layer.B.weight.zero_()
                layer.A.weight.grad = None;layer.B.weight.grad = None
                energy = s.square().sum();top = s[:RANK].square().sum()
                assert torch.isfinite(energy) and torch.isfinite(top)
                records.append(dict(module=name,gradient_shape=list(g.shape),gradient_digest=_digest([('G',g)]),
                    singular_values_top_r=s[:RANK].tolist(),total_squared_singular_energy=float(energy),
                    top_r_squared_singular_energy=float(top),
                    top_r_energy_fraction=float(top/energy) if float(energy)>0 else 0.,
                    A_digest=_digest([('A',layer.A.weight)]),B_digest=_digest([('B',layer.B.weight)]),
                    B_zero=True,A_row_norm_target=KAPPA,
                    A_row_norm_max_error=float((a.norm(dim=1)-KAPPA).abs().max()),
                    basis_orthogonality_max_error=float((basis@basis.T-torch.eye(RANK,device=g.device)).abs().max()),
                    svd_dtype='torch.float32',svd_driver='default',full_matrices=False,
                    hard_orthogonality_or_row_norm_cutoff=False))
            del g,u,s,vh,basis,a,energy,top
        assert not gradients
        return records

    def begin_stage(self, stage):
        from runtime_utils import rng, restore_rng
        assert not self.deployed and self._manifest is not None
        assert type(stage) is int and 1 <= stage <= 19
        assert self.active_stage is None and stage == self.completed_stage+1
        _configure_math();started = time.monotonic();saved_rng = rng()
        gradients = None
        try:
            folded = self._fold(final=False)
            gradients,calibration = self._estimate(stage)
            direction = self._initialize(gradients)
        finally:
            if gradients is not None:gradients.clear()
            restore_rng(saved_rng)
        self.active_stage = self.initialized_stage = stage
        record = dict(stage=stage,preflight_only=self._preflight,fold=folded,
            calibration=calibration,direction=direction,main_optimizer_coordinates=COORDINATES,
            cumulative_delta_rank_not_bounded_by_r8=True,begin_stage_RNG_restored=True,
            seconds=time.monotonic()-started)
        self._history.append(record)
        _write_new(self.state_root/f'task_gradient_stage_{stage:02d}.json',record)
        print(json.dumps(dict(event='task_gradient_stage_ready',stage=stage,
            preflight_only=self._preflight,calibration_backwards=calibration['backwards'],
            calibration_dense_coordinates=calibration['dense_gradient_coordinates'],
            gradient_bytes=calibration['gradient_bytes'],seconds=record['seconds'])),flush=True)

    def end_stage(self, stage):
        assert not self._preflight, 'Native preflight does not complete any formal stage'
        assert self.active_stage == stage and stage == self.completed_stage+1
        if stage == 19:
            started = time.monotonic();folded = self._fold(final=True)
            _write_new(self.state_root/'task_gradient_final_fold.json',
                dict(stage=19,modules=folded,deployed=True,seconds=time.monotonic()-started))
            self._train_rows.clear()
            self._processor = None
            self._manifest = None
        self.completed_stage = stage;self.active_stage = None

    def changed_state(self):
        # Unlike ordinary LoRA, previous stages already changed these dense W.
        wanted = {n+'.weight' for n in self.rank_plan}
        if not self.deployed:wanted |= {n+s for n in self.rank_plan for s in ('.A.weight','.B.weight')}
        state = self.model.state_dict();assert wanted <= set(state)
        return {n:state[n].detach().cpu().clone() for n in sorted(wanted)}

    def method_state(self, stage):
        return dict(method=self.name,requested_checkpoint_stage=stage,completed_stage=self.completed_stage,
            initialized_stage=self.initialized_stage,active_stage=self.active_stage,deployed=self.deployed,
            preflight_only=self._preflight,rank=RANK,alpha=ALPHA,scale=SCALE,dropout=DROPOUT,
            main_optimizer_coordinates=0 if self.deployed else COORDINATES,
            structural_AB_coordinates=0 if self.deployed else COORDINATES,
            calibration_dense_coordinates=22020096,calibration_gradient_bytes=88080384,
            completed_calibration_backwards=sum(r['calibration']['backwards'] for r in self._history),
            calibration_events=len(self._history),raw_default_SVD=True,QR_or_row_renormalization=False,
            target_A_row_norm=KAPPA,orthogonality_errors_recorded_not_admission_thresholds=True,
            permanent_G_or_factor_bank=False,cumulative_delta_rank_not_bounded_by_r8=True,
            optimizer_and_answer_loss='unchanged native runner',original_revision=self.original_revision,
            checkpoint_resume='full method graph and native RNG; TRAIN context contains no TEST rows')
