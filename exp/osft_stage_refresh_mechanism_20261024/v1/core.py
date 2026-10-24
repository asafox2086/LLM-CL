"""Original OSFT with only the declared stage refresh switch and boundary observations."""
import hashlib,time,sys
import torch
from common import PORT
sys.path.insert(0,str(PORT))
from osft_adapter import OSFTMethod


def rng_digest(state):
    out=hashlib.sha256(repr(state['python']).encode())
    for x in state['numpy']:
        out.update(x.tobytes() if hasattr(x,'tobytes') else repr(x).encode())
    out.update(digest([('cpu',state['cpu'])]+[(f'cuda{i}',t) for i,t in enumerate(state['cuda'])]).encode())
    return out.hexdigest()


class StageMethod(OSFTMethod):
    def __init__(self,model,arm):
        assert arm in ('prefix','Fixed','Refresh')
        super().__init__(model,'OSFT',['unused'])
        self.arm=arm;self.boundary_rows=[];self.write_boundary=None

    @staticmethod
    def actual_weight(layer):
        with torch.no_grad(),torch.autocast(layer.U_low.device.type,dtype=layer.output_dtype,enabled=layer.U_low.is_cuda):
            return layer.kernel['reconstruct_weight_matrix'](layer.svd(),torch.float32,layer.output_dtype).detach()

    @staticmethod
    def projector_change(a,b):
        # ||AA^T-BB^T||_F^2 via small Grams, even if floating columns are not perfectly orthogonal.
        with torch.no_grad(),torch.autocast(a.device.type,enabled=False):
            a=a.float();b=b.float();aa=a.T@a;bb=b.T@b;ab=a.T@b
            raw=float(aa.double().square().sum()+bb.double().square().sum()-2*ab.double().square().sum())
            eye=torch.eye(aa.shape[0],device=aa.device)
            return dict(raw_squared_projector_distance=raw,nonnegative_estimate=max(raw,0.),
                squared_projector_normalizer=float(aa.double().square().sum()+bb.double().square().sum()),
                gram_error_before=float((aa-eye).norm()),gram_error_after=float((bb-eye).norm()),
                products='FP32',scalar_reductions='FP64',not_sign_sensitive=True)

    def begin_stage(self,stage):
        if self.stage==0:
            assert self.arm=='prefix' and stage==1
            return super().begin_stage(stage)
        assert self.arm!='prefix' and self.stage==stage-1
        prior=self.stage;rows=[];started=time.monotonic()
        for i,layer in enumerate(self.layers):
            with torch.no_grad():
                before=self.actual_weight(layer);old_u=layer.U_high.detach().clone();old_v=layer.V_high.detach().clone()
                before_state=digest([(name,getattr(layer,name)) for name in ('U_high','S_high','V_high','U_low','S_low','V_low')])
                if self.arm=='Refresh':
                    # Exactly the original stage-boundary reconstruction context, outside outer TRAIN autocast.
                    raw=layer.kernel['reconstruct_weight_matrix'](layer.svd(),torch.float32,layer.output_dtype).detach()
                    layer.refactor(raw);del raw
                after=self.actual_weight(layer);delta=after.float()-before.float()
                row=dict(layer=i,shape=list(before.shape),refactor_called=self.arm=='Refresh',
                    actual_FP16_changed_coordinates=int((before!=after).sum()),
                    actual_FP16_squared_change=float(delta.double().square().sum()),
                    actual_FP16_before_energy=float(before.double().square().sum()),
                    U_high=self.projector_change(old_u,layer.U_high),V_high=self.projector_change(old_v,layer.V_high),
                    factor_state_before=before_state,
                    factor_state_after=digest([(name,getattr(layer,name)) for name in ('U_high','S_high','V_high','U_low','S_low','V_low')]))
                assert torch.isfinite(before).all() and torch.isfinite(after).all()
                if self.arm=='Fixed':assert row['actual_FP16_changed_coordinates']==0 and row['factor_state_before']==row['factor_state_after']
                rows.append(row);del before,after,delta,old_u,old_v
        self.stage=stage
        record=dict(stage=stage,prior_method_stage=prior,arm=self.arm,modules=len(rows),
            actual_refactor_calls=sum(x['refactor_called'] for x in rows),matrices=rows,seconds=time.monotonic()-started,
            package_intervention_not_pure_subspace_or_optimizer_causality=True)
        self.boundary_rows.append(record)
        assert self.write_boundary is not None;self.write_boundary(stage,record)

    def method_state(self,stage):
        return super().method_state(stage)|dict(arm=self.arm,boundary_refactor_enabled=self.arm=='Refresh')

def digest(items):
    out=hashlib.sha256()
    for name,t in sorted(items):
        t=t.detach().cpu().contiguous()
        out.update(name.encode());out.update(str((t.dtype,tuple(t.shape))).encode())
        out.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return out.hexdigest()


def layout(model):
    def count(items):
        groups={}
        for _,t in items:
            if t.numel()==0:continue
            st=t.untyped_storage();key=(str(t.device),st.data_ptr(),st.nbytes())
            entry=groups.setdefault(key,dict(element=t.element_size(),ranges=[]))
            assert entry['element']==t.element_size(),'Mixed-dtype storage is unsupported'
            axes=sorted((stride,size) for size,stride in zip(t.shape,t.stride()) if size>1)
            step=1;dense=True
            for stride,size in axes:
                if stride!=step:dense=False;break
                step*=size
            off=t.storage_offset()
            if dense:ranges=[(off,off+t.numel())]
            elif t.ndim==2 and min(t.stride())==1:
                axis=0 if t.stride(0)==1 else 1;other=1-axis
                width=t.shape[axis];pitch=t.stride(other);assert pitch>=width
                ranges=[(off+i*pitch,off+i*pitch+width) for i in range(t.shape[other])]
            else:raise AssertionError('Unsupported noncontiguous weight layout')
            assert all(0<=lo<hi<=st.nbytes()//t.element_size() for lo,hi in ranges)
            entry['ranges'].extend(ranges)
        n=b=backing=0
        for key,entry in groups.items():
            ranges=sorted(entry['ranges']);lo,hi=ranges[0];used=0
            for a,z in ranges[1:]:
                if a<=hi:hi=max(hi,z)
                else:used+=hi-lo;lo,hi=a,z
            used+=hi-lo;n+=used;b+=used*entry['element'];backing+=key[2]
        return dict(coefficients=n,bytes=b,backing_storage_bytes=backing,unused_backing_bytes=backing-b)
    high={'U_high','S_high','V_high'}
    weight_buffers=[(n,p) for n,p in model.core.named_buffers() if n.rsplit('.',1)[-1] in high]
    ordinary=[(n,p) for n,p in model.core.named_buffers() if n.rsplit('.',1)[-1] not in high]
    return dict(weights=count(list(model.core.named_parameters())+weight_buffers),ordinary_buffers=count(ordinary),
                trainable=sum(p.numel() for p in model.core.parameters() if p.requires_grad))


def logical_layout(value):
    return {kind:{k:v for k,v in item.items() if k not in ('backing_storage_bytes','unused_backing_bytes')}
            if isinstance(item,dict) else item for kind,item in value.items()}


def cpu_state(model):
    out={};seen={}
    for n,t in model.core.state_dict().items():
        key=(t.untyped_storage().data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()),t.dtype)
        if key not in seen:seen[key]=t.detach().cpu().clone()
        out[n]=seen[key]
    return out
