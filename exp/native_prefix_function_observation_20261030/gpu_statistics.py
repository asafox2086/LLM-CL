"""GPU FP64 route/expert moments. Small matrices only copied to CPU.
Primary t_ideal=sum_e U_e*g_e in FP64, not the native FP32 delta.
Native-operation reconstruction is separately certified by fixed FP32 error bound.
"""
import torch

def collect_qa(*,qa_id,domain,layer_id,experts,gates,base,actual_update,
               prediction_positions,position_kinds,closure_atol,closure_rtol):
    if any(x.dtype is not torch.float32 for x in (experts,gates,actual_update)):
        raise TypeError('FP32 expert/gate/native-update inputs required for gamma_FP32 certificate')
    U=experts.double();g=gates.double();b=base.double();native=actual_update.double()
    if U.ndim!=3 or min(U.shape)<1:raise ValueError('nonempty T,E,D required')
    n,e,d=U.shape
    if g.shape!=(n,e) or b.shape!=(n,d) or native.shape!=(n,d):raise ValueError('shape mismatch')
    if not all(bool(torch.isfinite(x).all()) for x in (U,g,b,native)):raise ValueError('nonfinite capture')
    if bool((g < -1e-7).any()) or float((g.sum(1)-1).abs().max())>1e-5:raise ValueError('effective simplex gate mismatch')
    p=list(prediction_positions);kinds=list(position_kinds)
    if len(p)!=n or len(set(p))!=n or any(type(x) is not int or x<0 for x in p):raise ValueError('positions')
    if len(kinds)!=n or any(k not in ('native_prediction',) for k in kinds):raise ValueError('kinds')
    # gamma_(2E+2) covers separately rounded FP32 products + accumulated sum.
    # min-normal term also conservatively covers FP32 flush-to-zero underflow.
    t=(U*g[:,:,None]).sum(1);error=(native-t).abs()
    operations=2*e+2;unit=2.0**-24;gamma=(operations*unit)/(1-operations*unit)
    bound=gamma*(U*g[:,:,None]).abs().sum(1)+operations*(2.0**-126)
    violation=(error-bound).clamp_min(0)
    mat=U.permute(1,0,2).reshape(e,-1);vec=t.reshape(-1)
    H=mat@mat.T/n;h=mat@vec/n;a=vec.square().sum()/n
    mu=U.mean(1);V=U-mu[:,None,:];s=t-mu
    aug=torch.cat((V,mu[:,None,:]),dim=1).permute(1,0,2).reshape(e+1,-1)
    cH=aug@aug.T/n;ch=aug@s.reshape(-1)/n;ca=s.square().sum()/n
    gm=g.mean(0);gc=g-gm
    # Welford-compatible within-QA covariance avoids raw gg subtraction loss.
    ga=torch.cat((g, (g.sum(1)-1)[:,None]),1);gam=ga.mean(0);gac=ga-gam
    closure=dict(rule='gamma_(2E+2)*sum_abs_Ug+(2E+2)*FP32_min_normal',
        operations=operations,FP32_unit_roundoff=unit,gamma=gamma,
        underflow_absolute_term=operations*(2.0**-126),
        all_elements_within_tolerance=bool((error<=bound).all()),
        mean_squared_vector_error=float(error.square().sum()/n),
        max_absolute_error=float(error.max()),max_absolute_bound=float(bound.max()),
        max_violation=float(violation.max()),violating_elements=int((error>bound).sum()),
        mean_squared_vector_bound=float(bound.square().sum()/n))
    return dict(qa_id=str(qa_id),domain=str(domain),layer_id=str(layer_id),
        selected_tokens=n,expert_count=e,output_width=d,prediction_positions=p,position_kinds=kinds,
        moment_target='FP64_sum_of_actual_expert_outputs_times_actual_effective_gates; not native FP32 delta',
        g_mean=gm.cpu().tolist(),gg_mean=(g.T@g/n).cpu().tolist(),
        gate_covariance_within_QA=(gc.T@gc/n).cpu().tolist(),
        gate_augmented_mean=gam.cpu().tolist(),gate_augmented_covariance_within_QA=(gac.T@gac/n).cpu().tolist(),
        H=H.cpu().tolist(),h=h.cpu().tolist(),a=float(a),base_energy=float(b.square().sum()/n),
        native_operation_update_energy=float(native.square().sum()/n),
        native_vs_ideal_update_squared_error=float(error.square().sum()/n),
        contrast_energy=float(V.square().sum()/n),
        centered_augmented_H=cH.cpu().tolist(),centered_augmented_h=ch.cpu().tolist(),centered_a=float(ca),
        uniform_R_direct=float(s.square().sum()/n),gate_sum_error_max=float((g.sum(1)-1).abs().max()),
        closure=closure,moment_arithmetic_closure=dict(closure))
