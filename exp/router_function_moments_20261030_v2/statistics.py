"""DRAFT: CPU-only sufficient statistics. No model, hook, data reader, score or solver.
Caller supplies actual, already-selected causal prediction positions of ONE QA/layer.
Experts include every inference scale/mask; gates are actual effective composite gates.
"""
from __future__ import annotations
import numpy as np

def _finite(name, value):
    x=np.asarray(value, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(name+': nonfinite')
    return x

def collect_qa(*, qa_id, domain, layer_id, experts, gates, base,
               actual_update, prediction_positions, position_kinds,
               closure_atol, closure_rtol):
    """All numerical inputs must be detached CPU arrays; no model modification.
    Shapes: experts[T,E,D], gates[T,E], base/actual_update[T,D].
    Each selected prediction position appears once. 'eos' is a target kind,
    NOT an instruction to select the EOS input position (causal shift required).
    T means selected nonpadding original-loss prediction positions, including EOS.
    The caller, not this function, must prove loss-mask alignment and real gates.
    """
    u=_finite('experts',experts); g=_finite('gates',gates)
    b=_finite('base',base); native=_finite('actual_update',actual_update); t=native
    if u.ndim!=3 or min(u.shape)<1:
        raise ValueError('nonempty T,E,D required')
    n,e,d=u.shape
    if g.shape!=(n,e) or b.shape!=(n,d) or t.shape!=(n,d):
        raise ValueError('shape mismatch')
    p=list(prediction_positions); kinds=list(position_kinds)
    if len(p)!=n or len(set(p))!=n or any(type(x) is not int or x<0 for x in p):
        raise ValueError('unique integer prediction positions required')
    if len(kinds)!=n or any(x not in ('answer','eos') for x in kinds):
        raise ValueError('position kind mismatch')
    if not qa_id or not domain or not layer_id:
        raise ValueError('explicit QA/domain/layer identity required')
    if not np.isfinite([closure_atol,closure_rtol]).all() or min(closure_atol,closure_rtol)<0:
        raise ValueError('adapter-declared numerical tolerances required')
    if (g < -1e-7).any() or np.max(np.abs(g.sum(axis=1)-1.0))>1e-5:
        raise ValueError('actual effective simplex gates required')
    reconstructed=np.einsum('ted,te->td',u,g)
    err=native-reconstructed
    operations=2*e+2; unit=2.0**-24; gamma=operations*unit/(1-operations*unit)
    allowed=gamma*np.sum(np.abs(u*g[:,:,None]),axis=1)+operations*(2.0**-126)
    t=reconstructed # Ideal FP64 target; native reconstruction kept separately.
    gc=g-g.mean(0)
    ga=np.concatenate((g,(g.sum(1)-1)[:,None]),axis=1);gam=ga.mean(0);gac=ga-gam
    # FP64 statistics of actual finite-precision outputs, not a new model forward.
    H=np.einsum('ted,tfd->ef',u,u)/n
    h=np.einsum('ted,td->e',u,t)/n
    a=float(np.einsum('td,td->',t,t)/n)
    mu=u.mean(axis=1)
    v=u-mu[:,None,:]
    centered_target=t-mu
    augmented=np.concatenate((v,mu[:,None,:]),axis=1)
    centered_H=np.einsum('ted,tfd->ef',augmented,augmented)/n
    centered_h=np.einsum('ted,td->e',augmented,centered_target)/n
    centered_a=float(np.einsum('td,td->',centered_target,centered_target)/n)
    uniform_residual=t-np.mean(u,axis=1)
    out={'qa_id':str(qa_id),'domain':str(domain),'layer_id':str(layer_id),
         'selected_tokens':n,'expert_count':e,'output_width':d,
         'prediction_positions':p,'position_kinds':kinds,
         'moment_target':'FP64_sum_of_actual_expert_outputs_times_actual_effective_gates; not native FP32 delta',
         'native_operation_update_energy':float(np.sum(native*native)/n),
         'native_vs_ideal_update_squared_error':float(np.sum(err*err)/n),
         'gate_covariance_within_QA':(gc.T@gc/n).tolist(),
         'gate_augmented_mean':gam.tolist(),'gate_augmented_covariance_within_QA':(gac.T@gac/n).tolist(),
         'g_mean':g.mean(axis=0).tolist(),
         'gg_mean':(g.T@g/n).tolist(),'H':H.tolist(),'h':h.tolist(),'a':a,
         'contrast_energy':float(np.einsum('ted,ted->',v,v)/n),
         'centered_augmented_H':centered_H.tolist(),
         'centered_augmented_h':centered_h.tolist(),
         'centered_a':centered_a,
         'uniform_R_direct':float(np.einsum('td,td->',uniform_residual,uniform_residual)/n),
         'gate_sum_error_max':float(np.max(np.abs(g.sum(axis=1)-1))),
         'base_energy':float(np.einsum('td,td->',b,b)/n),
         'closure':{'rule':'gamma_(2E+2)*sum_abs_Ug+(2E+2)*FP32_min_normal',
                    'operations':operations,'FP32_unit_roundoff':unit,'gamma':gamma,
                    'underflow_absolute_term':operations*(2.0**-126),
                    'max_absolute_bound':float(allowed.max()),
                    'violating_elements':int(np.sum(np.abs(err)>allowed)),
                    'max_violation':float(np.maximum(np.abs(err)-allowed,0).max()),
                    'mean_squared_vector_bound':float(np.sum(allowed*allowed)/n),
                    'all_elements_within_tolerance':bool((np.abs(err)<=allowed).all()),
                    'mean_squared_vector_error':float(np.einsum('td,td->',err,err)/n),
                    'max_absolute_error':float(np.max(np.abs(err)))}}
    return out

def _ratio(num,den):
    # Negative raw energies are never silently changed to zero.
    return float(num/den) if den>0 else None

def describe_layer(rows):
    """One layer, all preregistered QAs. Global mean-gate c is descriptive only.
    Equal QA weighting, including per-domain summaries. Domain c is NOT refit.
    No minimization, clipping, simplex solver, model replacement or quality claim.
    """
    rows=list(rows)
    if not rows:
        raise ValueError('no rows')
    if len({r['layer_id'] for r in rows})!=1 or len({r['expert_count'] for r in rows})!=1:
        raise ValueError('one layer and consistent expert count required')
    if len({r['qa_id'] for r in rows})!=len(rows):
        raise ValueError('duplicate QA/layer')
    if not all(r['closure']['all_elements_within_tolerance'] for r in rows):
        raise ValueError('original-update closure failed; retain raw records, no interpretation')
    c=np.mean([r['g_mean'] for r in rows],axis=0)
    def group(rs):
        H=np.mean([r['H'] for r in rs],axis=0)
        h=np.mean([r['h'] for r in rs],axis=0)
        a=float(np.mean([r['a'] for r in rs]))
        gm=np.mean([r['g_mean'] for r in rs],axis=0)
        gg=np.mean([r['gg_mean'] for r in rs],axis=0)
        # E[(g-c)(g-c)^T], global c even for each domain.
        dd=np.mean([np.array(r['gate_covariance_within_QA'])+np.outer(np.array(r['g_mean'])-c,np.array(r['g_mean'])-c) for r in rs],axis=0)
        terms=(a,-2*float(c@h),float(c@H@c))
        joint=float(sum(terms)); independent=float(np.sum(H*dd.T))
        base=float(np.mean([r['base_energy'] for r in rs]))
        u=np.full(len(c),1/len(c))
        uniform_terms=(a,-2*float(u@h),float(u@H@u))
        # Conservative FP64 arithmetic scale diagnostic, NOT model-equivalence tolerance.
        cq=np.concatenate((c,[c.sum()-1]))
        cH=np.mean([r['centered_augmented_H'] for r in rs],axis=0)
        ch=np.mean([r['centered_augmented_h'] for r in rs],axis=0)
        ca=float(np.mean([r['centered_a'] for r in rs]))
        centered_terms=(ca,-2*float(cq@ch),float(cq@cH@cq))
        centered_R=float(sum(centered_terms))
        gam=np.concatenate((c,[c.sum()-1]))
        augCov=np.mean([np.array(r['gate_augmented_covariance_within_QA'])+np.outer(np.array(r['gate_augmented_mean'])-gam,np.array(r['gate_augmented_mean'])-gam) for r in rs],axis=0)
        centered_independent=float(np.sum(cH*augCov.T))
        roundoff=64*np.finfo(np.float64).eps*sum(abs(v) for v in terms)
        return {'qa_count':len(rs),'selected_tokens':sum(r['selected_tokens'] for r in rs),
                'c_global_mean_gate':c.tolist(),'group_g_mean':gm.tolist(),
                'group_gg_mean':gg.tolist(),'gate_deviation_second_moment':dd.tolist(),
                'gate_deviation_trace_raw':float(np.trace(dd)),
                'mean_H':H.tolist(),'mean_h':h.tolist(),
                'mean_routed_update_energy':a,'mean_base_energy':base,
                'Rjoint_raw':joint,'Rindep_raw':independent,
                'Rjoint_minus_Rindep_raw':joint-independent,
                'centered_Rindep_raw':centered_independent,
                'centered_vs_uncentered_Rindep_difference':centered_independent-independent,
                'native_operation_update_energy':float(np.mean([r['native_operation_update_energy'] for r in rs])),
                'native_vs_ideal_update_squared_error':float(np.mean([r['native_vs_ideal_update_squared_error'] for r in rs])),
                'uniform_R_raw':float(sum(uniform_terms)),
                'Rjoint_terms':list(terms),'Rjoint_fp64_roundoff_scale':float(roundoff),
                'Rjoint_negative_beyond_roundoff':bool(joint < -roundoff),
                'Rjoint_over_update_energy':_ratio(joint,a),
                'Rjoint_over_base_energy':_ratio(joint,base),
                'Rjoint_over_Rindep':_ratio(joint,independent),
                'contrast_energy':float(np.mean([r['contrast_energy'] for r in rs])),
                'uniform_R_direct':float(np.mean([r['uniform_R_direct'] for r in rs])),
                'centered_Rjoint_raw':centered_R,
                'centered_Rjoint_terms':list(centered_terms),
                'centered_vs_uncentered_Rjoint_difference':centered_R-joint,
                'mean_gate_sum_error':float(c.sum()-1),
                'c_is_optimized':False}
    return {'layer_id':rows[0]['layer_id'],'all':group(rows),
            'domains':{d:group([r for r in rows if r['domain']==d])
                       for d in sorted({r['domain'] for r in rows})}}
