"""Fixed four-arm residual-exit pilot. GPU FP64 QA-weighted sufficient statistics.
No generation, score, model loader, optimizer, automatic damping or retry.
W is torch Linear layout [out,in]; bias is fixed. Only the selected existing W changes.
"""
import torch

def solve_delta(gram, rhs, damping):
    """Ridge delta solve; no fallback/jitter/sweep. Separate tiny-math tests allowed."""
    if gram.dtype!=torch.float64 or rhs.dtype!=torch.float64:raise TypeError('FP64 solve')
    if gram.ndim!=2 or gram.shape[0]!=gram.shape[1] or rhs.shape[0]!=gram.shape[0]:raise ValueError('shape')
    if not damping>0:raise ValueError('positive preregistered damping')
    system=gram+torch.eye(gram.shape[0],dtype=gram.dtype,device=gram.device)*damping
    L,info=torch.linalg.cholesky_ex(system)
    if int(info)!=0:raise ArithmeticError('Cholesky failed; no automatic damping change')
    delta=torch.cholesky_solve(rhs,L)
    if not bool(torch.isfinite(delta).all()):raise ArithmeticError('nonfinite ridge update')
    residual=system@delta-rhs
    return delta,dict(normal_equation_relative_residual=float(torch.linalg.vector_norm(residual)/(torch.linalg.vector_norm(rhs)+torch.finfo(torch.float64).tiny)))

class ExitRidge:
    def __init__(self, prior_weight, fixed_bias, expected_ids):
        if prior_weight.dtype!=torch.float16 or not prior_weight.is_cuda:raise TypeError('original FP16 GPU prior required')
        self.prior=prior_weight.detach().double().clone();self.dtype=prior_weight.dtype
        self.out,self.width=self.prior.shape
        self.bias=torch.zeros(self.out,device=self.prior.device,dtype=torch.float64) if fixed_bias is None else fixed_bias.detach().double().clone()
        self.expected_ids=list(expected_ids);self.seen=[]
        self.Gt=torch.zeros(self.width,self.width,device=self.prior.device,dtype=torch.float64)
        self.Gs=torch.zeros_like(self.Gt)
        self.R={a:torch.zeros(self.width,self.out,device=self.prior.device,dtype=torch.float64) for a in ('B','C','D')}
        self.target_energy={a:0. for a in ('B','C','D')}
    def add_qa(self,qa_id,teacher,student):
        if len(self.seen)>=len(self.expected_ids) or qa_id!=self.expected_ids[len(self.seen)]:raise ValueError('frozen QA order')
        Xt=teacher['down_input'].double();Xs=student['down_input'].double()
        Yt=teacher['down_output'].double();Ot=teacher['block_output'].double();Rs=student['skip'].double()
        if Xt.shape!=Xs.shape or Xt.ndim!=2 or Xt.shape[1]!=self.width or Xt.shape[0]<1:raise ValueError('input shape')
        if any(v.shape!=(Xt.shape[0],self.out) for v in (Yt,Ot,Rs)):raise ValueError('output shape')
        if any(not v.is_cuda or not bool(torch.isfinite(v).all()) for v in (Xt,Xs,Yt,Ot,Rs)):raise ValueError('GPU finite actual captures')
        if teacher['positions']!=student['positions'] or teacher['kinds']!=student['kinds']:raise ValueError('paired causal positions')
        T=Xt.shape[0];targets={'B':Yt-self.bias,'C':Yt-self.bias,'D':Ot-Rs-self.bias}
        self.Gt.add_(Xt.T@Xt/T);self.Gs.add_(Xs.T@Xs/T)
        for arm,X in [('B',Xt),('C',Xs),('D',Xs)]:
            residual=targets[arm]-X@self.prior.T
            self.R[arm].add_(X.T@residual/T)
            self.target_energy[arm]+=float(residual.square().sum()/T)
        self.seen.append(qa_id)
    def finish(self):
        if self.seen!=self.expected_ids or len(self.seen)!=80:raise ValueError('exactly frozen 80 QA required')
        N=len(self.seen);Gt=self.Gt/N;Gs=self.Gs/N
        damping=.01*float(torch.trace(Gt)/self.width)
        candidates={'A':self.prior.to(self.dtype)};certificates={}
        for arm,G in [('B',Gt),('C',Gs),('D',Gs)]:
            deltaT,certificate=solve_delta(G,self.R[arm]/N,damping)
            fitted=self.prior+deltaT.T;cast=fitted.to(self.dtype)
            if not bool(torch.isfinite(cast).all()):raise ArithmeticError('nonfinite cast weight')
            rhs=self.R[arm]/N;initial=self.target_energy[arm]/N
            def objective(change):
                DT=change.T;cross=-2*float((DT*rhs).sum());quad=float((DT*(G@DT)).sum())
                residual=initial+cross+quad;penalty=damping*float(DT.square().sum())
                absolute_scale=abs(initial)+2*float((DT*rhs).abs().sum())+float((DT.abs()*(G.abs()@DT.abs())).sum())
                eps=torch.finfo(torch.float64).eps;steps=self.width+N+4
                scale=float(32*(steps*eps)/(1-steps*eps)*absolute_scale)
                return dict(ideal_linear_residual_raw=residual,ridge_penalty=penalty,
                    regularized_objective_raw=residual+penalty,terms=[initial,cross,quad],
                    FP64_reduction_scale=scale,residual_negative_beyond_scale=bool(residual < -scale),
                    includes_actual_FP16_GEMM=False)
            certificate.update(precast_fit=objective(fitted-self.prior),
                cast_fit=objective(cast.double()-self.prior),
                lambda_value=damping,lambda_rule='0.01*meanDiag(equal-QA teacher Gram)',
                pre_cast_update_squared_norm=float((fitted-self.prior).square().sum()),
                post_cast_update_squared_norm=float((cast.double()-self.prior).square().sum()),
                rounding_squared_norm=float((cast.double()-fitted).square().sum()),
                changed_FP16_coefficients=int((cast!=self.prior.to(self.dtype)).sum()),
                baseline_fit_target_residual_energy=self.target_energy[arm]/N)
            candidates[arm]=cast;certificates[arm]=certificate
        return candidates,dict(QA=80,lambda_value=damping,teacher_gram_mean_diagonal=float(torch.trace(Gt)/self.width),
            fitted_existing_weight_coefficients=self.prior.numel(),additional_deployment_parameters=0,
            fixed_bias=True,candidate_weight_dtype=str(self.dtype),arms=certificates)
