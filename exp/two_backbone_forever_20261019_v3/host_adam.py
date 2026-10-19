"""Stock AdamW arithmetic with moment tensors retained in CPU memory."""
import torch


class HostAdamW(torch.optim.AdamW):
    def __init__(self,*args,**kwargs):
        kwargs['foreach']=False
        super().__init__(*args,**kwargs)

    def offload(self):
        for state in self.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):
                    state[key]=value.cpu()

    def load_state_dict(self,state):
        super().load_state_dict(state)
        self.offload()

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:
            raise ValueError('HostAdamW closure is not supported')
        original=[group['params'] for group in self.param_groups]
        try:
            for group in self.param_groups:
                group['params']=[]
            for group,parameters in zip(self.param_groups,original):
                for parameter in parameters:
                    if parameter.grad is None:
                        continue
                    state=self.state[parameter]
                    for key in ['exp_avg','exp_avg_sq','max_exp_avg_sq']:
                        if key in state:
                            state[key]=state[key].to(parameter.device)
                    group['params']=[parameter]
                    super().step()
                    group['params']=[]
                    for key,value in state.items():
                        if torch.is_tensor(value):
                            state[key]=value.cpu()
        finally:
            for group,parameters in zip(self.param_groups,original):
                group['params']=parameters
