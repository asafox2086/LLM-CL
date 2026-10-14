"""Check CPU-moment AdamW against stock single-tensor CUDA AdamW, including resume."""
import copy
import torch
from host_adam import HostAdamW


def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    left=[torch.nn.Parameter(torch.randn(7,9,device=device)) for _ in range(3)]
    right=[torch.nn.Parameter(p.detach().clone()) for p in left]
    stock=torch.optim.AdamW(left,lr=1e-4,weight_decay=.01,foreach=False)
    host=HostAdamW(right,lr=1e-4,weight_decay=.01)
    for step in range(5):
        for a,b in zip(left,right):
            gradient=torch.randn_like(a)
            a.grad,b.grad=gradient,gradient.clone()
        stock.step()
        host.step()
        assert all(torch.equal(a,b) for a,b in zip(left,right))
        assert all(value.device.type=='cpu' for state in host.state.values()
                   for value in state.values() if torch.is_tensor(value))
        if step==2:
            state=copy.deepcopy(host.state_dict())
            host=HostAdamW(right,lr=1e-4,weight_decay=.01)
            host.load_state_dict(state)
    print('PASS: stock AdamW and CPU-moment AdamW match exactly, including resume; device='+device)


if __name__=='__main__':
    main()
