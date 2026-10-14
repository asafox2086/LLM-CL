import sys,json,torch,copy
from pathlib import Path
from types import SimpleNamespace
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r));from aler_adapter import author_components
Config,Searcher,repair,revision,hashes=author_components()
class Tokenizer:
 def __call__(self,*args,**kwargs):return {"input_ids":torch.tensor([[1,2,3]]),"attention_mask":torch.ones(1,3,dtype=torch.long)}
class Tiny(torch.nn.Module):
 def __init__(self):super().__init__();self.embedding=torch.nn.Embedding(64,12);self.out=torch.nn.Linear(12,64)
 def get_input_embeddings(self):return self.embedding
 def forward(self,inputs_embeds,attention_mask=None,use_cache=False):return SimpleNamespace(logits=self.out(inputs_embeds.cumsum(dim=1)))
torch.manual_seed(42);new=Tiny();ref=copy.deepcopy(new)
with torch.no_grad():new.out.weight.add_(torch.randn_like(new.out.weight)*.02)
cfg=Config();search=Searcher(cfg,Tokenizer());result=search.search(model=new,ref_model=ref,device=torch.device("cpu"))
assert result["best_soft_prompt_embeds"].shape==(1,8,12);assert all(p.requires_grad for p in new.parameters());assert torch.isfinite(search.w).all();assert result["stats"]["alerdistill/search_iter"]==9
from alerdistill.latent_repair.danger import kl_divergence
x=torch.randn(2,64);y=torch.randn(2,64);expected=(x.log_softmax(-1).exp()*(x.log_softmax(-1)-y.log_softmax(-1))).sum(-1).mean();assert torch.allclose(kl_divergence(x,y),expected)

print("PASS: verify_aler_core.py; original author kernels verified on CPU")
