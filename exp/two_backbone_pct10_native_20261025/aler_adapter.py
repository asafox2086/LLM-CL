"""ALER author latent search and reverse-KL repair; full decoder tuning."""
import ast,copy,hashlib,json,sys,types,__future__
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
def author_components():
 folder=ROOT/"code/ALER";files=json.loads((folder/"files_sha256.json").read_text());verified={}
 for rel in ["alerdistill/latent_repair/searcher.py","alerdistill/latent_repair/danger.py","alerdistill/trainers/alerdistill_sft_trainer.py"]:
  p=folder/"source"/rel;expected=next(x["sha256"] for x in files if x["path"]==rel);assert hashlib.sha256(p.read_bytes()).hexdigest()==expected;verified[rel]=expected
 sys.path.insert(0,str(folder/"source"))
 from alerdistill.latent_repair.searcher import AlerDistillConfig,BatchedSoftPromptSearcher
 from alerdistill.latent_repair.danger import kl_divergence
 tree=ast.parse((folder/"source/alerdistill/trainers/alerdistill_sft_trainer.py").read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="AlerDistillSFTTrainer");node=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="_repair_loss_on_prompts");env={"torch":torch,"kl_divergence":kl_divergence}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(folder/"source/alerdistill/trainers/alerdistill_sft_trainer.py"),"exec",flags=__future__.annotations.compiler_flag),env)
 return AlerDistillConfig,BatchedSoftPromptSearcher,env["_repair_loss_on_prompts"],json.loads((folder/"source_manifest.json").read_text())["revision"],verified
class LanguageView(nn.Module):
 def __init__(self,decoder,head):super().__init__();self.decoder=decoder;self.head=head
 def get_input_embeddings(self):return self.decoder.embed_tokens
 def forward(self,inputs_embeds,attention_mask=None,use_cache=False):
  b,n,_=inputs_embeds.shape;positions=torch.arange(n,device=inputs_embeds.device).view(1,n).expand(b,n)
  if self.decoder.config.model_type=="qwen2_vl":positions=positions.unsqueeze(0).expand(3,b,n)
  output=self.decoder(inputs_embeds=inputs_embeds,attention_mask=attention_mask,position_ids=positions,use_cache=False,return_dict=True)
  return SimpleNamespace(logits=self.head(output.last_hidden_state))
class ALERMethod:
 def __init__(self,model,name,instructions,embedding=None):
  self.model=model;self.name=name;self.embedding=None;self.stage=0;self.learning_rate=4e-5;self.host_optimizer=True;self.offload_activations=True
  Config,Searcher,repair,self.original_revision,self.source_hashes=author_components();self.latent_repair_cfg=Config();self._Searcher=Searcher;self._author_repair=repair;self.ref_adapter=None;self.new_adapter=None
  self.ref_model=LanguageView(copy.deepcopy(model.model).half(),copy.deepcopy(model.lm_head).half()).eval()
  for p in self.ref_model.parameters():p.requires_grad=False
  for p in model.parameters():p.requires_grad=False
  model.model.float();model.lm_head.float()
  for p in model.model.parameters():p.requires_grad=True
  for p in model.lm_head.parameters():p.requires_grad=True
  self.student=LanguageView(model.model,model.lm_head);self._embedding_weight_snapshot=None;self.searcher=None;self.cached_prompt=None
 def set_instruction(self,task):pass
 def begin_stage(self,stage):
  if self.stage and stage!=self.stage:
   with torch.no_grad():
    for dst,src in zip(self.ref_model.decoder.parameters(),self.model.model.parameters()):dst.copy_(src.detach().to(dst.dtype))
    for dst,src in zip(self.ref_model.head.parameters(),self.model.lm_head.parameters()):dst.copy_(src.detach().to(dst.dtype))
  self.stage=stage
  self._embedding_weight_snapshot=self.model.get_input_embeddings().weight.detach().to("cpu",dtype=torch.float16).clone()
 def _ensure_ref_model(self,model):pass
 def _require_ref_available(self,model,where):assert self.ref_model is not None
 def _get_tokenizer(self):return self.tokenizer
 @staticmethod
 def _has_adapter_support(model):return False
 def _embed_from_snapshot(self,ids,device,dtype):
  ids=ids.detach().cpu();rows=self._embedding_weight_snapshot.index_select(0,ids.reshape(-1));return rows.view(*ids.shape,rows.shape[-1]).to(device=device,dtype=dtype)
 def prepare_update(self,processor):
  self.tokenizer=processor.tokenizer
  if self.searcher is None:self.searcher=self._Searcher(self.latent_repair_cfg,self.tokenizer,embedding_weight=self._embedding_weight_snapshot)
  else:self.searcher._embedding_weight=self._embedding_weight_snapshot
  with torch.autocast("cuda",dtype=torch.float16):out=self.searcher.search(model=self.student,ref_model=self.ref_model,device=next(self.model.parameters()).device)
  self.cached_prompt=out["best_soft_prompt_embeds"];self.model.train();print(json.dumps({"event":"aler_latent_search","stage":self.stage,"search_steps":10,"candidate_count":32,"prompt_batch_size":8,"stats":out["stats"]}),flush=True)
 def extra_loss(self):return self._author_repair(self,self.student,self.cached_prompt)
 def method_state(self,stage):return {"method":self.name,"stage":stage,"upstream_revision":self.original_revision,"latent_search_steps":10,"candidate_count":32,"repair_reverse_kl":True,"full_decoder_tuning":True,"reference_in_memory_only":True}
