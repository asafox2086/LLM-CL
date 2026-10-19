"""Native Qwen visual/text inputs for author attribution and FV auxiliary passes."""
import json,hashlib,ast,__future__
from collections import OrderedDict
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]

def author_nodes(folder,relative,names,namespace,transform=None):
    base=ROOT/'code'/folder;source=base/'source'/relative
    hashes=json.loads((base/'files_sha256.json').read_text())
    expected=next(row['sha256'] for row in hashes if row['path']==relative)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    text=source.read_text(encoding='utf-8')
    if transform:text=transform(text)
    tree=ast.parse(text);nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==len(names),(relative,names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec',flags=__future__.annotations.compiler_flag),namespace)
    return expected

class NativeIO:
    def __init__(self,method,processor,manifest,data,run):
        self.method,self.model,self.processor,self.manifest,self.data,self.run=method,method.model,processor,manifest,data,Path(run)
        self.cache=OrderedDict()
    @property
    def device(self):return self.model.device
    def stage_rows(self,stage):
        rows=self.data[(stage,'train')]
        if self.method.name=='PaRSP':
            return sorted(rows,key=lambda x:hashlib.sha256(('42:calibration:'+x['id']).encode()).digest())[:256]
        return rows
    def clear(self):self.cache.clear()
    @torch.no_grad()
    def batch(self,row,stage):
        if row['id'] not in self.cache:
            from run import input_batch
            batch,_=input_batch(row,self.manifest['tasks'][stage-1],self.processor,self.device)
            if 'pixel_values' in batch:
                from frozen_vision import prepare
                if hasattr(self.method,'vision_original'):prepare(batch,row,self.model,self.method.vision_original)
                visual=self.model.visual(batch['pixel_values'].to(self.model.visual.get_dtype()),grid_thw=batch['image_grid_thw'])
                batch['_encoded_visual']=visual.detach();del batch['pixel_values']
            self.cache[row['id']]={k:v.detach().cpu() for k,v in batch.items()}
            while len(self.cache)>32:self.cache.popitem(last=False)
        return {k:v.to(self.device) for k,v in self.cache[row['id']].items()}
    def target(self,answer):
        ids=self.processor.tokenizer(answer,add_special_tokens=False,truncation=True,max_length=512)['input_ids']
        return torch.tensor([ids+[self.processor.tokenizer.eos_token_id]],device=self.device)
    def embeddings(self,batch,ids=None):
        ids=batch['input_ids'] if ids is None else ids
        emb=self.model.get_input_embeddings()(ids)
        visual=batch.get('_encoded_visual')
        if visual is not None:
            mask=(ids==self.model.config.image_token_id).unsqueeze(-1).expand_as(emb)
            assert mask.sum().item()==visual.numel()
            emb=emb.masked_scatter(mask,visual.to(emb.dtype))
        positions,_=self.model.get_rope_index(ids,batch.get('image_grid_thw'),attention_mask=torch.ones_like(ids))
        return emb,positions
    def forward(self,batch,ids=None):
        ids=batch['input_ids'] if ids is None else ids
        emb,pos=self.embeddings(batch,ids)
        return self.model.model(inputs_embeds=emb,position_ids=pos,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
    def icl(self,query,demos,stage,corrupt=False):
        blocks=[];grids=[];visual=[]
        answers=[x['answer'] for x in demos]
        if corrupt:answers=answers[1:]+answers[:1]
        for i,row in enumerate(list(demos)+[query]):
            b=self.batch(row,stage);blocks.append(b['input_ids'])
            if '_encoded_visual' in b:visual.append(b['_encoded_visual']);grids.append(b['image_grid_thw'])
            if i<len(demos):blocks.append(self.target(answers[i]))
        result={'input_ids':torch.cat(blocks,dim=1)}
        assert result['input_ids'].shape[1]<=32768,'Do not truncate visual placeholder sequences'
        if visual:result.update(_encoded_visual=torch.cat(visual,dim=0),image_grid_thw=torch.cat(grids,dim=0))
        return result
    def record(self,name,value):
        folder=self.run/'auxiliary';folder.mkdir(parents=True,exist_ok=True)
        p=folder/name;temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)
