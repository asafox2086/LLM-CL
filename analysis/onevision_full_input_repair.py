"""Inference-only full confirmation with actual packed-feature token expansion.

Separate from immutable training sources. No image resize, QA filtering, parameter
update, or answer-dependent routing. Original failed confirmation stays intact.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,json,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
import onevision_capacity_probe as capacity
from run import Experiment,freeze,write
from onevision_research_worker import digest

ROOT=capacity.ROOT
REPORT=ROOT/'summary/onevision_diagnostics_20261016/full_input_repair'
ORIGINAL_REPORT=capacity.REPORT


class RepairedExperiment(capacity.CapacityExperiment):
    def __init__(self,manifest,run):
        super().__init__(manifest,run)
        write(run/'renderer_protocol.json',{
            'renderer_sha256':digest(Path(__file__)),
            'original_training_source_sha256':capacity.source_hashes(),
            'repair':'single contiguous image-token span expanded to actual packed visual feature rows',
            'unchanged':'question, answer, original image pixels, image preprocessing, model weights, router',
            'optimizer_updates':0,'old_partial_predictions_reused':False})
        self.model.eval()
        self.forward_gate()

    def inputs(self,row,view='image',wrong=None,retain_pixels=False):
        batch=super().inputs(row,view,wrong,retain_pixels)
        if '_features' not in batch:return batch
        ids=batch['input_ids'];token=self.model.config.image_token_index
        positions=(ids[0]==token).nonzero().flatten()
        actual=batch['_features'].shape[0]
        if len(positions)!=actual:
            assert ids.shape[0]==1 and len((wrong or row)['images'])==1
            assert len(positions)>0 and int(positions[-1]-positions[0]+1)==len(positions)
            start,end=int(positions[0]),int(positions[-1])+1
            revised=torch.cat([ids[:,:start],torch.full((1,actual),token,device=ids.device,dtype=ids.dtype),ids[:,end:]],dim=1)
            assert torch.equal(ids[ids!=token],revised[revised!=token])
            batch['input_ids']=revised;batch['attention_mask']=torch.ones_like(revised)
            with (self.run/'image_token_repairs.jsonl').open('a') as f:
                f.write(json.dumps({'id':row['id'],'processor_tokens':len(positions),'packed_tokens':actual,
                                    'image_sizes':batch['image_sizes'].tolist()})+'\n')
        assert int((batch['input_ids']==token).sum())==actual
        return batch

    @torch.no_grad()
    def forward_gate(self):
        manifest,data,domains,_,_,_,_=freeze()
        rows=[data[tasks[0],'train'][0] for tasks in domains.values()]+[data[16,'test'][400]]
        assert rows[-1]['id']=='FinMME:5082'
        evidence=[]
        for row in rows:
            raw=Experiment.inputs(self,row,retain_pixels=True)
            corrected=self.inputs(row,retain_pixels=True)
            old_count=int((raw['input_ids']==self.model.config.image_token_index).sum())
            actual=int(raw['_features'].shape[0])
            if old_count==actual:assert torch.equal(raw['input_ids'],corrected['input_ids'])
            assert torch.equal(raw['pixel_values'],corrected['pixel_values'])
            target=self.processor.tokenizer(row['answer'],add_special_tokens=False)['input_ids'][:512]+[self.processor.tokenizer.eos_token_id]
            labels=torch.tensor([target],device='cuda')
            ids=torch.cat([corrected['input_ids'],labels],dim=1)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                manual=self.model.language_model(inputs_embeds=self.embeddings(ids,corrected),attention_mask=torch.ones_like(ids),use_cache=False,num_logits_to_keep=len(target)+1).logits[:,:-1].float()
                native={k:v for k,v in corrected.items() if k!='_features'}
                native.update(input_ids=ids,attention_mask=torch.ones_like(ids))
                official=self.model(**native,use_cache=False,num_logits_to_keep=len(target)+1).logits[:,:-1].float()
                a=F.cross_entropy(manual.reshape(-1,manual.shape[-1]),labels.flatten())
                b=F.cross_entropy(official.reshape(-1,official.shape[-1]),labels.flatten())
            error=abs(float(a)-float(b));assert error<.015 and torch.isfinite(a) and torch.isfinite(b)
            evidence.append({'id':row['id'],'processor_tokens':old_count,'actual_features':actual,
                             'native_forward_nll_error':error,'unchanged_when_compatible':old_count==actual})
        write(self.run/'renderer_gpu_gate.json',{'passed':True,'cases':evidence,'optimizer_updates':0})
        print(json.dumps({'event':'renderer_gate_passed','cases':evidence}),flush=True)


def main(cap):
    REPORT.mkdir(parents=True,exist_ok=True)
    # Immutable references to already completed training; new inference pointers
    # and predictions live in a disjoint report/run namespace.
    original=ORIGINAL_REPORT/f'cap{cap}_complete.json'
    proof=json.loads(original.read_text())
    assert digest(ROOT/proof['run']/'latest.pt')==proof['final_state_sha256']
    write(REPORT/f'cap{cap}_complete.json',proof)
    capacity.REPORT=REPORT
    capacity.CapacityExperiment=RepairedExperiment
    capacity.main(cap,confirm=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cap',type=int,choices=[1,3],required=True)
    args=parser.parse_args()
    try:main(args.cap)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        write(REPORT/f'cap{args.cap}_full_confirm_needs_attention.json',{'traceback':traceback.format_exc(),'renderer_sha256':digest(Path(__file__))})
        raise
