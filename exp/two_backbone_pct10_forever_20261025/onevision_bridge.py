"""Expose the native OneVision decoder to unchanged author-method kernels.

Only model/processor interfaces differ. Training, order, optimizer, schedule,
method kernels, evaluation and checkpoint code are shared with Qwen2-VL.
"""
import copy
import torch
from torch import nn
from transformers import LlavaOnevisionForConditionalGeneration

ACTIVE = None
LAST_NATIVE = None
NATIVE_GATE_MODE = False


def bind(model):
    global ACTIVE
    ACTIVE = model


class PackedVision(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.tower = core.vision_tower
        self.projector = core.multi_modal_projector
        self.image_newline = core.image_newline
        object.__setattr__(self, 'core', core)

    def get_dtype(self):
        return next(self.tower.parameters()).dtype

    def forward(self, pixels, grid_thw=None):
        # Native image_sizes is retained under the common transport key;
        # it is never used as a Qwen positional grid by this backbone.
        if pixels.ndim == 2 and pixels.shape[-1] == self.core.config.text_config.hidden_size:
            return pixels
        with torch.autocast(pixels.device.type, dtype=self.get_dtype()):
            features = self.core.get_image_features(pixels.to(self.get_dtype()), grid_thw,
                self.core.config.vision_feature_layer, self.core.config.vision_feature_select_strategy)
            packed, _ = self.core.pack_image_features(features, grid_thw, self.image_newline,
                                                     self.core.config.vision_aspect_ratio)
        return packed


class OneVisionBridge(nn.Module):
    llmcl_backbone = 'onevision'

    def __init__(self, core):
        super().__init__()
        self.model = core.language_model.model
        self.lm_head = core.language_model.lm_head
        self.visual = PackedVision(core)
        self.config = copy.deepcopy(core.config.text_config)
        self.config.image_token_id = core.config.image_token_index
        object.__setattr__(self, 'core', core)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        # Match Qwen2-VL's base/visual dtype, including methods that keep
        # FP32 decoder parameters but run the common FP16 autocast path.
        return self.visual.get_dtype()

    def train(self, mode=True):
        super().train(mode)
        self.core.train(mode)
        return self

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def gradient_checkpointing_enable(self, **kwargs):
        self.core.language_model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        self.core.language_model.enable_input_require_grads()

    def get_rope_index(self, ids, grid=None, attention_mask=None):
        # OneVision uses the Qwen2 language decoder's ordinary 1D positions.
        return torch.arange(ids.shape[1], device=ids.device).unsqueeze(0).expand_as(ids), None

    def generate(self, input_ids, pixel_values=None, image_grid_thw=None,
                 attention_mask=None, **kwargs):
        embeddings = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            features = self.visual(pixel_values, image_grid_thw)
            mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(embeddings)
            assert int(mask.sum()) == features.numel()
            embeddings = embeddings.masked_scatter(mask, features.to(embeddings.dtype))
        return self.core.language_model.generate(input_ids=input_ids, inputs_embeds=embeddings,
                attention_mask=attention_mask, **kwargs)


class Factory:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        return OneVisionBridge(LlavaOnevisionForConditionalGeneration.from_pretrained(*args, **kwargs))


def pack_native_batch(batch,row):
    global LAST_NATIVE
    assert isinstance(ACTIVE, OneVisionBridge)
    batch = {k: v.to(ACTIVE.device) for k,v in batch.items()}
    sizes = batch.pop('image_sizes')
    raw_pixels = batch['pixel_values'] if NATIVE_GATE_MODE else None
    batch['image_grid_thw'] = sizes
    from frozen_vision import prepare
    hit = prepare(batch,row,ACTIVE,ACTIVE._llmcl_visual_original)
    features = batch['pixel_values']
    ids = batch['input_ids']; token = ACTIVE.config.image_token_id
    positions = (ids[0] == token).nonzero().flatten()
    assert len(positions) and int(positions[-1]-positions[0]+1)==len(positions)
    if len(positions)!=len(features):
        begin,end=int(positions[0]),int(positions[-1])+1
        revised=torch.cat([ids[:,:begin],torch.full((1,len(features)),token,device=ids.device,dtype=ids.dtype),ids[:,end:]],dim=1)
        assert torch.equal(ids[ids!=token],revised[revised!=token])
        batch['input_ids']=revised;batch['attention_mask']=torch.ones_like(revised)
    assert int((batch['input_ids']==token).sum())==len(features)
    if NATIVE_GATE_MODE:
        LAST_NATIVE={'input_ids':batch['input_ids'],'attention_mask':batch['attention_mask'],'pixel_values':raw_pixels,'image_sizes':sizes}
    return batch,hit


@torch.no_grad()
def forward_gate(row, task, processor):
    from runtime_utils import input_batch, answer_loss
    model = ACTIVE
    was = model.training
    model.eval()
    try:
        batch, _ = input_batch(row, task, processor, model.device)
        raw = dict(LAST_NATIVE)
        raw_prompt = dict(raw)
        target = processor.tokenizer(row['answer'], add_special_tokens=False)['input_ids'][:512]
        target += [processor.tokenizer.eos_token_id]
        labels = torch.tensor([target], device=model.device)
        ids = torch.cat([batch['input_ids'], labels], dim=1)
        with torch.autocast('cuda', dtype=model.dtype):
            manual, _ = answer_loss(model, batch, row['answer'], processor)
            raw.update(input_ids=ids, attention_mask=torch.ones_like(ids))
            official = model.core(**raw, use_cache=False, num_logits_to_keep=len(target) + 1).logits[:, :-1].float()
            native_loss = torch.nn.functional.cross_entropy(official.reshape(-1, official.shape[-1]), labels.flatten())
        error = abs(float(manual) - float(native_loss))
        assert torch.isfinite(manual) and torch.isfinite(native_loss) and error < .015
        settings = dict(max_new_tokens=4, do_sample=False, use_cache=True,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id)
        with torch.autocast('cuda', dtype=model.dtype):
            cached_generation = model.generate(**batch, **settings)
            official_generation = model.core.generate(**raw_prompt, **settings)
        assert torch.equal(cached_generation, official_generation), 'Native cached generation differs'
        return {'id': row['id'], 'official_NLL_error': error,
                'official_4_token_generation_exact': True,
                'image_tokens': int((batch['input_ids'] == model.config.image_token_id).sum())}
    finally:
        model.train(was)
