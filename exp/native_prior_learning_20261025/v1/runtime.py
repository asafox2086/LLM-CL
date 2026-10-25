"""Fixed native SmolVLM input/answer semantics; no model work on import.

The input compatibility changes preserve the official2048 resize/512 tile/64
visual-token contract, without changing the pixel algorithm. Answer-body tokens
are truncated to512 before the native EOS rather than rejecting longer bodies.
FP16 is this new condition, not a BF16 comparison or a fallback search.
"""
import hashlib
import json
import math
import os
from pathlib import Path
from types import FunctionType, MethodType

ATTENTION = {'vision_config': 'eager', 'text_config': 'sdpa'}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def configure_processor(processor, model_config, processor_config):
    """Change one instance's helper binding, never the installed module/class.

    The unchanged original resize bytecode calls the original size calculator
    with max_image_size=2048. Its PIL conversion/resampling and the512 tile path
    are inherited. The function's globals are a private shallow copy.
    """
    config = model_config.to_dict() if hasattr(model_config, 'to_dict') else model_config
    vision = config['vision_config']
    assert (vision['image_size'], vision['patch_size'], config['scale_factor']) == (512, 16, 4)
    expected_seq_len = ((vision['image_size'] // vision['patch_size']) ** 2) // config['scale_factor'] ** 2
    assert processor_config['image_seq_len'] == expected_seq_len == 64
    before_seq_len = processor.image_seq_len
    # Some old ProcessorMixin loading paths leave the constructor default169.
    # Restore the bound official64-token-per-tile contract, without a token search.
    assert before_seq_len in (64, 169)
    processor.image_seq_len = expected_seq_len
    assert processor.tokenizer.convert_tokens_to_ids(str(processor.image_token)) == config['image_token_id']
    image_processor = processor.image_processor
    assert image_processor.size == {'longest_edge': 2048}
    assert image_processor.max_image_size == {'longest_edge': 512}
    assert image_processor.do_image_splitting is True
    assert not hasattr(image_processor, '_llmcl_resize_compatibility')
    original = image_processor.resize.__func__
    assert original.__module__ == 'transformers.models.idefics3.image_processing_idefics3'
    original_helper = original.__globals__['get_resize_output_image_size']

    def bounded_size(image, resolution_max_side, max_image_size=2048, input_data_format=None):
        assert max_image_size == 2048 and 0 < resolution_max_side <= 2048
        return original_helper(image, resolution_max_side=resolution_max_side,
                               max_image_size=2048, input_data_format=input_data_format)

    private_globals = dict(original.__globals__)
    private_globals['get_resize_output_image_size'] = bounded_size
    repaired = FunctionType(original.__code__, private_globals, original.__name__,
                            original.__defaults__, original.__closure__)
    repaired.__kwdefaults__ = original.__kwdefaults__
    repaired.__module__ = original.__module__
    repaired.__doc__ = original.__doc__
    image_processor.resize = MethodType(repaired, image_processor)
    assert original.__globals__['get_resize_output_image_size'] is original_helper
    assert image_processor.resize.__func__.__code__ is original.__code__
    receipt = dict(scope='this processor instance resize function globals only',
        installed_source_sha256=_sha(original.__code__.co_filename),
        original_resize_bytecode_preserved=True, installed_module_and_class_unmodified=True,
        original_resolution_longest_edge=2048, helper_max_image_size=2048,
        original_tile_longest_edge=512, original_pixel_resize_algorithm=True,
        processor_image_seq_len_before=before_seq_len, official_image_seq_len=64,
        processor_image_seq_len_after=processor.image_seq_len,
        model_geometry=dict(image_size=512, patch_size=16, scale_factor=4),
        image_seq_len_restore_is_bound_asset_semantics=True,
        no_resolution_or_backend_search=True)
    image_processor._llmcl_resize_compatibility = receipt
    return receipt


def policy(processor, generation_config):
    """Resolve the same native assistant terminator as the original runtime."""
    stops = generation_config.eos_token_id
    stops = [stops] if isinstance(stops, int) else list(stops)
    assert stops and all(isinstance(i, int) and not isinstance(i, bool) for i in stops)
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'LLMCL synthetic interface probe.'}]}]
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    marker = 'LLMCL_SYNTHETIC_ASSISTANT_BOUNDARY'
    full = processor.apply_chat_template(messages + [{'role': 'assistant', 'content': [
        {'type': 'text', 'text': marker}]}], tokenize=False, add_generation_prompt=False)
    assert full.startswith(prefix) and full.count(marker) == 1
    suffix = full.split(marker, 1)[1]
    matches = [(i, processor.tokenizer.convert_ids_to_tokens(i)) for i in stops
               if processor.tokenizer.convert_ids_to_tokens(i) in suffix]
    assert len(matches) == 1, 'Native assistant terminator is not uniquely resolved'
    stop_id, stop_text = matches[0]
    assert processor.tokenizer(stop_text, add_special_tokens=False)['input_ids'] == [stop_id]
    return dict(generation_stop_ids=stops, assistant_stop_id=stop_id,
        assistant_stop_text=stop_text, tokenizer_eos_id=processor.tokenizer.eos_token_id,
        pad_id=processor.tokenizer.pad_token_id,
        native_template_SHA256=hashlib.sha256(json.dumps(dict(
            processor=getattr(processor, 'chat_template', None),
            tokenizer=getattr(processor.tokenizer, 'chat_template', None)),
            ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        different_text_boundary_and_generation_stop_allowed=True, not_runtime_model_proof=True)


def tensor_hash(batch):
    """CPU copies are digest I/O, not CPU model execution."""
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(batch.items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        value = tensor.detach().cpu().contiguous()
        h.update(str((name, tuple(value.shape), str(value.dtype))).encode())
        array = value.reshape(-1).view(torch.uint8).numpy().reshape(-1)
        for start in range(0, array.size, 8 * 1024 ** 2):
            h.update(array[start:start + 8 * 1024 ** 2].tobytes())
    return h.hexdigest()


class NativeRuntime:
    def __init__(self, processor, model, root, policy_value, context_limit=8192,
                 input_dtype=None, image_SHA256=None, asset_root=None):
        import torch
        self.processor, self.model, self.root, self.policy = processor, model, Path(root), policy_value
        self.asset_root = Path(asset_root or os.environ.get('LLMCL_ASSET', Path(__file__).resolve().parents[3]))
        self.context_limit = context_limit
        self.input_dtype = torch.float16 if input_dtype is None else input_dtype
        assert self.input_dtype == torch.float16 and context_limit == 8192
        assert next(model.parameters()).device.type == 'cuda'
        assert hasattr(processor.image_processor, '_llmcl_resize_compatibility')
        assert model.model.image_seq_len == processor.image_seq_len == 64
        self.input_records, self.requests = [], []
        self.phase = 'UNASSIGNED'
        self.image_SHA256 = image_SHA256
        self.last_generation_evidence = None
        self.last_target_evidence = None

    def input_batch(self, row, task, processor, device):
        import torch
        from PIL import Image
        from retrieval import retrieve
        assert processor is self.processor
        device = torch.device(device)
        assert device.type == 'cuda' and device == next(self.model.parameters()).device
        context, retrieval = row.get('context', ''), []
        if len(context) > 6000:
            chunks = [context[start:start + 1000] for start in range(0, len(context), 800)]
            retrieval = sorted(retrieve(row['question'], chunks, 5))
            context = '\n'.join(chunks[index] for index in retrieval)
        content, images, shapes, hashes = [], [], [], []
        for name in row.get('images', []):
            path = self.asset_root / name
            image_sha = _sha(path)
            assert self.image_SHA256 is not None and self.image_SHA256[name] == image_sha
            hashes.append(dict(path=name, sha256=image_sha))
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                before = rgb.size
                if rgb.width * rgb.height > 401408:
                    ratio = math.sqrt(401408 / (rgb.width * rgb.height))
                    rgb = rgb.resize((max(1, int(rgb.width * ratio)), max(1, int(rgb.height * ratio))),
                                     Image.Resampling.BICUBIC)
                assert rgb.width * rgb.height <= 401408
                images.append(rgb)
                shapes.append(dict(original=list(before), common_input=list(rgb.size)))
            content.append({'type': 'image'})
        question = row['question']
        if '<image>' in question:
            assert question.count('<image>') == len(images)
            question = question.replace('<image>', '')
        text = task['instruction'] + ('\nContext:\n' + context if context else '') + '\nQuestion: ' + question
        messages = [{'role': 'user', 'content': content + [{'type': 'text', 'text': text}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = dict(processor(text=[prompt], images=images or None, return_tensors='pt', truncation=False))
        for key, value in batch.items():
            assert isinstance(value, torch.Tensor), key
            if value.is_floating_point():
                batch[key] = value.to(dtype=self.input_dtype)
        assert batch['input_ids'].shape[0] == 1
        assert batch['input_ids'].shape[1] < self.context_limit, 'Input would exceed native context'
        prepared = tensor_hash(batch)
        raw_contract = dict(images=hashes, instruction=task['instruction'], question=row['question'],
            original_context=row.get('context', ''), resolved_context=context,
            retrieved_chunk_indexes=retrieval, image_shapes=shapes, common_image_area_cap=401408)
        raw_sha = hashlib.sha256(json.dumps(raw_contract, ensure_ascii=False,
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        details = dict(prompt_tokens=int(batch['input_ids'].shape[1]),
            prepared_input_SHA256=prepared,
            raw_input_sha256=raw_sha, image_SHA256=hashes, image_shapes=shapes,
            retrieved_chunk_indexes=retrieval, common_image_area_cap=401408,
            visual_tokens=int((batch['input_ids'] == self.model.config.image_token_id).sum()),
            input_shape=list(batch['input_ids'].shape), native_processor_geometry=True)
        assert self.phase in ('TRAIN', 'DEV', 'RESOURCE')
        self.input_records.append(dict(id=row['id'], task=task['task'], phase=self.phase, **details))
        batch = {key: value.to(device) for key, value in batch.items()}
        batch['_messages'], batch['_prompt'] = messages, prompt
        return batch, details

    def target(self, batch, answer):
        processor, stop = self.processor, self.policy['assistant_stop_text']
        assert stop not in answer, 'Answer contains a native control token'
        full = processor.apply_chat_template(batch['_messages'] + [{'role': 'assistant', 'content': [
            {'type': 'text', 'text': answer}]}], tokenize=False, add_generation_prompt=False)
        prompt = batch['_prompt']
        assert full.startswith(prompt)
        suffix = full[len(prompt):]
        assert suffix.count(stop) == 1
        suffix = suffix[:suffix.index(stop) + len(stop)]
        a = processor.tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
        b = processor.tokenizer(prompt + suffix, add_special_tokens=False, truncation=False)['input_ids']
        assert b[:len(a)] == a, 'Native prompt/assistant token boundary is not a prefix'
        native = b[len(a):]
        assert native and native[-1] == self.policy['assistant_stop_id']
        # The original answer policy truncates the body, then appends EOS.
        target = native[:-1][:512] + [self.policy['assistant_stop_id']]
        assert batch['input_ids'].shape[1] + len(target) <= self.context_limit, 'Full answer exceeds native context'
        self.last_target_evidence = dict(body_tokens_before_truncation=len(native) - 1,
            body_tokens_after_truncation=len(target) - 1, target_tokens=len(target),
            target_body_limit=512, native_assistant_EOS_appended=True,
            target_sha256=hashlib.sha256(json.dumps(target, separators=(',', ':')).encode()).hexdigest())
        return target

    def answer_loss(self, model, batch, answer, processor):
        import torch
        assert model is self.model and processor is self.processor
        assert batch['input_ids'].is_cuda
        target = self.target(batch, answer)
        labels = torch.tensor([target], device=batch['input_ids'].device, dtype=torch.long)
        length = batch['input_ids'].shape[1]
        ids = torch.cat([batch['input_ids'], labels], dim=1)
        payload = {key: value for key, value in batch.items()
                   if not key.startswith('_') and key not in ('input_ids', 'attention_mask')}
        output = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             use_cache=False, return_dict=True, **payload)
        hidden = output.last_hidden_state[:, length - 1:-1, :]
        logits = model.lm_head(hidden).float()
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
        self.requests.append(dict(kind='answer_NLL', **self.last_target_evidence))
        return loss, len(target)

    def generate(self, batch, cap, fixed=False):
        import torch
        assert batch['input_ids'].is_cuda and 0 < cap
        assert batch['input_ids'].shape[1] + cap <= self.context_limit, 'Generation cap exceeds native context'
        assert not fixed or cap == 32
        kwargs = {key: value for key, value in batch.items() if not key.startswith('_')}

        class FiniteScores:
            def __call__(self, input_ids, scores):
                assert not torch.isnan(scores).any() and not torch.isposinf(scores).any()
                assert torch.isfinite(scores).any(dim=-1).all()
                return scores

        options = dict(do_sample=False, num_beams=1, max_new_tokens=cap, use_cache=True,
            eos_token_id=self.policy['generation_stop_ids'], pad_token_id=self.policy['pad_id'],
            logits_processor=[FiniteScores()])
        if fixed:
            options['min_new_tokens'] = cap
        with torch.cuda.device(batch['input_ids'].device), torch.inference_mode(), torch.autocast('cuda', dtype=self.input_dtype):
            output = self.model.generate(**kwargs, **options)
        response = output[0, batch['input_ids'].shape[1]:].tolist()
        assert 0 < len(response) <= cap and (not fixed or len(response) == 32)
        terminal = 'eos' if response[-1] in self.policy['generation_stop_ids'] else 'cap'
        assert terminal == 'eos' or len(response) == cap
        self.last_generation_evidence = dict(output_tokens=response,
            final_token_id=response[-1], terminal_reason=terminal, fixed_new_tokens=fixed)
        self.requests.append(dict(kind='generation', output_tokens=len(response), cap=cap, fixed=fixed))
        return self.processor.tokenizer.decode(response, skip_special_tokens=True), len(response)
