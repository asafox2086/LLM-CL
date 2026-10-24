"""Real local tokenizer, synthetic text only: pair and token-span CPU control."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time


def rejected(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError('Expected invalid special token/template to be rejected')


def main():
    tick = time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=Path)
    args = parser.parse_args()
    root = Path(os.environ['LLMCL_ROOT'])
    path = args.tokenizer or root / '.deployment/paired_image_support_20261024_view/model/LLaVA-OneVision-Qwen2-0.5B-OV'
    from transformers import AutoProcessor
    import evidence

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = processor.tokenizer
    # No real IDs, images, questions, answers, model, or scoring enter this control.
    rows = [dict(id=f'synthetic:{i:02}', image_id=f'image:{i}',
                 ocr_tokens=[f'Label {i}', '中文 2026', 'A/B'] if i < 18 else [])
            for i in range(20)]
    rows[1]['image_id'] = rows[0]['image_id']
    groups = {row['id']: f'group:{i}' for i, row in enumerate(rows)}
    groups[rows[3]['id']] = groups[rows[2]['id']]
    donors = evidence.pair_sources(rows, groups)
    assert donors == evidence.pair_sources(list(reversed(rows)), groups)
    by_id = {row['id']: row for row in rows}
    order = sorted((row['id'] for row in rows if row['ocr_tokens']),
                   key=lambda rid: hashlib.sha256(('lexical-evidence-20261025:' + rid).encode()).digest())
    for rid, donor in donors.items():
        if not by_id[rid]['ocr_tokens']:
            assert donor is None
            continue
        at = order.index(rid)
        eligible = [order[(at + k) % len(order)] for k in range(1, len(order))
                    if by_id[order[(at + k) % len(order)]]['image_id'] != by_id[rid]['image_id']
                    and groups[order[(at + k) % len(order)]] != groups[rid]]
        assert donor == eligible[0] and donor != rid

    prompt = processor.apply_chat_template([
        dict(role='user', content=[dict(type='image'),
                                  dict(type='text', text='Inspect the synthetic label.')])
    ], tokenize=False, add_generation_prompt=True)
    original = tokenizer(prompt, add_special_tokens=False)['input_ids']
    assert isinstance(original, list)
    snapshot = list(original)
    image_id = tokenizer.convert_tokens_to_ids(processor.image_token)
    assert tokenizer(processor.image_token, add_special_tokens=False)['input_ids'] == [image_id]
    assert image_id in evidence.reserved_ids(tokenizer) and original.count(image_id) == 1
    header = tokenizer(evidence.HEADER, add_special_tokens=False)['input_ids']
    tail = tokenizer(evidence.TAIL, add_special_tokens=False)['input_ids']
    # Unequal lengths exercise both truncation and cycling; the empty pair keeps
    # the same header and suffix while inserting no OCR body tokens.
    cases = [(['A'], ['Donor text with several tokens 2035']),
             (['Long synthetic OCR line 123456789', '中文条目', 'second line'], ['Z']),
             ([], [])]
    lengths = []
    for correct_words, donor_words in cases:
        body = evidence.body_ids(tokenizer, dict(ocr_tokens=correct_words))
        donor = evidence.body_ids(tokenizer, dict(ocr_tokens=donor_words))
        assert body == tokenizer('\n'.join(correct_words), add_special_tokens=False)['input_ids']
        result = evidence.intervention(original, body, donor, tokenizer)
        start, stop, at = result['start'], result['stop'], result['insert_at']
        a, b = result['correct'], result['wrong']
        assert original == snapshot and start == at + len(header)
        assert a[:at] == original[:at] and a[at:start] == header
        assert a[start:stop] == body and a[stop:stop + len(tail)] == tail
        assert a[stop + len(tail):] == original[at:]
        assert a[:start] == b[:start] and a[stop:] == b[stop:]
        assert len(a) == len(b) == len(original) + len(header) + len(body) + len(tail)
        assert b[start:stop] == ([donor[i % len(donor)] for i in range(len(body))] if body else [])
        assert a.count(image_id) == b.count(image_id) == original.count(image_id)
        if not body:
            assert a == b and start == stop
        lengths.append(dict(correct=len(body), donor=len(donor)))
    assert lengths[0]['correct'] < lengths[0]['donor']
    assert lengths[1]['correct'] > lengths[1]['donor']
    special = tokenizer.convert_tokens_to_ids('<|im_end|>')
    rejected(lambda: evidence.body_ids(tokenizer, dict(ocr_tokens=['<|im_end|>'])))
    rejected(lambda: evidence.body_ids(tokenizer, dict(ocr_tokens=[processor.image_token])))
    rejected(lambda: evidence.body_ids(tokenizer, dict(ocr_tokens=[processor.video_token])))
    rejected(lambda: evidence.intervention(original, [special], [special], tokenizer))
    rejected(lambda: evidence.intervention(original, [image_id], [image_id], tokenizer))
    rejected(lambda: evidence.intervention(original + [42], [], [], tokenizer))
    torch_module = sys.modules.get('torch')
    assert torch_module is None or not torch_module.cuda.is_initialized()
    print(json.dumps(dict(complete=True, passed=True, header_ids=header, tail_ids=tail,
                          seconds=time.monotonic() - tick,
                          synthetic_pair_rows=20, token_pairs=3, body_lengths=lengths,
                          image_placeholder_id=image_id,
                          image_in_all_special_ids=image_id in tokenizer.all_special_ids,
                          CUDA_initialized=False, models=0, real_QA=0), allow_nan=False))


if __name__ == '__main__':
    main()
