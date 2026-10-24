"""A fixed source pairing and exact-length token-body intervention, no model learning."""
import hashlib

HEADER='\nImage text (OCR; may contain recognition errors):\n'
TAIL='\n'

def reserved_ids(tokenizer):
    # Added multimodal placeholders need not be listed by all_special_ids.
    ids=set(tokenizer.all_special_ids)
    for token in ('<image>','<video>'):
        index=tokenizer.convert_tokens_to_ids(token)
        assert isinstance(index,int) and index>=0
        assert tokenizer(token,add_special_tokens=False)['input_ids']==[index],token
        ids.add(index)
    return ids

def pair_sources(rows, group_by_id):
    by_id={r['id']:r for r in rows}
    assert len(by_id)==len(rows)==20
    order=sorted((r['id'] for r in rows if r['ocr_tokens']),key=lambda rid:hashlib.sha256(('lexical-evidence-20261025:'+rid).encode()).digest())
    assert len(order)==18
    donors={}
    for rid,row in by_id.items():
        if not row['ocr_tokens']:
            donors[rid]=None
            continue
        at=order.index(rid)
        donors[rid]=next(other for k in range(1,len(order)) if (other:=order[(at+k)%len(order)])!=rid and by_id[other]['image_id']!=row['image_id'] and group_by_id[other]!=group_by_id[rid])
    return donors

def body_ids(tokenizer, row):
    text='\n'.join(row['ocr_tokens'])
    ids=tokenizer(text,add_special_tokens=False)['input_ids']
    assert not set(ids).intersection(reserved_ids(tokenizer))
    return ids

def intervention(original, correct_body, donor_body, tokenizer):
    """Lists only; preserve the entire original prompt around an added evidence span."""
    end=tokenizer.convert_tokens_to_ids('<|im_end|>')
    assert end in tokenizer.all_special_ids and original.count(end)>=1
    at=max(i for i,x in enumerate(original) if x==end)
    # The saved OneVision processor template joins these two markers directly.
    # It uses a space after the user role, unlike the bare tokenizer template.
    expected=tokenizer('<|im_end|><|im_start|>assistant\n',add_special_tokens=False)['input_ids']
    assert original[at:]==expected
    begin=tokenizer.convert_tokens_to_ids('<|im_start|>')
    assert begin in tokenizer.all_special_ids
    completed_starts=[i for i,x in enumerate(original[:at]) if x==begin]
    assert completed_starts
    assert tokenizer.decode(original[completed_starts[-1]+1:at],skip_special_tokens=False).startswith('user ')
    prefix=tokenizer(HEADER,add_special_tokens=False)['input_ids']
    tail=tokenizer(TAIL,add_special_tokens=False)['input_ids']
    assert not set(prefix+tail+correct_body+donor_body).intersection(reserved_ids(tokenizer))
    n=len(correct_body)
    assert donor_body or n==0
    wrong_body=[donor_body[i%len(donor_body)] for i in range(n)] if n else []
    left=original[:at]+prefix;right=tail+original[at:]
    correct=left+correct_body+right;wrong=left+wrong_body+right
    start=len(left);stop=start+n
    assert len(correct)==len(wrong)
    assert correct[:start]==wrong[:start] and correct[stop:]==wrong[stop:]
    assert correct[:at]+correct[stop+len(tail):]==original
    if n==0:assert correct==wrong
    return dict(correct=correct,wrong=wrong,start=start,stop=stop,insert_at=at,correct_body=correct_body,wrong_body=wrong_body)
