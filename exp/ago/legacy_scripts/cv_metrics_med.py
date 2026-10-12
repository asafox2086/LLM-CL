"""Medical QA generation accuracy plus independent next-token knowledge probes."""
import json
import re
import torch
from common import ROOT, digest_bytes, write_json
from cv_runtime import evaluate as original_evaluate, batch_tensors

def letter_correct(text, reference):
    # Strict free generation: a lone A/B/C/D, optional closing punctuation. Do not
    # remove English articles: the old generic normalizer maps A to empty text.
    match = re.fullmatch(r'\s*([ABCD])\s*[.)]?\s*', text, re.I)
    return bool(match and match.group(1).upper() == reference)

def evaluate_med(model, tokenizer, examples, config, output, metadata, method=None, resume=False):
    scores = original_evaluate(model,tokenizer,examples,config,output,metadata,method,resume)
    if examples[0]['task_id'] in ['medical_medqa','medical_medmcqa']:
        records = [json.loads(x) for x in output.read_text().splitlines()]
        for r,row in zip(records,examples):
            correct = 100*letter_correct(r['prediction'],r['references'][0])
            r['scores'].update(exact_match=correct,token_f1=correct)
            r.update(choices=row['choices'], question=row['question'], scoring='strict single answer letter; optional punctuation')
        tmp = output.with_suffix('.med.tmp')
        tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));tmp.replace(output)
        scores.update(exact_match=sum(r['scores']['exact_match'] for r in records)/len(records),
                      token_f1=sum(r['scores']['token_f1'] for r in records)/len(records))
        write_json(output.with_suffix('.scores.json'),scores)
    return scores

def evaluate_probe(model, method, proc, output, stage):
    manifest = json.loads((ROOT/'data/knowledge_probe/manifest.json').read_text())
    raw = (ROOT/'data/knowledge_probe/probes.jsonl').read_bytes()
    assert digest_bytes(raw) == manifest['selection_sha256']
    spec = {'source_sha256':manifest['selection_sha256'],'role':'heldout only; never trained or used for model selection',
            'scoring':'next-token logits restricted to A/B/C/D; distinct from free generation accuracy',
            'selection':'existing fixed 198-question panel: medical 96, general 102; not full MMLU'}
    write_json(output.parent/'protocol.json',spec)
    if output.exists():
        records=[json.loads(x) for x in output.read_text().splitlines()]
        assert len(records)==198 and all(r['stage']==stage and r['source_sha256']==spec['source_sha256'] for r in records)
    else:
        rows=[];tok=proc.tokenizer
        for r in [json.loads(x) for x in raw.splitlines()]:
            prompt='Answer this multiple-choice question. Respond only with the letter of the correct answer.\n\n'+r['question']+'\n'
            prompt+='\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(r['choices']))
            rendered=proc.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
            ids=tok.encode(rendered,add_special_tokens=False);assert len(ids)<=1536
            rows.append({**r,'prompt':prompt,'rendered_prompt':rendered,'prompt_ids':ids})
        ids=[tok.encode(c,add_special_tokens=False) for c in 'ABCD'];assert all(len(i)==1 for i in ids)
        letters=torch.tensor([i[0] for i in ids],device=model.device)
        records=[];model.eval()
        with torch.inference_mode():
            for start in range(0,len(rows),4):
                chosen=rows[start:start+4];batch=batch_tensors(chosen,tok,model.device,False)
                if hasattr(method,'prepare_batch'):method.prepare_batch(chosen,tok,batch,False)
                pos,_=model.get_rope_index(batch['input_ids'],attention_mask=batch['attention_mask'])
                hidden=model.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],position_ids=pos,
                                   use_cache=False,return_dict=True).last_hidden_state[:,-1]
                logits=model.lm_head(hidden).float()[:,letters]
                for row,pred,probs in zip(chosen,logits.argmax(-1).cpu().tolist(),logits.softmax(-1).cpu().tolist()):
                    records.append({**row,'stage':stage,'prediction':'ABCD'[pred],'reference':'ABCD'[row['answer']],
                                    'correct':int(pred==row['answer']),'choice_probabilities':probs,
                                    'source_sha256':spec['source_sha256']})
        output.parent.mkdir(parents=True,exist_ok=True)
        tmp=output.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));tmp.replace(output)
    result={g:{'count':len(group),'accuracy':100*sum(r['correct'] for r in group)/len(group)}
            for g in ['medical','general'] if (group:=[r for r in records if r['group']==g])}
    write_json(output.with_suffix('.scores.json'),result)
    return result
