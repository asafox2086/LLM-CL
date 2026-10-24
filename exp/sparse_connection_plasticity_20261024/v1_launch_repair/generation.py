"""One complete unscored 100-DEV generation; no resume, ranking or selection."""
import os
import json
import time
import torch
from common import sha


def generate(method, processor, manifest, dev, run, checked_input):
    model = method.model
    model.eval()
    files = {}; total = 0
    eos = processor.tokenizer.eos_token_id
    assert isinstance(eos, int)
    with torch.inference_mode():
        for task in manifest['tasks']:
            tid = task['task']; rows = dev[tid]
            if not rows:
                continue
            path = run / 'evaluations/stage_19' / f'T{tid:02d}_dev_predictions.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.jsonl.partial')
            assert not path.exists() and not temporary.exists()
            with temporary.open('x', encoding='utf8') as stream:
                for row in rows:
                    batch, details = checked_input(row, task, processor, model.device)
                    started = time.monotonic()
                    with torch.autocast('cuda', dtype=torch.float16):
                        output = model.generate(**batch, do_sample=False, num_beams=1,
                            max_new_tokens=task['max_new_tokens'], use_cache=True,
                            pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=eos)
                    response = output[0, batch['input_ids'].shape[1]:]
                    ids = response.detach().cpu().tolist()
                    assert 0 < len(ids) <= task['max_new_tokens']
                    assert ids[-1] == eos or len(ids) == task['max_new_tokens']
                    record = dict(id=row['id'], prediction=processor.tokenizer.decode(response, skip_special_tokens=True),
                        references=row['answers'], output_token_IDs=ids, output_tokens=len(ids),
                        eos_token_id=eos, stop='eos' if ids[-1] == eos else 'length',
                        generation_cap=task['max_new_tokens'], generation_seconds=time.monotonic() - started,
                        **details)
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stream.flush(); total += 1
                os.fsync(stream.fileno())
            temporary.replace(path)
            files[path.relative_to(run).as_posix()] = sha(path)
    assert total == 100
    return dict(count=total, file_sha256=files, scores_not_computed=True)
