"""Describe the origin of late medical F1 gains on paired, unchanged questions."""
import argparse
import json
from pathlib import Path

import numpy as np

from parameter_efficiency_retention_20261118 import METHODS, RUNS, digest


def changes(scores):
    before, immediate, final = scores.T
    early = immediate - before
    late = final - immediate
    upward = np.maximum(late, 0)
    restored = np.minimum(upward, np.maximum(-early, 0))
    return dict(before=before, immediate=immediate, final=final, early=early, late=late,
                later_up=upward, regained_earlier_loss=restored,
                above_both_earlier_levels=upward-restored, later_loss=np.maximum(-late,0))


def main(root, backbone):
    evidence_dir = root / 'summary/parameter_efficiency_retention_20261118'
    previous_path = evidence_dir / (backbone+'.json')
    previous = json.loads(previous_path.read_bytes())
    manifest = root/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert digest(manifest) == previous['manifest_sha256']
    records = json.loads(manifest.read_bytes())['tasks']
    out = {'backbone':backbone,'parent_sha256':digest(previous_path),'source_sha256':digest(Path(__file__)),
           'methods':{},'pre_test_files':{},'units':'F1 points, per-question means; restored and above both sum to gross upward change'}
    for method, name in zip(METHODS,RUNS[backbone]):
        run = root / 'exp/two_backbone_native_cl_20261017_v2/runs' / backbone / 'pct01' / method / name
        all_scores, tasks = [], {}
        for task in records:
            if task['domain'] != 'medical':
                continue
            i = task['task']
            test = root/task['files']['test']['path']
            assert digest(test) == task['files']['test']['sha256']
            rows = [json.loads(s) for s in test.read_bytes().splitlines()]
            cols = []
            for stage in (i-1,i,19):
                p = run/f'evaluations/stage_{stage:02}/T{i:02}_test_predictions.jsonl'
                key = str(p.relative_to(root))
                if stage == i-1:
                    out['pre_test_files'][key]=digest(p)
                else:
                    assert digest(p)==previous['predictions_sha256'][key]
                values = [json.loads(s) for s in p.read_bytes().splitlines()]
                assert len(values)==len(rows)==task['files']['test']['count']
                assert all(v['id']==r['id'] and v['references']==r['answers'] and v['generation_cap']==task['max_new_tokens']
                           for v,r in zip(values,rows))
                score = np.array([v['scores']['token_f1'] for v in values])
                assert np.isfinite(score).all() and score.min()>=0 and score.max()<=100
                assert abs(score.mean()-json.loads(p.with_suffix('.scores.json').read_bytes())['token_f1'])<1e-8
                cols.append(score)
            scores=np.column_stack(cols)
            all_scores.append(scores)
            tasks[str(i)]={k:float(v.mean()) for k,v in changes(scores).items()}
            tasks[str(i)].update(qa=len(rows))
        scores=np.concatenate(all_scores)
        assert scores.shape==(2094,3) and len(tasks)==10
        summary={k:float(v.mean()) for k,v in changes(scores).items()}
        anchor=previous['domains']['medical']
        prefix='small' if method=='SeqFT_LoRA' else 'expert'
        assert abs(summary['immediate']-anchor[prefix+'_immediate'])<1e-8
        assert abs(summary['final']-anchor[prefix+'_end'])<1e-8
        assert abs(summary['regained_earlier_loss']+summary['above_both_earlier_levels']-summary['later_loss']-summary['late'])<1e-8
        summary['regained_share_of_positive_change']=summary['regained_earlier_loss']/summary['later_up']
        out['methods'][method]=dict(summary=summary,tasks=tasks)
    target=evidence_dir/(backbone+'_medical_trajectory.json')
    with target.open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({m:v['summary'] for m,v in out['methods'].items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('backbone',choices=RUNS)
    args=p.parse_args()
    main(args.root,args.backbone)
