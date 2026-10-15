"""Exploratory answer-format flags, never official execution accuracy."""
import json
import re
from pathlib import Path

OUT=Path('/data1/liyapeng/program/LLMCL/summary/ten_task_cl_20261014_geo_fin/failure_diagnosis_20261015')
NUMBER=r'[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?'


def number(s):
    values=list(re.finditer(NUMBER,s))
    if len(values)!=1:return None
    match=values[0];value=float(match.group().replace(',',''));tail=s[match.end():].strip().lower()
    if re.match(r'(billion|bn)\b',tail):value*=1e9
    elif re.match(r'(million|mn)\b',tail):value*=1e6
    elif re.match(r'thousand\b',tail):value*=1e3
    return value,('%' in tail or 'percent' in tail),match.group()


def main():
    summary={};details=[]
    for x in map(json.loads,(OUT/'probe_predictions.jsonl').open()):
        if x['task']!=7:continue
        target=number(x['references'][0])
        answer=re.split(r'(?i)(?:final\s+answer|answer)\s*:',x['prediction'])[-1].strip()
        parsed=number(answer);matched=False
        if target and parsed:
            value,percent,literal=target
            decimals=len(literal.split('.')[-1]) if '.' in literal else 0
            tolerance=.5*10**(-decimals)
            if percent:value/=100;tolerance/=100
            pv,pp,_=parsed
            candidates=[pv/100] if pp else ([pv,pv/100] if percent else [pv])
            matched=any(abs(z-value)<=tolerance+1e-9 for z in candidates)
        row=summary.setdefault(x['arm'],{'n':0,'parseable':0,'numeric_equivalence_candidates':0})
        row['n']+=1;row['parseable']+=parsed is not None;row['numeric_equivalence_candidates']+=matched
        details.append({'id':x['id'],'arm':x['arm'],'prediction':x['prediction'],'reference':x['references'][0],
                        'match_candidate':matched,'parseable':parsed is not None})
    result={'rule':'Last Answer: field, otherwise entire prediction; exactly one signed numeric literal; normalize comma and percent/million units; reference rounding half-unit tolerance; percent-reference raw ratio/percent ambiguity accepted. Exploratory optimistic numeric-equivalence flag, NOT FinQA official execution accuracy or replacement score.',
            'summary':summary,'details':details}
    (OUT/'numeric_format_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
