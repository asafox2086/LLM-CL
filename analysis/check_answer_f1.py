"""Verify primary metric reproduces immutable stored scores without new normalization."""
import hashlib,json
from pathlib import Path
from answer_f1 import answer_f1,VERSION
ROOT=Path(__file__).resolve().parents[1]

def main():
    cases=[('same','CT',['CT'],100.),('case/punctuation','CT.',['ct'],100.),
           ('Chinese','肝脏',['肝脏'],100.),('order','red blue',['blue red'],100.),
           ('partial','red',['red blue'],200/3),('empty','',['CT'],0.),
           ('opposite','yes',['no'],0.),('multiple references','wine',['champagne','wine'],100.),
           ('NO option mapping','23.5',['A'],0.),('NO wrapper stripping','The answer is CT.',['CT'],40.),
           ('NO semantic judge','car',['automobile'],0.)]
    for name,p,g,value in cases:assert abs(answer_f1(p,g)-value)<1e-10,(name,answer_f1(p,g),value)
    audit=json.loads((ROOT/'summary/onevision_diagnostics_20261016/ACQUISITION_COLLAPSE_AUDIT.json').read_text())
    checked=0;files=0
    for name,digest in audit['sources'].items():
        if not name.endswith('_image.jsonl'):continue
        path=ROOT/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;files+=1
        for row in map(json.loads,path.read_text().splitlines()):
            if 'prediction' not in row:continue
            assert answer_f1(row['prediction'],row['references'])==row['scores']['token_f1']
            checked+=1
    result={'passed':True,'metric_version':VERSION,'boundary_cases':len(cases),'actual_prediction_files':files,
            'actual_generated_rows_exactly_equal_existing_token_f1':checked,'option_mapping':False,
            'new_normalization_or_model_inference':False,'metric_sha256':hashlib.sha256((ROOT/'analysis/answer_f1.py').read_bytes()).hexdigest()}
    dest=ROOT/'summary/primary_metric_20261017';dest.mkdir(exist_ok=True,parents=True)
    (dest/'CPU_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
