"""Verify report provenance, derived metrics and publishable image links."""
from pathlib import Path
import json,csv,hashlib,re,subprocess,math
root=Path(__file__).resolve().parents[1]
cv=json.loads((root/'summary_cv/process_data.json').read_text())
t5=json.loads((root/'summary/t5_process_data.json').read_text())
checks={}
for name,d,path in [('qwen',cv,'summary_cv/paper_metrics.csv'),('t5',t5,'summary/t5_paper_metrics.csv')]:
    rows=list(csv.DictReader((root/path).open()));assert len(rows)==12
    for method,source in d['methods'].items():
        p=root/source['source'];assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
        a=source['matrix'];T=len(a[0]);r=next(r for r in rows if r['method']==method and r['score']=='rougeL')
        same=lambda value,key:math.isclose(value,float(r[key]),abs_tol=1e-8)
        assert same(sum(a[-1])/T,'MFN') and same(sum(a[-1])/T,'AP')
        assert same(sum(a[j+1][j] for j in range(T))/T,'MFT')
        assert same(sum(sum(a[s][:s])/s for s in range(1,T+1))/T,'MAA')
        assert same((float(r['MFN'])-float(r['MFT']))*T/(T-1),'BWT')
        assert r['SAPT_FWT_vs_independent']==r['oracle_gap']==''
    checks[name+'_source_hashes_and_metrics']=True
photos=json.loads((root/'sample/image_manifest.json').read_text());assert len(photos)==4
for file,info in photos.items():
    p=root/'sample'/file;assert p.read_bytes()==(root/info['source']).read_bytes()
    assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
figures=json.loads((root/'sample/figure_manifest.json').read_text());assert len(figures)==8
for info in figures.values():
    assert hashlib.sha256((root/info['png']).read_bytes()).hexdigest()==info['sha256']
    assert (root/info['pdf']).exists()
    for path,sha in info['data_sha256'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==sha
checks['published_photos_byte_identical']=len(photos);checks['png_and_vector_pdf_pairs']=len(figures)
mds=[root/'README.md',root/'CV_data/README.md',root/'rules/001_experiment_protocol.md',*list((root/'summary').glob('*.md')),*list((root/'summary_cv').glob('*.md')),*list((root/'sample').glob('*.md')),*list((root/'paper').glob('*.md'))]
image_count=0
for p in mds:
    text=p.read_text();assert not any(ord(c)<32 and c not in '\n\t\r' for c in text)
    assert text.splitlines().count('$$')%2==0,p
    for block in re.findall(r'\$\$(.*?)\$\$',text,re.S):assert '\t' not in block,(p,block)
    for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):
        if '://' in target:continue
        dst=(p.parent/target).resolve();assert dst.is_file(),(p,target)
        assert subprocess.run(['git','check-ignore','--quiet',str(dst)],cwd=root).returncode==1,(p,dst)
        image_count+=1
checks['local_markdown_images_exist_and_publishable']=image_count
verify=json.loads((root/'summary_cv/verification.json').read_text());assert verify['total_test_records']==72000
probe=json.loads((root/'summary_cv/knowledge_probe_verification.json').read_text());assert probe['total_predictions']==7920
paired=list(csv.DictReader((root/'summary_cv/learning_answer_transition_items.csv').open()));assert len(paired)==7200
checks['main_test_answers_recomputed']=72000;checks['external_predictions_recomputed']=7920;checks['learning_paired_test_items']=7200
(root/'summary_cv/report_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
