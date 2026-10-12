from pathlib import Path
import urllib.request, json, hashlib

root=Path('/data2/liyapeng_grp/program/LLMCL')
dest=root/'summary/omnimed_med_cv/diagnosis_cl_20261010/papers'
dest.mkdir(parents=True,exist_ok=True)
papers={
 'MedQwen_CVPR2026.pdf':[
  'https://openaccess.thecvf.com/content/CVPR2026/papers/Nejatimanzari_Sparse_Spectral_LoRA_Routed_Experts_for_Medical_VLMs_CVPR_2026_paper.pdf',
  'https://arxiv.org/pdf/2604.01310'],
 'RA_LDL_ICLR2026.pdf':[
  'https://proceedings.iclr.cc/paper_files/paper/2026/file/68a3919db3858f548dea769f2dbba611-Paper-Conference.pdf',
  'https://openreview.net/pdf?id=mduCc7XKXH']}
manifest=[]
for name,urls in papers.items():
    errors=[]
    for url in urls:
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
            with urllib.request.urlopen(request,timeout=20) as response: data=response.read()
            assert data.startswith(b'%PDF-')
            (dest/name).write_bytes(data)
            record=dict(file=name,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            manifest.append(record);print(json.dumps(record),flush=True);break
        except Exception as exc: errors.append(str(exc))
    else:
        manifest.append(dict(file=name,status='download_failed',errors=errors));print(name,errors,flush=True)
(dest/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
