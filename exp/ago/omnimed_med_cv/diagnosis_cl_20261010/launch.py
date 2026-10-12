import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
exp=Path(__file__).resolve().parent;root=exp.parents[2]
p=argparse.ArgumentParser();p.add_argument('--group',choices=['standard','medical'],required=True);args=p.parse_args()
selection_path=root/'analysis/omnimed_20261009/selection.json'
selection=json.loads(selection_path.read_text())
split_hashes={s:set() for s in ['train','dev','test']};counts=[]
for probe in selection['probes']:
    counts.append([len(probe['splits'][s]) for s in split_hashes])
    for split in split_hashes:
        for row in probe['splits'][split]:
            split_hashes[split].add(row['image_sha256'])
            assert (root/'analysis/omnimed_20261009/features128'/(row['image_sha256']+'.pt')).exists()
assert counts==[[137,46,48],[129,45,46],[106,36,37]],counts
assert not(split_hashes['train'] & split_hashes['dev'] or split_hashes['train'] & split_hashes['test'] or split_hashes['dev'] & split_hashes['test'])
audit=dict(counts=counts,selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),image_overlap=0,
           code_hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in exp.glob('*.py')},protocol_sha256=hashlib.sha256((exp/'protocol.json').read_bytes()).hexdigest())
(exp/f'launch_audit_{args.group}.json').write_text(json.dumps(audit,indent=2))
groups={'standard':[(0,['seq_lora','ewc_lora','kd_lora']),(1,['migu_lora','olora']),(2,['sapt_lora'])],
        'medical':[(3,['medqwen','moe_lora','seq_capacity']),("cpu",['ra_ldl'])]}
records=[];(exp/'logs').mkdir(exist_ok=True)
for gpu,methods in groups[args.group]:
    for script in (['raldl_run.py'] if methods==['ra_ldl'] else ['run.py']):assert (exp/script).exists()
    pidfile=exp/f'worker_gpu{gpu}.pid'
    if pidfile.exists():
        pid=int(pidfile.read_text())
        try:os.kill(pid,0);raise RuntimeError(f'Worker already running: {pid}')
        except ProcessLookupError:pass
    with (exp/'logs'/f'worker_gpu{gpu}.log').open('a') as log:
        process=subprocess.Popen([sys.executable,'-u',str(exp/'worker.py'),'--gpu',str(gpu),'--methods',*methods],
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pidfile.write_text(str(process.pid));records.append(dict(pid=process.pid,gpu=gpu,methods=methods,time=time.time()))
(exp/f'launch_{args.group}.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
