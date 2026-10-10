import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--methods',nargs='+',required=True);args=p.parse_args()
exp=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='4'
records=[]
for method in args.methods:
    for seed in [42,43,44]:
        name=f'{method}_seed{seed}'; log=exp/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
        status=exp/'runs'/name/'status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='completed':continue
        script='raldl_run.py' if method=='ra_ldl' else 'run.py'
        with log.open('a') as handle:
            start=time.time()
            result=subprocess.run([sys.executable,'-u',str(exp/script),'--method',method,'--seed',str(seed)],stdout=handle,stderr=subprocess.STDOUT)
        records.append(dict(method=method,seed=seed,returncode=result.returncode,seconds=time.time()-start))
        (exp/f'worker_gpu{args.gpu}.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(records[-1]),flush=True)
