"""Nohup coordinator; GPU1 only, fresh process per predeclared cap1/3."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_diagnostics_20261016/full_input_repair'


def write(value):
    temp=REPORT/'queue_state.partial';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(REPORT/'queue_state.json')


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=(REPORT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for cap in [1,3]:
        name=f'cap{cap}_full_confirm'
        if (REPORT/(name+'_needs_attention.json')).exists():
            write({'phase':'needs_attention','active_job':name});return
        if (REPORT/(name+'_complete.json')).exists():continue
        while True:
            used,util=map(int,subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            owners=subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if used<500 and util<10 and not owners:break
            write({'phase':'waiting_idle_gpu1','active_job':name,'owners':owners.splitlines()});time.sleep(15)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONUNBUFFERED='1')
        with (REPORT/(name+'_nohup.log')).open('a') as log:
            worker=subprocess.Popen(['/usr/bin/nohup',str(ROOT/'.vision-env/bin/python'),'-u',str(ROOT/'analysis/onevision_full_input_repair.py'),'--cap',str(cap)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write({'phase':'running','active_job':name,'pid':worker.pid,'coordinator_pid':os.getpid()})
            code=worker.wait()
        if code or not (REPORT/(name+'_complete.json')).exists():
            write({'phase':'needs_attention','active_job':name,'exit_code':code});return
    write({'phase':'completed','full_test_qa_per_cap':21454,'caps':[1,3]})


if __name__=='__main__':main()
