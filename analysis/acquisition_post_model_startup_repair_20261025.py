"""One explicit verifier ROOT initialization repair before its first scoring.

No model, prediction, scoring rule, frozen source or original budget is changed.
The failed controller stays failed; its two original error files are archived.
"""
from pathlib import Path
import ast,copy,hashlib,json,os,subprocess,sys,time,traceback

VERIFIER_SHA='f5951d82919691dcf4ad195529e91d075f8702af418642920c0a69b57ca45f2f'
OLD="ROOT = Path(os.environ.get('LLMCL_ROOT', Path(__file__).resolve().parents[3]))"
NEW="ROOT = Path(os.environ['LLMCL_ROOT'])"
FAILURES=('CONTROLLER_FAILURE.json','VERIFIER_EXIT.json')

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def corrected_tree(raw):
    original=ast.parse(raw);old=ast.dump(ast.parse(OLD).body[0],include_attributes=False)
    new=ast.dump(ast.parse(NEW).body[0],include_attributes=False);changed=[]
    class Replace(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.dump(node,include_attributes=False)==old:
                changed.append(node.lineno);return ast.copy_location(ast.parse(NEW).body[0],node)
            return node
    tree=ast.fix_missing_locations(Replace().visit(copy.deepcopy(original)))
    class Inverse(ast.NodeTransformer):
        def visit_Assign(self,node):
            return ast.copy_location(ast.parse(OLD).body[0],node) if ast.dump(node,include_attributes=False)==new else node
    restored=Inverse().visit(copy.deepcopy(tree))
    assert changed==[16] and ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
    return tree,dict(old_assignment=OLD,new_assignment=NEW,changed_assignment_line=16,
        all_other_AST_nodes_unchanged=True,
        original_AST_sha256=hashlib.sha256(ast.dump(original,include_attributes=False).encode()).hexdigest(),
        corrected_AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())

def service(unit):
    env=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus')
    raw=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],env=env,text=True)
    return dict(x.split('=',1) for x in raw.splitlines())

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--check-source':
        path=Path(sys.argv[2]);assert sha(path)==VERIFIER_SHA
        tree,proof=corrected_tree(path.read_bytes());compile(tree,str(path),'exec');print(json.dumps(proof));return
    run=Path(os.environ['LLMCL_RUN']);permit=read(run/'POST_MODEL_STARTUP_REPAIR_PERMIT.json')
    assert sha(__file__)==permit['wrapper_sha256'] and sha(run/'verify.py')==VERIFIER_SHA
    assert sha(run/'READY.json')==permit['ready_sha256']
    ready=read(run/'READY.json')
    for n,h in ready['sources'].items():assert sha(run/n)==h,n
    activation=read(run/'ACTIVATION.json')
    assert permit['absolute_wall_deadline']==activation['unix']+2100
    assert time.time()<permit['absolute_wall_deadline']
    tree,correction=corrected_tree((run/'verify.py').read_bytes())
    if len(sys.argv)>1 and sys.argv[1]=='--score':
        assert (run/'POST_MODEL_STARTUP_REPAIR_STARTED.json').exists()
        assert not (run/'VERIFY_STARTED.json').exists() and not (run/'VERIFIED.json').exists()
        exec(compile(tree,str(run/'verify.py')+' [ROOT initialization repaired]','exec'),
             dict(__name__='__main__',__file__=str(run/'verify.py')))
        return
    tick=time.monotonic()
    assert not (run/'POST_MODEL_STARTUP_REPAIR_STARTED.json').exists()
    assert not (run/'VERIFY_STARTED.json').exists() and not (run/'VERIFIED.json').exists()
    original_unit=service(ready['unit'])
    assert original_unit==dict(ActiveState='failed',MainPID='0',ExecMainStatus='1')
    start=read(run/'acquisition_readout_STARTED.json');end=read(run/'acquisition_readout_COMPLETE.json');ex=read(run/'acquisition_readout_EXIT.json')
    assert ex['returncode']==0 and ex['pid']==start['identity']['pid'] and end['identity']==start['identity']
    assert end['new_generations']==80 and end['new_scores']==0 and end['seconds']<=1800
    assert sha(run/'acquisition_readout_COMPLETE.json')==permit['model_complete_sha256']
    for n,h in end['files'].items():assert sha(run/n)==h,n
    who=start['identity'];proc=Path('/proc')/str(who['pid'])/'stat'
    assert not proc.exists() or proc.read_text().rsplit(') ',1)[1].split()[19]!=str(who['startticks'])
    failure=read(run/'VERIFIER_EXIT.json')
    assert failure['returncode']==1 and failure['stdout']==''
    assert 'ROOT = Path(os.environ.get' in failure['stderr'] and 'IndexError: 3' in failure['stderr']
    archive=run/'post_model_startup_failure_20261025';archive.mkdir()
    for n in FAILURES:
        assert sha(run/n)==permit['failed_files'][n]
        (run/n).rename(archive/n);assert sha(archive/n)==permit['failed_files'][n]
    write(run/'POST_MODEL_STARTUP_REPAIR_STARTED.json',dict(unix=time.time(),pid=os.getpid(),
        permit_sha256=sha(run/'POST_MODEL_STARTUP_REPAIR_PERMIT.json'),correction=correction,
        original_failed_unit=original_unit,previous_scores=0,model_worker_exited=True,
        first_score_marker_absent=True,original_failure_archive=archive.name))
    p=subprocess.run([sys.executable,'-u',__file__,'--score'],env=os.environ.copy(),capture_output=True,text=True,
        timeout=min(180,permit['absolute_wall_deadline']-time.time()))
    write(run/'VERIFIER_EXIT.json',dict(unix=time.time(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
        post_model_startup_repair=True,original_failed_exit=archive.name+'/VERIFIER_EXIT.json'))
    assert p.returncode==0,p.stderr
    verified=read(run/'VERIFIED.json');assert verified['complete'] and verified['new_scores']==80
    assert verified['source_COMPLETE_sha256']==permit['model_complete_sha256']
    for n,h in end['files'].items():assert sha(run/n)==h,n
    for n,h in ready['sources'].items():assert sha(run/n)==h,n
    assert sha(run/'READY.json')==permit['ready_sha256'] and time.time()<=permit['absolute_wall_deadline']
    assert service(ready['unit'])==original_unit
    write(run/'POST_MODEL_STARTUP_REPAIR_VERIFIED.json',dict(complete=True,unix=time.time(),
        VERIFIED_sha256=sha(run/'VERIFIED.json'),permit_sha256=sha(run/'POST_MODEL_STARTUP_REPAIR_PERMIT.json'),
        wrapper_sha256=sha(__file__),ready_sha256=permit['ready_sha256'],correction=correction,
        model_complete_sha256=permit['model_complete_sha256'],original_failed_files=permit['failed_files'],
        original_failure_archive=archive.name,original_unit_failed_preserved=True,
        original_failed_verifier_scores=0,first_successful_scoring_new_scores=80,scoring_executions=1,
        new_models=0,new_generations=0,model_outputs_and_frozen_sources_unchanged=True,
        absolute_original_deadline_preserved=True,absolute_wall_deadline=permit['absolute_wall_deadline'],
        seconds=time.monotonic()-tick,elapsed_since_original_activation=time.time()-activation['unix']))

if __name__=='__main__':
    try:main()
    except BaseException:
        if 'LLMCL_RUN' in os.environ:
            p=Path(os.environ['LLMCL_RUN'])/'POST_MODEL_STARTUP_REPAIR_FAILURE.json'
            if not p.exists():write(p,dict(unix=time.time(),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise
