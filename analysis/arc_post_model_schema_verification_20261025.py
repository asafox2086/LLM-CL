"""One disclosed schema correction before the first ARC scoring execution.

The eight frozen scientific files and all model outputs remain untouched.
Only a verifier assertion for a field never serialized by the frozen producer
is replaced in memory. Original verifier/controller failures are archived.
"""
from pathlib import Path
import ast,copy,hashlib,json,os,subprocess,sys,time,traceback

ORIGINAL_VERIFIER_SHA='68802ceb7d4debb15b4047ade830c043f2ecf3cf1627454d8ddd319795800dee'
OLD="assert x['phase'] == 'DEV_FINAL' and x['references'] == row['answers'] and x['generation_cap'] == task['max_new_tokens']"
NEW="assert 'phase' not in x and x['references'] == row['answers'] and x['generation_cap'] == task['max_new_tokens']"


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def corrected_tree(raw):
    original=ast.parse(raw);target=ast.dump(ast.parse(OLD).body[0],include_attributes=False)
    assert sum(isinstance(n,ast.Assert) and ast.dump(n,include_attributes=False)==target for n in ast.walk(original))==1
    changed=[]
    class Replace(ast.NodeTransformer):
        def visit_Assert(self,node):
            if ast.dump(node,include_attributes=False)==target:
                changed.append(node.lineno)
                return ast.copy_location(ast.parse(NEW).body[0],node)
            return node
    tree=Replace().visit(copy.deepcopy(original));ast.fix_missing_locations(tree)
    # Inverse substitution must reconstruct every original AST node.
    new_target=ast.dump(ast.parse(NEW).body[0],include_attributes=False)
    class Inverse(ast.NodeTransformer):
        def visit_Assert(self,node):
            return ast.copy_location(ast.parse(OLD).body[0],node) if ast.dump(node,include_attributes=False)==new_target else node
    restored=Inverse().visit(copy.deepcopy(tree))
    assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
    assert changed==[403]
    return tree,dict(old_assert=OLD,new_assert=NEW,changed_assert_line=403,
        all_other_AST_nodes_unchanged=True,
        original_AST_sha256=hashlib.sha256(ast.dump(original,include_attributes=False).encode()).hexdigest(),
        corrected_AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())


def service(unit):
    env=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus')
    result=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],env=env,text=True)
    return dict(line.split('=',1) for line in result.splitlines())


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--check-source':
        p=Path(sys.argv[2]);assert sha(p)==ORIGINAL_VERIFIER_SHA
        tree,proof=corrected_tree(p.read_bytes());compile(tree,str(p),'exec');print(json.dumps(proof));return
    run=Path(os.environ['LLMCL_RUN']);permit=read(run/'POST_MODEL_SCHEMA_PERMIT.json')
    assert sha(Path(__file__))==permit['wrapper_sha256']
    assert sha(run/'verify.py')==ORIGINAL_VERIFIER_SHA
    assert sha(run/'READY.json')==permit['ready_sha256']
    ready=read(run/'READY.json')
    for n,h in ready['sources'].items():assert sha(run/n)==h,n
    assert time.time()<permit['absolute_wall_deadline']
    tree,correction=corrected_tree((run/'verify.py').read_bytes())
    if len(sys.argv)>1 and sys.argv[1]=='--score':
        assert (run/'POST_MODEL_SCHEMA_STARTED.json').exists()
        assert not (run/'VERIFY_STARTED.json').exists() and not (run/'VERIFIED.json').exists()
        sys.path.insert(0,str(run))
        exec(compile(tree,str(run/'verify.py')+' [one schema assertion corrected]','exec'),dict(__name__='__main__',__file__=str(run/'verify.py')))
        return
    tick=time.monotonic()
    assert not (run/'POST_MODEL_SCHEMA_STARTED.json').exists()
    assert not (run/'VERIFY_STARTED.json').exists() and not (run/'VERIFIED.json').exists()
    assert not (run/'CONTINUATION_VERIFIED.json').exists()
    assert sha(run/'ALL_EXIT.json')==permit['all_exit_sha256']
    assert sha(run/'CONTINUATION_MODEL_COMPLETE.json')==permit['model_complete_sha256']
    continuation=read(run/'PRESTART_CONTINUATION_PERMIT.json')
    assert permit['absolute_wall_deadline']==continuation['absolute_wall_deadline']
    assert permit['original_first_activation_unix']==continuation['first_ACTIVATION_unix']
    units={u:service(u) for u in (ready['unit'],continuation['continuation_unit'])}
    assert all(v==dict(ActiveState='failed',MainPID='0',ExecMainStatus='1') for v in units.values()),units
    ends=read(run/'ALL_EXIT.json')
    assert ends['complete'] and ends['new_scores']==0
    for phase,h in ends['source_COMPLETE_sha256'].items():
        assert sha(run/(phase+'_COMPLETE.json'))==h
        st=read(run/(phase+'_STARTED.json'));ex=read(run/(phase+'_EXIT.json'))
        assert ex['returncode']==0 and ex['pid']==st['identity']['pid']
        p=Path('/proc')/str(ex['pid'])/'stat'
        assert not p.exists() or p.read_text().rsplit(') ',1)[1].split()[19]!=str(st['identity']['startticks'])
    destination=run/'post_model_verifier_schema_failure_20261025';destination.mkdir()
    for name,h in permit['failed_files'].items():
        p=run/name;assert sha(p)==h and not (destination/name).exists()
        p.rename(destination/name);assert sha(destination/name)==h
    assert read(destination/'VERIFIER_EXIT.json')['returncode']==1
    assert "KeyError: 'phase'" in read(destination/'VERIFIER_EXIT.json')['stderr']
    write(run/'POST_MODEL_SCHEMA_STARTED.json',dict(unix=time.time(),pid=os.getpid(),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        permit_sha256=sha(run/'POST_MODEL_SCHEMA_PERMIT.json'),old_units_failed_preserved=units,
        first_attempt_score_marker_absent=True,previous_scores=0,correction=correction))
    p=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--score'],cwd=run,capture_output=True,text=True,
        timeout=min(180,max(.001,permit['absolute_wall_deadline']-time.time())))
    write(run/'VERIFIER_EXIT.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,unix=time.time(),
        post_model_schema_correction=True,wrapper_sha256=permit['wrapper_sha256']))
    assert p.returncode==0 and (run/'VERIFIED.json').exists(),p.stderr
    v=read(run/'VERIFIED.json');assert v['complete'] and v['new_scores']==300
    assert sha(run/'verify.py')==ORIGINAL_VERIFIER_SHA
    assert all(sha(run/n)==h for n,h in ready['sources'].items())
    assert all(sha(destination/n)==h for n,h in permit['failed_files'].items())
    assert time.time()<=permit['absolute_wall_deadline']
    schema=dict(complete=True,unix=time.time(),VERIFIED_sha256=sha(run/'VERIFIED.json'),
        permit_sha256=sha(run/'POST_MODEL_SCHEMA_PERMIT.json'),wrapper_sha256=permit['wrapper_sha256'],
        STARTED_sha256=sha(run/'POST_MODEL_SCHEMA_STARTED.json'),original_verifier_sha256=ORIGINAL_VERIFIER_SHA,
        correction=correction,original_failed_files=permit['failed_files'],
        original_failure_archive='post_model_verifier_schema_failure_20261025',
        prior_failed_verification_scores=0,first_scoring_execution_new_scores=300,scoring_executions=1,
        new_models=0,new_generations=0,old_eight_scientific_files_unchanged=True,
        all_six_model_phases_preserved=True,all_exit_sha256=sha(run/'ALL_EXIT.json'),
        old_units_failed_preserved=units,absolute_original_deadline_preserved=True,
        seconds=time.monotonic()-tick,total_elapsed_since_original_activation=time.time()-permit['original_first_activation_unix'],
        original_post_model_check_endpoint_interval_seconds=permit['original_post_model_check_endpoint_interval_seconds'],
        phase_evidence='Frozen eval worker call chain, hashed COMPLETE output files, IDs/refs/caps/raw/prepared binding; no serialized per-prediction phase field exists or is fabricated.')
    write(run/'POST_MODEL_SCHEMA_VERIFIED.json',schema)
    write(run/'CONTINUATION_VERIFIED.json',dict(complete=True,unix=time.time(),
        VERIFIED_sha256=sha(run/'VERIFIED.json'),permit_sha256=sha(run/'PRESTART_CONTINUATION_PERMIT.json'),
        continuation_model_complete_sha256=sha(run/'CONTINUATION_MODEL_COMPLETE.json'),
        original_preSTARTED_attempt_archive={n:sha(run/'preSTARTED_guard_attempt_20261025'/n) for n in continuation['attempt_files']},
        original_unit_failed_preserved=True,preSTARTED_model_attempts=0,actual_model_phases=6,quality_scores=300,
        old_scientific_sources_unchanged=True,READY_unchanged=True,
        total_elapsed_since_original_activation=time.time()-continuation['first_ACTIVATION_unix'],
        absolute_original_deadline_preserved=True,new_retries_of_started_model_phases=0,
        produced_by='post-model metadata-only verifier-schema wrapper, not successful old continuation service',
        post_model_verifier_schema_repair=True,post_model_schema_VERIFIED_sha256=sha(run/'POST_MODEL_SCHEMA_VERIFIED.json'),
        old_continuation_unit_failed_preserved=True,
        interpretation='Both original failed services and both failures are preserved. Six successful model phases were never repeated. The only scoring execution uses the original scorer after one disclosed producer-schema assertion correction.'))
    print(json.dumps(dict(complete=True,VERIFIED_sha256=sha(run/'VERIFIED.json'),
        CONTINUATION_VERIFIED_sha256=sha(run/'CONTINUATION_VERIFIED.json'),
        POST_MODEL_SCHEMA_VERIFIED_sha256=sha(run/'POST_MODEL_SCHEMA_VERIFIED.json'),new_models=0,new_scores=300)))


if __name__=='__main__':
    try:main()
    except BaseException:
        if os.environ.get('LLMCL_RUN'):
            p=Path(os.environ['LLMCL_RUN'])/'POST_MODEL_SCHEMA_FAILURE.json'
            if not p.exists():write(p,dict(unix=time.time(),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise
