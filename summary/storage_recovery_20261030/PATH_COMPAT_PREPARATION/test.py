"""AST whitelist and six actual temporary-path cases; zero models/metric calls."""
from pathlib import Path
import ast,copy,difflib,hashlib,importlib.util,json,os,tempfile
R=Path('/data1/liyapeng/program/LLMCL');D=R/'summary/storage_recovery_20261030/PATH_COMPAT_PREPARATION'
ORIG=R/'analysis/stage_checkpoint_native_complete_verify_20261027.py';NEW=D/'consumer.py';BASE=R/'analysis/pct10_verified_baseline_report_20261025.py'
PINS={ORIG:'1434c424dcc7d003494da33054bf6ef823c3b49d7e3710805ff360856bf80cc2',NEW:'414a424f9bb5a7c631661fffb83bf1d7cf678c1eacb672f56bc5094d75e38c0a',BASE:'2b2a3202fa049c911324081aacb31d0e7c7508b1020a2727893f1f85ab15c376'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n):return ast.dump(n,include_attributes=False)
def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def write(p,j):
 with p.open('x') as f:json.dump(j,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
for p,h in PINS.items():assert sha(p)==h
orig=ast.parse(ORIG.read_bytes());new=ast.parse(NEW.read_bytes());base=ast.parse(BASE.read_bytes())
# Module differences are only inherited path adaptation and truthful proof fields.
for name in {n.name for n in orig.body if isinstance(n,ast.FunctionDef)}-{'inherited','main'}:assert dump(function(orig,name))==dump(function(new,name)),name
om=function(orig,'main');nm=copy.deepcopy(function(new,'main'))
def verified(fn):
 return next(n.value for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='verified' for t in n.targets))
a=verified(om);b=verified(nm);oldmeta={k.arg:k.value for k in a.keywords}
b.keywords=[k for k in b.keywords if k.arg not in ('original_native_consumer_sha256','path_compatibility_counts')]
next(k for k in b.keywords if k.arg=='source_adaptations').value=copy.deepcopy(oldmeta['source_adaptations'])
assert dump(om)==dump(nm),'Main control/score logic changed outside truthful proof metadata'
with tempfile.TemporaryDirectory(prefix='path_compat_six_',dir=D) as temporary:
 T=Path(temporary);binding=T/'test_binding.json';binding.write_text('{}')
 spec=dict(function='_verify_uncached')
 names={'sha','eligible','candidate','_verify_uncached','read','rooted'}
 def adapt(tree):
  inherited=function(tree,'inherited');cls=next(n for n in inherited.body if isinstance(n,ast.ClassDef) and n.name=='Adapt')
  ns=dict(ast=ast,binding_path=binding,root=R,spec=spec)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(cls)],type_ignores=[])),str(NEW),'exec'),ns)
  v=ns['Adapt']();nodes=[copy.deepcopy(n) for n in base.body if isinstance(n,ast.FunctionDef) and n.name in names]
  body=ast.fix_missing_locations(v.visit(ast.Module(body=nodes,type_ignores=[])));return v,body
 oldadapt,A=adapt(orig);newadapt,B=adapt(new)
 assert (oldadapt.paths,oldadapt.maps,oldadapt.counters)==(2,1,1)
 assert (newadapt.paths,newadapt.maps,newadapt.counters,newadapt.checkpoint_paths,newadapt.checkpoint_containment)==(2,1,1,2,2)
 class Undo(ast.NodeTransformer):
  assigns=calls=0
  def visit_Assign(self,n):
   expected=ast.parse("checkpoint=run/latest['path']").body[0]
   if dump(n)==dump(expected):self.assigns+=1;return ast.parse("checkpoint=(run/latest['path']).resolve()").body[0]
   return self.generic_visit(n)
  def visit_Call(self,n):
   expected=ast.parse("checkpoint.resolve().is_relative_to((run/'checkpoints').resolve())",mode='eval').body
   if dump(n)==dump(expected):self.calls+=1;return ast.parse("checkpoint.is_relative_to(run/'checkpoints')",mode='eval').body
   return self.generic_visit(n)
 undo=Undo();reversed_B=ast.fix_missing_locations(undo.visit(copy.deepcopy(B)))
 assert (undo.assigns,undo.calls)==(2,2) and dump(A)==dump(reversed_B),'AST difference exceeds four path nodes'
 delta=''.join(difflib.unified_diff(ast.unparse(A).splitlines(True),ast.unparse(B).splitlines(True),fromfile='original inherited AST',tofile='compat inherited AST'))
 with (D/'AST_FOUR_NODE_DIFF.txt').open('x') as f:f.write(delta);f.flush();os.fsync(f.fileno())
 # Execute actual inherited builder only: defines helpers, never calls verifier/metric.
 sp=importlib.util.spec_from_file_location('native_compat_no_main',NEW);mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 score_calls=[0];namespace,actual_adapted_sha=mod.inherited(R,binding,mod.SPECS['pct10'],{},score_calls)
 expected_adapted_sha=hashlib.sha256(dump(B).encode()).hexdigest();assert actual_adapted_sha==expected_adapted_sha and score_calls==[0]
 # Extract exactly the transformed assignment/containment used by both functions.
 tests=[]
 for fn in ('candidate','_verify_uncached'):
  body=function(B,fn);assignment=next(n for n in body.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='checkpoint' for t in n.targets))
  guard=next(n for n in ast.walk(body) if isinstance(n,ast.Call) and dump(n)==dump(ast.parse("checkpoint.resolve().is_relative_to((run/'checkpoints').resolve())",mode='eval').body))
  arguments=ast.arguments(posonlyargs=[],args=[ast.arg(arg='run'),ast.arg(arg='latest')],kwonlyargs=[],kw_defaults=[],defaults=[])
  unit=ast.Module(body=[ast.FunctionDef(name='check',args=arguments,body=[copy.deepcopy(assignment),ast.Assert(test=copy.deepcopy(guard)),ast.parse("checkpoint.stat(); checkpoint.relative_to(ROOT)").body[0],ast.parse("checkpoint.relative_to(ROOT)").body[0],ast.Return(value=ast.Name(id='checkpoint',ctx=ast.Load()))],decorator_list=[])],type_ignores=[])
  ns=dict(ROOT=T);exec(compile(ast.fix_missing_locations(unit),'<actual transformed path section>','exec'),ns)
  for case,expected in [('normal_dir',True),('directory_symlink',True),('leaf_escape',False),('dotdot_escape',False),('absolute_escape',False),('dangling_leaf',False)]:
   c=T/(fn+'_'+case);c.mkdir();run=c/'run';run.mkdir();cp=run/'checkpoints';target=cp/'stage_19.pt';latest=dict(path='checkpoints/stage_19.pt')
   if case=='directory_symlink':real=c/'persistent_cp';real.mkdir();cp.symlink_to(real,target_is_directory=True);target.write_bytes(b'x')
   else:
    cp.mkdir()
    if case=='normal_dir':target.write_bytes(b'x')
    elif case=='leaf_escape':outside=c/'outside.pt';outside.write_bytes(b'x');target.symlink_to(outside)
    elif case=='dotdot_escape':(run/'outside.pt').write_bytes(b'x');latest['path']='checkpoints/../outside.pt'
    elif case=='absolute_escape':outside=c/'outside.pt';outside.write_bytes(b'x');latest['path']=str(outside)
    elif case=='dangling_leaf':target.symlink_to(cp/'missing.pt')
   error=None
   try:p=ns['check'](run,latest);actual=True
   except (AssertionError,FileNotFoundError,ValueError) as e:actual=False;error=type(e).__name__
   assert actual==expected,(fn,case,actual,expected)
   tests.append(dict(function=fn,case=case,expected_accept=expected,actual_accept=actual,error=error))
 result=dict(complete=True,source_sha256={str(p):h for p,h in PINS.items()},original_base_unchanged=True,original_consumer_unchanged=True,only_four_inherited_AST_path_nodes_changed=True,AST_diff_sha256=sha(D/'AST_FOUR_NODE_DIFF.txt'),adapted_AST_sha256=actual_adapted_sha,path_counts=namespace['_path_compat_counts'],six_real_temporary_cases_twice=tests,score_calls=score_calls[0],models=0,actual_run_or_queue_touched=False,temporary_files_removed_by_TemporaryDirectory=True,actual_consumer_not_run=True)
write(D/'AST_AND_SIX_PATH_TESTS.json',result)
print(json.dumps(dict(complete=True,result=str(D/'AST_AND_SIX_PATH_TESTS.json'),sha256=sha(D/'AST_AND_SIX_PATH_TESTS.json'),tests=len(tests),score_calls=score_calls[0],models=0)))
