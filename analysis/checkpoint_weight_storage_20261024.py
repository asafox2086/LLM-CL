"""Read saved graph metadata symbolically; never execute a pickle global or tensor."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickletools
import signal
import time
import zipfile


class Node:
    def __init__(self, kind, args=(), state=None):
        self.kind, self.args, self.state = kind, args, state


class Storage:
    def __init__(self, key, dtype, elements, location):
        self.key, self.dtype, self.elements, self.location = str(key), dtype, elements, location


class Tensor:
    def __init__(self, storage, offset, shape, stride, requires_grad=None):
        assert isinstance(storage, Storage)
        self.storage, self.offset, self.shape, self.stride = storage, offset, tuple(shape), tuple(stride)
        self.requires_grad = requires_grad


DTYPE_BYTES = {'DoubleStorage':8,'FloatStorage':4,'HalfStorage':2,'BFloat16Storage':2,
               'LongStorage':8,'IntStorage':4,'ShortStorage':2,'CharStorage':1,'ByteStorage':1,
               'BoolStorage':1,'ComplexFloatStorage':8,'ComplexDoubleStorage':16,'UntypedStorage':1}


def symbolic_load(raw):
    """A closed pickle opcode VM, with REDUCE/NEWOBJ/BUILD represented as data."""
    stack=[];memo={};mark=object();storages={};opcounts=Counter()
    def popmark():
        index=len(stack)-1
        while stack[index] is not mark:index-=1
        items=stack[index+1:];del stack[index:];return items
    def global_name(value):
        return value.kind if isinstance(value,Node) else None
    def reduce_data(function,args):
        name=global_name(function)
        if name in ('collections.OrderedDict','builtins.dict','__builtin__.dict'):
            return dict(*args)
        if name=='collections.defaultdict':return dict(args[1]) if len(args)>1 else {}
        if name=='collections.Counter':return dict(*args)
        if name in ('builtins.set','__builtin__.set'):return set(*args)
        if name in ('builtins.frozenset','__builtin__.frozenset'):return frozenset(*args)
        if name in ('builtins.list','__builtin__.list'):return list(*args)
        if name in ('builtins.tuple','__builtin__.tuple'):return tuple(*args)
        if name=='_codecs.encode':
            assert len(args)==2 and args[1]=='latin1';return args[0].encode('latin1')
        if name in ('torch._utils._rebuild_tensor','torch._utils._rebuild_tensor_v2'):
            return Tensor(*args[:4],requires_grad=args[4] if len(args)>4 else None)
        if name in ('torch._utils._rebuild_parameter','torch._utils._rebuild_parameter_with_state'):
            tensor=args[0];assert isinstance(tensor,Tensor);tensor.requires_grad=args[1];return tensor
        if name and name.startswith('torch.') and '_rebuild' in name:
            raise ValueError('Unsupported tensor constructor: '+name)
        # No callable named by the archive is ever imported or executed.
        return Node('reduce',args=(function,args))
    def persistent(value):
        assert isinstance(value,tuple) and value[0]=='storage' and len(value)>=5
        _,dtype,key,location,elements,*rest=value
        name=global_name(dtype);assert name and name.startswith('torch.')
        short=name.rsplit('.',1)[-1];assert short in DTYPE_BYTES,short
        record=Storage(key,short,elements,location)
        if record.key in storages:
            old=storages[record.key];assert (old.dtype,old.elements)==(short,elements);return old
        storages[record.key]=record;return record
    for op,arg,pos in pickletools.genops(raw):
        name=op.name;opcounts[name]+=1
        if name in ('PROTO','FRAME'):pass
        elif name=='MARK':stack.append(mark)
        elif name=='STOP':
            assert len(stack)==1;return stack[0],storages,dict(opcounts)
        elif name=='POP':stack.pop()
        elif name=='POP_MARK':popmark()
        elif name=='DUP':stack.append(stack[-1])
        elif name=='NONE':stack.append(None)
        elif name=='NEWTRUE':stack.append(True)
        elif name=='NEWFALSE':stack.append(False)
        elif name in ('INT','BININT','BININT1','BININT2','LONG','LONG1','LONG4','FLOAT','BINFLOAT',
                      'STRING','BINSTRING','SHORT_BINSTRING','UNICODE','BINUNICODE','BINUNICODE8',
                      'SHORT_BINUNICODE','BINBYTES','SHORT_BINBYTES','BINBYTES8','BYTEARRAY8'):stack.append(arg)
        elif name=='EMPTY_LIST':stack.append([])
        elif name=='LIST':stack.append(popmark())
        elif name=='APPEND':value=stack.pop();stack[-1].append(value)
        elif name=='APPENDS':items=popmark();stack[-1].extend(items)
        elif name=='EMPTY_TUPLE':stack.append(())
        elif name=='TUPLE':stack.append(tuple(popmark()))
        elif name in ('TUPLE1','TUPLE2','TUPLE3'):
            count=int(name[-1]);items=stack[-count:];del stack[-count:];stack.append(tuple(items))
        elif name=='EMPTY_DICT':stack.append({})
        elif name=='DICT':
            items=popmark();assert len(items)%2==0;stack.append(dict(zip(items[::2],items[1::2])))
        elif name=='SETITEM':value=stack.pop();key=stack.pop();stack[-1][key]=value
        elif name=='SETITEMS':
            items=popmark();assert len(items)%2==0;stack[-1].update(zip(items[::2],items[1::2]))
        elif name=='EMPTY_SET':stack.append(set())
        elif name=='ADDITEMS':items=popmark();stack[-1].update(items)
        elif name=='FROZENSET':stack.append(frozenset(popmark()))
        elif name in ('PUT','BINPUT','LONG_BINPUT'):memo[int(arg)]=stack[-1]
        elif name=='MEMOIZE':memo[len(memo)]=stack[-1]
        elif name in ('GET','BINGET','LONG_BINGET'):stack.append(memo[int(arg)])
        elif name=='GLOBAL':stack.append(Node(arg.replace(' ','.')))
        elif name=='STACK_GLOBAL':
            symbol=stack.pop();module=stack.pop();stack.append(Node(module+'.'+symbol))
        elif name=='REDUCE':args=stack.pop();function=stack.pop();stack.append(reduce_data(function,args))
        elif name=='NEWOBJ':args=stack.pop();cls=stack.pop();stack.append(Node('instance',args=(cls,args)))
        elif name=='NEWOBJ_EX':kwargs=stack.pop();args=stack.pop();cls=stack.pop();stack.append(Node('instance',args=(cls,args,kwargs)))
        elif name=='BUILD':
            state=stack.pop();obj=stack[-1]
            if isinstance(obj,Node):obj.state=state
            elif isinstance(obj,dict):pass  # OrderedDict bookkeeping, never a module.
            else:raise ValueError('Unexpected BUILD target '+type(obj).__name__)
        elif name=='BINPERSID':stack.append(persistent(stack.pop()))
        else:raise ValueError(f'Unsupported opcode {name} at {pos}')
    raise ValueError('No pickle STOP')


def attributes(node):
    if not isinstance(node,Node):return {}
    state=node.state
    if isinstance(state,dict):return state
    if isinstance(state,tuple) and len(state)==2:
        result={}
        for part in state:
            if isinstance(part,dict):result.update(part)
        return result
    return {}


def tensors(value,path='',seen=None):
    if isinstance(value,Tensor):yield path,value;return
    if seen is None:seen=set()
    if not isinstance(value,(Node,dict,list,tuple,set,frozenset)):return
    if id(value) in seen:return
    seen.add(id(value))
    if isinstance(value,Node):
        yield from tensors(value.state,path+'.state',seen)
        yield from tensors(value.args,path+'.args',seen)
    elif isinstance(value,dict):
        for key,item in value.items():yield from tensors(item,path+'.'+str(key),seen)
    else:
        for i,item in enumerate(value):yield from tensors(item,path+'.'+str(i),seen)


def registered(model):
    result=[];seen=set()
    def walk(module,path):
        if id(module) in seen:return
        seen.add(id(module));state=attributes(module)
        assert isinstance(state.get('_modules',{}),dict),path
        for kind in ('_parameters','_buffers'):
            values=state.get(kind,{})
            assert isinstance(values,dict),(path,kind,type(values).__name__)
            for name,tensor in values.items():
                if tensor is None:continue
                assert isinstance(tensor,Tensor),(path,kind,name,type(tensor).__name__)
                result.append((path+'.'+name,kind,tensor))
        for name,child in state.get('_modules',{}).items():
            if child is not None:walk(child,path+'.'+name)
    walk(model,'model');assert result;return result


def read_checkpoint(path,method):
    begin=time.monotonic()
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist();pkls=[v for v in entries if v.filename.endswith('/data.pkl')]
        assert len(pkls)==1 and pkls[0].file_size<64*1024**2
        raw=archive.read(pkls[0]);payload,storages,opcounts=symbolic_load(raw)
        storage_entries={v.filename.rsplit('/',1)[-1]:v.file_size for v in entries if '/data/' in v.filename}
    assert isinstance(payload,dict) and payload['format']=='llmcl-method-graph-v1' and payload['stage']==19
    assert payload['config']['method']==method
    for key,storage in storages.items():
        assert storage_entries[key]==storage.elements*DTYPE_BYTES[storage.dtype],key
    assert set(storages)==set(storage_entries)
    state=attributes(payload['method']);model=state['model'];weights=registered(model)
    essential=[]
    if method=='CL_MoE':
        assert 'anchors' in state
        essential.extend(tensors(state['anchors'],'method.anchors'))
        for index,layer in enumerate(state['layers']):
            fields=attributes(layer);assert 'task_weights' in fields
            essential.extend(tensors(fields['task_weights'],f'method.layers.{index}.task_weights'))
    registered_keys={tensor.storage.key for _,_,tensor in weights}
    selected=registered_keys|{tensor.storage.key for _,tensor in essential}
    profile=Counter()
    for key in selected:
        storage=storages[key];profile[storage.dtype]+=storage_entries[key]
    names=[dict(name=name,kind=kind,shape=tensor.shape,stride=tensor.stride,storage_offset=tensor.offset,
                storage_key=tensor.storage.key,dtype=tensor.storage.dtype,
                logical_elements=math.prod(tensor.shape),requires_grad=tensor.requires_grad) for name,kind,tensor in weights]
    logical_unique={}
    for record in names:
        identity=(record['storage_key'],record['storage_offset'],tuple(record['shape']),tuple(record['stride']))
        logical_unique[identity]=record
    extra={}
    for name,tensor in tensors(model,'model'):
        if tensor.storage.key not in selected:
            extra.setdefault(tensor.storage.key,dict(first_path=name,shape=tensor.shape,dtype=tensor.storage.dtype,storage_bytes=storage_entries[tensor.storage.key]))
    return dict(stage=payload['stage'],format=payload['format'],checkpoint_path=str(path),
        config_method=payload['config']['method'],config_backbone=payload['config'].get('backbone'),
        config_model=payload['config'].get('model'),config_condition=payload['config'].get('condition'),
        manifest_sha256=payload['config']['manifest_sha256'],model_registered_layout=names,
        registered_buffers=[r for r in names if r['kind']=='_buffers'],
        registered_module_tensor_entries=len(names),registered_unique_tensor_views=len(logical_unique),
        registered_unique_view_coefficients=sum(r['logical_elements'] for r in logical_unique.values()),
        essential_unregistered=[dict(name=name,shape=tensor.shape,dtype=tensor.storage.dtype,storage_key=tensor.storage.key) for name,tensor in essential],
        selected_storage_keys_before_buffer_classification=sorted(selected,key=int),
        all_registered_storage_bytes=sum(storage_entries[k] for k in registered_keys),
        selected_storage_bytes_before_buffer_classification=sum(storage_entries[k] for k in selected),
        selected_storage_dtype_bytes_before_buffer_classification=dict(profile),model_unclassified_storage=list(extra.values()),
        storage_layout={k:dict(dtype=s.dtype,elements=s.elements,bytes=storage_entries[k],location=s.location) for k,s in storages.items()},
        all_checkpoint_storage_bytes=sum(storage_entries.values()),all_checkpoint_storage_count=len(storages),
        outside_selected_weight_storage_bytes=sum(v for k,v in storage_entries.items() if k not in selected),
        pickle_bytes=len(raw),pickle_sha256=hashlib.sha256(raw).hexdigest(),pickle_opcode_counts=opcounts,
        checkpoint_stat_bytes=path.stat().st_size,metadata_reader_wall_seconds=time.monotonic()-begin,
        weights_not_loaded=True,pickle_globals_not_executed=True,raw_weight_bytes_not_read=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--method',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120-second metadata cap')))
    signal.alarm(120);value=read_checkpoint(args.checkpoint,args.method)
    value['reader_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('model_registered_layout','registered_buffers','storage_layout','selected_storage_keys_before_buffer_classification','pickle_opcode_counts','model_unclassified_storage')}))


if __name__=='__main__':main()
