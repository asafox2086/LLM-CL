"""Standard-library protocol/data helpers; safe before Torch/CUDA admission."""
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PORT = ROOT / 'exp/two_backbone_native_cl_20261017_v2'
REPORT = ROOT / 'summary/final_weight_reduction_20261024/sparse_connection_plasticity_v1'
WORK = HERE / 'private'
ASSETS = HERE / 'assets_private'
CASES = ('Frozen', 'Plastic')
BASE_VALUES = 893675552
FINAL_VALUES = 737323040
FFN_SLOTS = 156893184
QV_COORDINATES = 540672
PATTERN_BYTES = 78446592


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def data():
    """Read only original TRAIN containers, split by the existing frozen IDs."""
    old_ready = json.loads((ASSETS / 'screen_READY.json').read_text())
    split_path = ASSETS / 'SPLIT.json'
    assert sha(split_path) == old_ready['split_sha256']
    split = json.loads(split_path.read_text())
    assert (split['FIT_QA'], split['DEV_QA']) == (1213, 100)
    policy = json.loads((PORT / 'execution_policy.json').read_text())
    manifest_path = ROOT / policy['conditions']['pct01']['manifest']
    assert sha(manifest_path) == old_ready['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['seed'] == 42 and len(manifest['tasks']) == 19
    mapped = {}; fit = {}; dev = {}; plan = []; id_group = {}
    for gid, group in split['groups'].items():
        for ident in group['ids']:
            assert ident not in id_group
            id_group[ident] = gid
    for task in manifest['tasks']:
        tid = task['task']
        source = task['files']['train']
        path = ROOT / source['path']
        assert sha(path) == source['sha256']
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == source['count']
        fs = set(split['fit_ids'][str(tid)]); ds = set(split['dev_ids'][str(tid)])
        assert not fs & ds
        fit[tid] = [r for r in rows if r['id'] in fs]
        dev[tid] = [r for r in rows if r['id'] in ds]
        assert [r['id'] for r in fit[tid]] == split['fit_ids'][str(tid)]
        assert [r['id'] for r in dev[tid]] == split['dev_ids'][str(tid)]
        assert len(fit[tid]) + len(dev[tid]) == len(rows)
        mapped[tid, 'train'] = fit[tid]; mapped[tid, 'dev'] = dev[tid]
        order = sorted(fit[tid], key=lambda r: hashlib.sha256(f"42:{tid}:{r['id']}".encode()).digest())
        task['training_order_sha256'] = hashlib.sha256(''.join(r['id'] + '\n' for r in order).encode()).hexdigest()
        task['files'] = {'train': {'count': len(fit[tid])}, 'dev': {'count': len(dev[tid])}}
        plan.append(dict(stage=tid, QA=len(fit[tid]), updates=math.ceil(len(fit[tid]) / 16)))
    assert sum(p['QA'] for p in plan) == 1213 and sum(p['updates'] for p in plan) == 87
    assert sum(map(len, dev.values())) == 100 and len(fit[1]) == 327
    # Only current-S1 representatives determine the initialization statistics.
    groups = {}
    for row in fit[1]:
        groups.setdefault(id_group[row['id']], []).append(row)
    ordered = sorted(groups, key=lambda g: hashlib.sha256(('42:sparse-current-S1:' + g).encode()).digest())
    representatives = [min(groups[g], key=lambda r: hashlib.sha256(('42:sparse-QA:' + r['id']).encode()).digest()) for g in ordered]
    assert len(representatives) >= 32
    calibration, diagnostic = representatives[:16], representatives[16:32]
    assert not {id_group[r['id']] for r in calibration} & {id_group[r['id']] for r in diagnostic}
    return manifest_path, manifest, split, fit, dev, mapped, plan, id_group, calibration, diagnostic


def training_ast():
    tree = ast.parse((PORT / 'run.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in fn.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'stage')
    cutoff = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'wanted' for t in node.targets))
    prefix = copy.deepcopy(loop.body[:cutoff - 1])
    assert isinstance(prefix[-1], ast.If)
    digest = hashlib.sha256(ast.dump(ast.Module(body=prefix, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    return prefix, digest


def check_ready(*, base=False):
    ready = json.loads((REPORT / 'READY.json').read_text())
    for key in ('sources', 'dependencies', 'evidence'):
        for relative, digest in ready[key].items():
            assert sha(ROOT / relative) == digest, relative
    if base:
        for relative, digest in ready['base_files'].items():
            assert sha(ROOT / relative) == digest, relative
    assert ready['training_ast_sha256'] == training_ast()[1]
    assert ready['cases'] == list(CASES)
    assert ready['host'] == 'OLD' and ready['allowed_physical_GPUs'] == [0, 1, 2, 3]
    return ready


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    root = Path('/proc') / str(pid)
    stat = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
    return dict(pid=pid, uid=root.stat().st_uid, startticks=stat[19], pgid=os.getpgid(pid),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def inventory():
    """Read compute owners; utilization=0 alone never means a card is free."""
    import subprocess
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    cards = {}
    for line in gpu.splitlines():
        index, uuid, memory, utilization = [x.strip() for x in line.split(',')]
        cards[int(index)] = dict(uuid=uuid, memory_MiB=int(memory), utilization=int(utilization), owners=[])
    by_uuid = {v['uuid']: v for v in cards.values()}
    for line in apps.splitlines():
        pid, uuid, _ = [x.strip() for x in line.split(',')]
        if uuid in by_uuid:
            try:
                identity = process_identity(int(pid))
            except (FileNotFoundError, ProcessLookupError):
                identity = dict(pid=int(pid), identity_unresolved=True)
            by_uuid[uuid]['owners'].append(identity)
    return cards


def all_weight_layout(model):
    """Runtime-only import; union storage counts all parameters and buffers."""
    from core import Packed2of4Linear
    coefficients = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    packed = [m for m in model.modules() if isinstance(m, Packed2of4Linear)]
    buffers = list(model.named_buffers())
    storages = {}
    for tensor in [*model.parameters(), *(v for _, v in buffers)]:
        if tensor.numel():
            storage = tensor.untyped_storage()
            storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return dict(weight_coefficients=coefficients, parameter_bytes=weight_bytes,
        trainable_coordinates=sum(p.numel() for p in model.parameters() if p.requires_grad),
        packed_modules=len(packed), packed_slots=sum(m.values.numel() for m in packed),
        pattern_bytes=sum(m.pattern.numel() * m.pattern.element_size() for m in packed),
        lookup_bytes=sum(m.pattern_offsets.numel() * m.pattern_offsets.element_size() for m in packed),
        all_buffer_bytes=sum(t.numel() * t.element_size() for _, t in buffers),
        buffer_layout={n: dict(shape=list(t.shape), dtype=str(t.dtype), bytes=t.numel() * t.element_size()) for n, t in buffers},
        registered_storage_union_bytes=sum(storages.values()),
        packed_layers={name: module.metadata() for name, module in model.named_modules() if isinstance(module, Packed2of4Linear)},
        temporary_dense_reconstruction_not_in_registered_storage=True)


def tensor_digest(named_tensors):
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda row: row[0]):
        cpu = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(cpu.dtype).encode()); h.update(str(tuple(cpu.shape)).encode())
        h.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def mask_digest(model):
    from core import Packed2of4Linear
    return tensor_digest([(name + '.' + k, getattr(layer, k))
        for name, layer in model.named_modules() if isinstance(layer, Packed2of4Linear)
        for k in ('pattern', 'pattern_offsets')])


def ffn_fp16_observation(model):
    """Stream layer hashes; retain no extra full-model weight snapshot."""
    import torch
    from core import Packed2of4Linear
    layers = {name: tensor_digest([('values', layer.values.detach().to(torch.float16))])
        for name, layer in model.named_modules() if isinstance(layer, Packed2of4Linear)}
    return dict(layer_sha256=layers,
        sha256=hashlib.sha256(json.dumps(layers, sort_keys=True).encode()).hexdigest(),
        scope='Retained values rounded to prescribed final FP16; hash/changed layers are not semantic importance or F1.')


def finish_resources(monitor, model, phase):
    """Close the inherited sampler without inventing TRAIN labels for init."""
    monitor.done.set()
    monitor.thread.join(timeout=6)
    assert not monitor.thread.is_alive(), 'Sampler must finish before sealing resources'
    monitor.snapshot('phase_finished', monitor.stage, model)
    rows = [json.loads(line) for line in monitor.path.read_text().splitlines()]
    assert rows and all(row['pid'] == os.getpid() for row in rows)
    keys = ('VmRSS_bytes', 'Pss_bytes', 'cpu_uss_bytes', 'gpu_allocated_bytes',
            'gpu_reserved_bytes', 'gpu_process_nvml_bytes')
    before, after = rows[0], rows[-1]
    result = dict(unit='bytes', phase=phase, pid=os.getpid(), samples=len(rows),
        before=before, after=after,
        growth_bytes={key: after[key] - before[key]
            if after.get(key) is not None and before.get(key) is not None else None for key in keys},
        observed_peak_bytes={key: max(row[key] for row in rows if row.get(key) is not None)
            if any(row.get(key) is not None for row in rows) else None for key in keys},
        scope='Entire same-PID phase, including loading, probes, serialization and generation where applicable. '
              'Sampled peaks are not controlled TRAIN or inference peaks; init has no optimizer or TRAIN.')
    write(monitor.run / 'memory_summary.json', result)
    return result
