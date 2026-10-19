"""TRAIN-only screen: can OSFT's additional update locations help ordinary LoRA?

No TEST row is opened. Locations and rank come from existing compared methods.
This is a mechanism screen, not a new algorithm or an edge-deployment claim.
"""
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PORT = ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT = ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1'
WORK = Path(__file__).parent/'private'
QV = ['q_proj', 'v_proj']
EXTRA = ['k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
CASES = {
    'LoRA_qv': dict(targets=QV, method='LoRA', learning_rate=1e-4),
    'OSFT_all': dict(targets=QV+EXTRA, method='OSFT', learning_rate=1e-5),
    'LoRA_all': dict(targets=QV+EXTRA, method='LoRA', learning_rate=1e-4),
    'LoRA_extra': dict(targets=EXTRA, method='LoRA', learning_rate=1e-4),
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def write(p, v):
    import os
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)


def original_train():
    policy = json.loads((PORT/'execution_policy.json').read_text())
    info = policy['conditions']['pct01']; path = ROOT/info['manifest']
    assert sha(path) == info['manifest_sha256']
    m = json.loads(path.read_text()); data = {}
    assert len(m['tasks']) == 19 and m['seed'] == 42
    for t in m['tasks']:
        f = t['files']['train']; p = ROOT/f['path']; assert sha(p) == f['sha256']
        rows = [json.loads(s) for s in p.read_text().splitlines()]
        assert len(rows) == f['count']
        data[t['task']] = rows
    assert sum(map(len, data.values())) == 1313
    return path, m, data


def split_train(m, data):
    """Group by exact image bytes and dataset-native image/scene identifiers.

    Twenty QA per domain is a *measurement budget target*, not an optimum.
    Complete groups and at least one FIT row in every native task take priority.
    """
    rows = [(t['task'], t['domain'], r) for t in m['tasks'] for r in data[t['task']]]
    parent = list(range(len(rows))); seen = {}; images = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b): parent[find(a)] = find(b)
    for i, (_, _, r) in enumerate(rows):
        keys = []
        assert r.get('images'), 'This screen requires image QA, not synthetic text views'
        for image in r['images']:
            if image not in images: images[image] = sha(ROOT/image)
            keys.append(('image_sha256', images[image]))
        md = r.get('metadata', {})
        for k in ['original_image_id', 'img_id', 'image_group_sha256', 'original_id']:
            if md.get(k) is not None:
                keys.append((r['dataset'], k, str(md[k])))
        for k in keys:
            if k in seen: union(i, seen[k])
            else: seen[k] = i
    groups = collections.defaultdict(list)
    for i in range(len(rows)): groups[find(i)].append(i)
    counts = collections.Counter(t for t, _, _ in rows); held = collections.Counter(); chosen = set()
    domain_counts = {}; group_records = []
    for indexes in groups.values():
        ids = sorted(rows[i][2]['id'] for i in indexes)
        gid = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
        domains = {rows[i][1] for i in indexes}; assert len(domains) == 1
        group_records.append((gid, indexes, next(iter(domains))))
    for domain in dict.fromkeys(t['domain'] for t in m['tasks']):
        n = 0
        candidates = sorted((g for g in group_records if g[2] == domain),
                            key=lambda g: hashlib.sha256(('42:osft-increment-dev:'+g[0]).encode()).digest())
        for gid, indexes, _ in candidates:
            delta = collections.Counter(rows[i][0] for i in indexes)
            if any(counts[t]-held[t]-v < 1 for t, v in delta.items()): continue
            if abs(n+len(indexes)-20) >= abs(n-20): continue
            chosen.update(indexes); held.update(delta); n += len(indexes)
        assert n > 0, domain
        domain_counts[domain] = n
    fit, dev = {}, {}
    ledger = {'seed': 42, 'domain_QA_budget_target': 20, 'domains_dev_QA': domain_counts,
              'fit_ids': {}, 'dev_ids': {}, 'image_sha256': images, 'groups': {},
              'TEST_rows_opened': 0, 'not_original_main_experiment_split': True}
    for t in m['tasks']:
        tid = t['task']; a = [r for i, (j, _, r) in enumerate(rows) if j == tid and i not in chosen]
        b = [r for i, (j, _, r) in enumerate(rows) if j == tid and i in chosen]
        assert a
        fit[tid] = a; dev[tid] = b
        ledger['fit_ids'][str(tid)] = [r['id'] for r in a]
        ledger['dev_ids'][str(tid)] = [r['id'] for r in b]
    for gid, indexes, domain in group_records:
        membership = {i in chosen for i in indexes}; assert len(membership) == 1
        ledger['groups'][gid] = dict(domain=domain, dev=membership.pop(), ids=[rows[i][2]['id'] for i in indexes])
    assert sum(map(len, fit.values()))+sum(map(len, dev.values())) == 1313
    ledger['FIT_QA'] = sum(map(len, fit.values())); ledger['DEV_QA'] = sum(map(len, dev.values()))
    ledger['native_tasks_without_dev'] = [t for t in dev if not dev[t]]
    return ledger, fit, dev


def training_ast():
    tree = ast.parse((PORT/'run.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loop = next(n for n in fn.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'stage')
    cutoff = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'wanted' for t in n.targets))
    prefix = copy.deepcopy(loop.body[:cutoff-1])
    assert isinstance(prefix[-1], ast.If)
    h = hashlib.sha256(ast.dump(ast.Module(body=prefix, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    return prefix, h


def make_method(model, case, instructions):
    sys.path.insert(0, str(PORT))
    if CASES[case]['method'] == 'OSFT':
        from osft_adapter import OSFTMethod
        return OSFTMethod(model, 'OSFT', instructions)
    return TargetLoRA(model, CASES[case]['targets'])


class TargetLoRA:
    def __init__(self, model, targets):
        import torch
        from torch import nn
        from methods import SeqLinear
        self.model = model; self.name = 'SeqFT_LoRA'; self.layers = []; self.embedding = None
        self.learning_rate = 1e-4; self.original_revision = 'existing-qv-r8-alpha32-dropout0.1-with-location-control'
        for p in model.parameters(): p.requires_grad = False
        candidates = [(n, l) for n, l in model.named_modules() if n.startswith('model.layers.')
                      and n.rsplit('.', 1)[-1] in targets and isinstance(l, nn.Linear)]
        for name, base in candidates:
            parent, child = name.rsplit('.', 1)
            if child in QV:
                layer = SeqLinear(base)
            else:
                # Extra positions must not change the RNG used for shared q/v initialization.
                devices = [base.weight.device.index] if base.weight.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    seed = int(hashlib.sha256(('42:osft-extra-init:'+name).encode()).hexdigest()[:8], 16)
                    torch.manual_seed(seed)
                    layer = SeqLinear(base)
            model.get_submodule(parent).__setattr__(child, layer); self.layers.append(layer)
        assert self.layers
    def set_instruction(self, task): pass
    def begin_stage(self, stage): pass


def cpu_gate():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    sys.path.insert(0, str(PORT))
    from methods import Method
    torch.set_num_threads(2)
    c = Qwen2Config(vocab_size=31, hidden_size=32, intermediate_size=64,
                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    base = Qwen2ForCausalLM(c).eval(); a = copy.deepcopy(base); b = copy.deepcopy(base)
    torch.manual_seed(42); Method(a, 'SeqLoRA', ['x'])
    torch.manual_seed(42); make_method(b, 'LoRA_qv', ['x'])
    assert all(torch.equal(a.state_dict()[n], v) for n, v in b.state_dict().items())
    ids = torch.tensor([[2, 4, 7, 9]])
    assert torch.equal(a(ids).logits, b(ids).logits)
    all_model = copy.deepcopy(base); torch.manual_seed(42); make_method(all_model, 'LoRA_all', ['x'])
    for n, p in b.named_parameters():
        if '.A.' in n or '.B.' in n: assert torch.equal(p, dict(all_model.named_parameters())[n])
    finite = {}
    for case in CASES:
        m = copy.deepcopy(base); torch.manual_seed(42); method = make_method(m, case, ['x'])
        out = m(ids).logits; assert torch.isfinite(out).all()
        out.float().square().mean().backward()
        grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if hasattr(method, 'post_backward'): method.post_backward(1, 0.)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        finite[case] = dict(trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),
                            finite_tiny_HF_forward_backward=True)
    assert not torch.cuda.is_initialized()
    return dict(original_qv_state_and_forward_exact=True, shared_qv_initialization_exact=True,
                four_conditions=finite, CPU_only=True, not_full_model_evidence=True)


def prepare():
    import math
    REPORT.mkdir(parents=True, exist_ok=True)
    assert not (REPORT/'READY.json').exists()
    gate = cpu_gate(); path, m, data = original_train(); ledger, fit, dev = split_train(m, data)
    # Image/scene groups and every FIT task nonempty are substantive screening invariants.
    assert set(sum(ledger['fit_ids'].values(), [] )).isdisjoint(sum(ledger['dev_ids'].values(), []))
    write(REPORT/'SPLIT.json', ledger); write(REPORT/'CPU_GATE.json', gate)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in PORT.glob('*.py')}
    sources.update({str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')})
    for rel in ['code/OSFT/source/src/mini_trainer/osft_utils.py', 'code/OSFT/files_sha256.json',
                'code/OSFT/source_manifest.json', 'exp/two_backbone_native_cl_20261017_v2/execution_policy.json']:
        sources[rel] = sha(ROOT/rel)
    write(REPORT/'READY.json', dict(sources=sources, manifest_sha256=sha(path),
        split_sha256=sha(REPORT/'SPLIT.json'), protocol_sha256=sha(REPORT/'PROTOCOL.md'),
        CPU_GATE_sha256=sha(REPORT/'CPU_GATE.json'), train_AST_sha256=training_ast()[1],
        base_files={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV').glob('*.safetensors')},
        cases=CASES, FIT_QA=ledger['FIT_QA'], DEV_QA=ledger['DEV_QA'],
        updates=sum(math.ceil(len(r)/16) for r in fit.values()),
        stages=19, TEST_QA=0, full_evaluation_automatic=False, scientific_sources_FROZEN=True))


if __name__ == '__main__': prepare()
