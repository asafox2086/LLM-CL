"""Training-only answer-type supervision; no model loaded on import.

This is preparation, not a validated experiment worker. The original answer
loss is copied by AST with only its return value extended. No hidden-state
stack, extra decoder pass, or answer-token representation is requested.
"""
import ast
import hashlib
import random
from pathlib import Path

RUNTIME_SHA = '4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240'
TYPES = ('single_choice', 'multiple_choice', 'numerical')


def bind_sources(root, source_dir, binding):
    assert binding['schema'] == 'type-auxiliary-supervision-v1'
    assert set(binding['sources']) == {'worker.py', 'helpers.py', 'training.py'}
    for name, expected in binding['sources'].items():
        assert hashlib.sha256((source_dir/name).read_bytes()).hexdigest() == expected
    assert binding['limits'] == {'prefix': 3600, 'train': 2400, 'eval': 600}
    required = {'exp/two_backbone_native_cl_20261017_v2/'+p.name
                for p in (root/'exp/two_backbone_native_cl_20261017_v2').glob('*.py')}
    required.update({
        'exp/conditional_shared_budget_20261026/v1/worker.py',
        'exp/conditional_shared_budget_20261026/v1/adaptation.py',
        'exp/osft_increment_screen_20261019/v1/study.py',
        'exp/position_coordinate_control_20261022/v1/core.py',
        'exp/ffn_static_delete_20261025/v1/core.py',
        'exp/paired_image_support_20261024/v1/common.py',
        *('exp/plain_r8_gpu_control_20261025/v1/'+name for name in
          ('control_adaptation.py','control_deployment.py','common.py','readouts.py'))})
    assert required <= set(binding['inherited_sources'])
    for relative, expected in binding['inherited_sources'].items():
        path = Path(relative)
        assert not path.is_absolute() and '..' not in path.parts and path.suffix == '.py'
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == expected, relative
    return dict(binding['limits'])


def bind_answer_loss(runtime_path, torch):
    raw = runtime_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == RUNTIME_SHA
    tree = ast.parse(raw)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'answer_loss')
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert len(returns) == 1 and isinstance(returns[0].value, ast.Tuple)
    assert len(returns[0].value.elts) == 2
    returns[0].value.elts.append(ast.parse(
        'output.last_hidden_state[:, prompt_length-1, :]', mode='eval').body)
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 str(runtime_path) + ':return_prompt_state', 'exec'), namespace)
    return namespace['answer_loss']


def fit_labels(rows):
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)) == 69
    true = [TYPES.index(r['metadata']['question_type']) for r in rows]
    shuffled = list(true)
    random.Random(42).shuffle(shuffled)
    assert sorted(true) == sorted(shuffled)
    return dict(zip(ids, true)), dict(zip(ids, shuffled))


def make_head(torch, hidden_size, device):
    assert hidden_size == 896, 'Bind to this experiment backbone, not a guessed dimension'
    # Initialize on CPU with a private generator; neither global CPU nor CUDA
    # streams advance. All arms obtain exactly the same initial observer head.
    generator = torch.Generator(device='cpu').manual_seed(42)
    with torch.device('meta'):
        head = torch.nn.Linear(hidden_size, 3, bias=True, dtype=torch.float32)
    head.to_empty(device='cpu')
    bound = hidden_size ** -0.5
    with torch.no_grad():
        head.weight.uniform_(-bound, bound, generator=generator)
        head.bias.uniform_(-bound, bound, generator=generator)
    return head.to(device)


def auxiliary_loss(torch, head, prompt_hidden, label):
    assert prompt_hidden.shape == (1, 896) and 0 <= label < 3
    # FP32 head CE remains differentiable into prompt state and shared LoRA.
    with torch.autocast(device_type='cuda', enabled=False):
        logits = head(prompt_hidden.float())
        target = torch.tensor([label], device=logits.device, dtype=torch.long)
        return torch.nn.functional.cross_entropy(logits, target)
