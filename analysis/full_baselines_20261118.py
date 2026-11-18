"""Run the remaining full-data baselines after the two priority jobs have GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import full_only_20261118 as primary


SCOPE = "full_baselines_20261118"
COMMON = ["SpaRTA", "PASs_MoE", "OSFT", "PaRSP"]
# GiB for original checkpoint format, output, and method-specific state.
SPACE = {
    "onevision": dict(SpaRTA=4, PASs_MoE=5, OSFT=6, PaRSP=12, SMoLoRA=4,
                      FOREVER=7, FullFT=70),
    "qwen": dict(SpaRTA=7, PASs_MoE=9, OSFT=11, PaRSP=32, SMoLoRA=8,
                 FOREVER=14, FullFT=180),
}


def inventory(host):
    if host == "old2080":
        return [("onevision", method) for method in COMMON]
    return ([("qwen", method) for method in COMMON + ["SMoLoRA"]] +
            [("onevision", "SMoLoRA")] +
            [(backbone, method) for backbone in ("onevision", "qwen")
             for method in ("FOREVER", "FullFT")])


def versioned(path, blob):
    if path.exists():
        assert path.read_bytes() == blob, path
    else:
        path.write_bytes(blob)


def prepare(root, host):
    scope = root / "summary" / SCOPE
    scope.mkdir(parents=True, exist_ok=True)
    base = root / "exp" / primary.NAME
    study = primary.read(root / "summary" / primary.NAME / "DESIGN.json")
    assert study["manifest_sha256"] == primary.sha(root / study["manifest"]) == primary.MANIFEST_SHA
    assert (study["train"], study["test"], study["updates"]) == (131018, 21454, 8197)
    hosts = {"old2080": "onevision"} if host == "old2080" else {
        "new3090": "qwen", "new3090_onevision": "onevision"}
    variants = {"native": base}
    if host == "new3090":
        variants.update(forever=root / "exp/two_backbone_pct10_forever_dynamic_20261025",
                        fullft=root / "exp/full_parameter_seq_20261028/v2")
    bindings = {}
    for kind, source in variants.items():
        dest = root / "exp" / (SCOPE + "_" + kind)
        dest.mkdir(exist_ok=True)
        edits = {}
        if kind == "native":
            edits["run.py"] = [("choices=['full', 'pct01']", "choices=['full']"),
                               ("'primary_score': 'rougeL'", "'primary_score': 'token_f1'")]
        elif kind == "forever":
            edits["run.py"] = [("choices=['pct10']", "choices=['full']")]
        else:
            edits["run.py"] = [
                ("choices=['pct01','pct10']", "choices=['full']"),
                ("in (1,2,3), 'NEW GPU0 is reserved'", "in (0,1,2,3)"),
                ("{'pct01':1313,'pct10':13105}", "{'full':131018}"),
                ("{'pct01':.01,'pct10':.10}", "{'full':1.0}"),
            ]
            edits["core.py"] = [("in ('pct01','pct10')", "== 'full'")]
            edits["reload_probe.py"] = edits["core.py"]
        for name in edits:
            assert (source / name).is_file()
        for file in sorted(source.glob("*.py")):
            blob = file.read_bytes()
            for before, after in edits.get(file.name, []):
                assert blob.count(before.encode()) == 1, (file, before)
                blob = blob.replace(before.encode(), after.encode())
            compile(blob, str(dest / file.name), "exec")
            versioned(dest / file.name, blob)
        if kind != "fullft":
            policy = primary.read(source / "execution_policy.json")
            methods = ["FOREVER"] if kind == "forever" else COMMON + ["SMoLoRA"]
            policy.update(methods=methods, conditions=primary.read(base / "execution_policy.json")["conditions"],
                          hosts={k: dict(backbone=v, reserved_gpus=[], conditions={"full": list(range(4))})
                                 for k, v in hosts.items()},
                          ownership={k: ["full:" + m for m in methods] for k in hosts},
                          checkpoint_all_stages_methods=[], evaluation="Initial all, current and next, final all.")
            for key in ("future_full", "scope_revision_20261018"):
                policy.pop(key, None)
            versioned(dest / "execution_policy.json", json.dumps(policy, indent=2).encode())
        bindings[kind] = {p.name: primary.sha(p) for p in dest.iterdir() if p.is_file()}
    design = dict(host=host, methods=inventory(host), manifest=study["manifest"],
                  manifest_sha256=primary.MANIFEST_SHA, train=131018, test=21454, main_updates=8197,
                  seed=42, source=primary.sha(Path(__file__)), ports=bindings)
    versioned(scope / "DESIGN.json", json.dumps(design, indent=2).encode())


def run_queue(root, host):
    scope = root / "summary" / SCOPE
    design = primary.read(scope / "DESIGN.json")
    assert design["source"] == primary.sha(Path(__file__)) and design["host"] == host
    prior = root / "summary" / primary.NAME
    primary.put(scope / "QUEUE_STARTED.json", dict(pid=os.getpid(), startticks=primary.ticks(os.getpid()),
                                                    unix=time.time(), host=host))
    pending = inventory(host)
    while pending:
        # No future admissions from the research controller once both identities exist.
        if not all((prior / (m + "_IDENTITY.json")).exists() for m in primary.METHODS):
            time.sleep(900)
            continue
        claimed = {primary.read(prior / (m + "_IDENTITY.json"))["gpu"] for m in primary.METHODS
                   if not (prior / (m + "_EXIT.json")).exists()}
        cards = [g for g in primary.idle_cards() if g not in claimed]
        project = primary.occupied_by_project(root) | claimed
        if not cards or (host == "new3090" and len(project) >= 3):
            time.sleep(900)
            continue
        # Reserve the space still needed for priority jobs' 19 snapshots.
        remainder = 0
        for method, budget in zip(primary.METHODS, (6050000000, 110000000) if host == "new3090"
                                  else (3020000000, 60000000)):
            if not (prior / (method + "_EXIT.json")).exists():
                ck = root / "exp" / primary.NAME / "runs" / primary.HOSTS[host] / method / "checkpoints"
                remainder += max(0, budget - sum(p.stat().st_size for p in ck.glob("stage_*.pt")))
        free = primary.available(root) - remainder
        job = next((j for j in pending if SPACE[j[0]][j[1]] * 1024**3 <= free), None)
        if job is None:
            time.sleep(900)
            continue
        backbone, method = job
        kind = {"FullFT": "fullft", "FOREVER": "forever"}.get(method, "native")
        port = root / "exp" / (SCOPE + "_" + kind)
        assert all(primary.sha(port / n) == digest for n, digest in design["ports"][kind].items())
        assert primary.sha(root / design["manifest"]) == primary.MANIFEST_SHA
        run = port / "runs" / backbone / "full" / method
        args = ["--condition", "full", "--run-dir", str(run)]
        if kind == "fullft":
            args += ["--backbone", backbone, "--manifest", str(root / design["manifest"]),
                     "--manifest-sha256", primary.MANIFEST_SHA]
        else:
            args += ["--method", method]
        label = backbone + "_" + method
        gpu = cards[0]
        if gpu not in primary.idle_cards() or (host == "new3090" and len(primary.occupied_by_project(root) | claimed) >= 3):
            continue
        assert not run.exists(), run
        role = "new3090_onevision" if host == "new3090" and backbone == "onevision" else host
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, LLMCL_HOST=role, LLMCL_BACKBONE=backbone,
                   LLMCL_ROOT=str(root), OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONUNBUFFERED="1")
        with (scope / (label + ".log")).open("x") as log:
            proc = subprocess.Popen([sys.executable, str(port / "run.py"), *args], cwd=root, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            primary.put(scope / (label + "_IDENTITY.json"), dict(pid=proc.pid, gpu=gpu,
                        startticks=primary.ticks(proc.pid), run=str(run), unix=time.time()))
            rc = proc.wait()
        status_file = run / "run_status.json"
        status = primary.read(status_file) if status_file.exists() else {}
        primary.put(scope / (label + "_EXIT.json"), dict(rc=rc, status=status, unix=time.time()))
        pending.remove(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "queue"])
    parser.add_argument("root", type=Path)
    parser.add_argument("host", choices=primary.HOSTS)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else run_queue)(args.root, args.host)
