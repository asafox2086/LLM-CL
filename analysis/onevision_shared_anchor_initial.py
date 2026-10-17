"""Inference-only common acquired stage1 on the full native medical anchors.

New measurement after the first cross-domain contrast exposed missing full
acquisition references. Original training/checkpoints/predictions stay immutable.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from onevision_matched_task_domain import Mechanism, checked_checkpoint
from onevision_knowledge_intervention import reset_seed
from onevision_capacity_probe import finish_resources
from onevision_research_worker import digest
from run import freeze, write

ROOT = Path(__file__).resolve().parents[1]
MATCHED = ROOT / "summary/onevision_diagnostics_20261016/matched_task_domain"
REPORT = ROOT / "summary/onevision_diagnostics_20261016/shared_anchor_initial"


def plan():
    parent = json.loads((MATCHED/"CPU_PLAN_CHECK.json").read_text())
    assert parent["passed"] and parent["GPU_used"] is False
    old = parent["spec"]
    sources = dict(old["source_sha256"])
    for p,h in sources.items():
        assert digest(ROOT/p)==h
    sources[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    metric = json.loads((ROOT/"summary/primary_metric_20261017/metric_lock.json").read_text())
    assert not metric["option_mapping"] and digest(ROOT/"analysis/answer_f1.py")==metric["metric_sha256"]
    sources["analysis/answer_f1.py"] = metric["metric_sha256"]
    path = ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    assert digest(path)==old["manifest_sha256"]
    manifest = json.loads(path.read_text())
    inputs = {}
    for anchor,count in [(4,507),(2,228)]:
        f = manifest["tasks"][anchor-1]["files"]["test"]
        assert digest(ROOT/f["path"])==f["sha256"]
        ids = [r["id"] for r in map(json.loads,(ROOT/f["path"]).read_text().splitlines())]
        assert ids==old["final_SeqFT_full_anchor_ids"][str(anchor)] and len(ids)==len(set(ids))==count
        assert set(old["anchor_probe_ids"][str(anchor)]) <= set(ids)
        inputs[str(anchor)] = {"test_file":f,"full_ids":ids,"reload_gate_ids":old["anchor_probe_ids"][str(anchor)]}
    return {"parent_spec":old,"source_sha256":sources,"inputs":inputs,"optimizer_updates":0,
            "checkpoint":"common canonical stage1; never choose by TEST scores",
            "evaluation":"fresh full507/228 each, preceded by exact32 original stage1 reload gate",
            "posthoc_followup":True,"goal":"distinguish full-anchor forgetting from absent same-domain positive transfer",
            "limits":["Source full21454 not evaluated here; no full five-domain FWT/BWT.",
                      "Same anchor and already observed test pool; not untouched confirmation.",
                      "Adds inference only; all original training/data/model settings preserved."]}


def worker(anchor,spec):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="1"
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    name=f"T{anchor:02d}";old=json.loads((MATCHED/f'{name}_repeat_anchor__SeqFT_LoRA_complete.json').read_text())
    assert old["complete"] and old["stage1_exact"] and old["spec"]==spec["parent_spec"]
    canonical_path=MATCHED/f"canonical_anchor{anchor}.pt"
    canonical=checked_checkpoint(canonical_path);assert canonical["source_sha256"]==spec["parent_spec"]["source_sha256"]
    gate=json.loads((ROOT/old["run"]/"stage1_exact_gate.json").read_text())
    assert gate["canonical_sha256"]==digest(canonical_path)
    stage1_path=ROOT/old["run"]/"adapters/stage_01.pt"
    assert digest(stage1_path)==old["artifact_sha256"]["adapters/stage_01.pt"]
    stage1=checked_checkpoint(stage1_path)["adapter"]
    assert stage1.keys()==canonical["adapter"].keys()
    assert all(torch.equal(stage1[n][f],canonical["adapter"][n][f]) for n in stage1 for f in ("A","B"))
    pointer=REPORT/f"{name}_attempt.json"
    if pointer.exists():
        interrupted=json.loads(pointer.read_text())
        write(REPORT/f"{name}_interrupted_{time.time_ns()}.json",interrupted)
    run=ROOT/"exp/onevision_shared_anchor_initial_20261017"/name/time.strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True);write(pointer,{"run":str(run.relative_to(ROOT)),"spec":spec})
    write(run/"protocol.json",spec)
    manifest,data,*_=freeze();full=data[anchor,"test"]
    assert [r["id"] for r in full]==spec["inputs"][str(anchor)]["full_ids"]
    by_id={r["id"]:r for r in full};probes=[by_id[i] for i in spec["inputs"][str(anchor)]["reload_gate_ids"]]
    x=Mechanism(manifest,run);x.process_started=time.time();x.boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    x.prompt("original");x.configure(canonical["adapter"],finetune_shared=True);x.model.eval()
    torch.cuda.reset_peak_memory_stats();x.monitor.snapshot("inference_before",model=x.model)
    write(REPORT/"status.json",{"phase":"reload_gate","anchor":anchor,"pid":os.getpid(),"run":str(run.relative_to(ROOT))})
    x.evaluate("reload_gate",{"anchor":probes},{"anchor":probes})
    original_path=ROOT/old["run"]/"evaluations/anchor_after1/anchor_image.jsonl"
    assert digest(original_path)==old["artifact_sha256"]["evaluations/anchor_after1/anchor_image.jsonl"]
    original=[json.loads(s) for s in original_path.read_text().splitlines()]
    current=[json.loads(s) for s in (run/"evaluations/reload_gate/anchor_image.jsonl").read_text().splitlines()]
    fields=["id","references","prediction","scores","nll"]
    assert [[r[k] for k in fields] for r in original]==[[r[k] for k in fields] for r in current],"Common stage1 exact reload failed; no tolerance"
    write(run/"reload_exact_gate.json",{"passed":True,"no_tolerance":True,"QA":32,"reference_sha256":digest(original_path),"canonical_sha256":digest(canonical_path)})
    write(REPORT/"status.json",{"phase":"full_acquired_anchor","anchor":anchor,"pid":os.getpid(),"run":str(run.relative_to(ROOT))})
    x.evaluate("full_acquired_anchor",{"anchor":full},{"anchor":full})
    after=x.snapshot()
    assert all(torch.equal(after[n][f],canonical["adapter"][n][f]) for n in after for f in ("A","B"))
    for p,h in spec["source_sha256"].items():assert digest(ROOT/p)==h
    x.monitor.snapshot("inference_after",model=x.model)
    finish_resources(x,run,"inference_before","inference_after","inference-only acquired-stage1 anchor; optimizer_updates0")
    artifacts=[*run.rglob("*.json"),*run.rglob("*.jsonl")]
    write(REPORT/f"{name}_complete.json",{"complete":True,"anchor":anchor,"spec":spec,"run":str(run.relative_to(ROOT)),"optimizer_updates":0,"full_anchor_QA":len(full),"parameters_unchanged":True,"canonical_sha256":digest(canonical_path),"parent_completion_sha256":digest(MATCHED/f'{name}_repeat_anchor__SeqFT_LoRA_complete.json'),"artifact_sha256":{str(p.relative_to(run)):digest(p) for p in artifacts}})


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument("--check",action="store_true");p.add_argument("--anchor",type=int,choices=[4,2]);args=p.parse_args()
    spec=plan()
    if args.check:
        assert not torch.cuda.is_initialized()
        write(REPORT/"CPU_PLAN_CHECK.json",{"passed":True,"CPU_only":True,"spec":spec,"full_counts":{"4":507,"2":228},"actual_GPU_reload_and_scores":"pending"})
        print(json.dumps({"CPU_only":True,"full_QA":735,"GPU_started":False}));return
    assert args.anchor is not None
    assert not (REPORT/f"T{args.anchor:02d}_complete.json").exists(),"Completed anchor does not need rerun"
    worker(args.anchor,spec)


if __name__=="__main__":
    try:main()
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True);write(REPORT/"needs_attention.json",{"pid":os.getpid(),"traceback":traceback.format_exc()});raise
