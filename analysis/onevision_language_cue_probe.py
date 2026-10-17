"""Fixed-weight output-instruction diagnostic, no new training or scoring.

Two predeclared matched endpoints x four existing instruction views. Probe32
and full507 are separately declared; all full cells remain required regardless
of probe outcomes. The active dispatcher is not modified by this worker.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from onevision_matched_task_domain import Mechanism, checked_checkpoint
from onevision_knowledge_intervention import PROMPTS, reset_seed
from onevision_capacity_probe import finish_resources
from onevision_research_worker import digest
from run import MODEL, freeze, write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"summary/onevision_diagnostics_20261016/matched_task_domain"
REPORT=ROOT/"summary/onevision_diagnostics_20261016/language_cue"
VIEWS=["original","answer_only","question_language","both"]
SEQUENCES=["within_medical","cross_short"]


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def make_plan():
    gate=json.loads((BASE/"CPU_PLAN_CHECK.json").read_text());assert gate["passed"]
    parent=gate["spec"];sources=dict(parent["source_sha256"])
    for p,h in sources.items():assert digest(ROOT/p)==h
    sources[str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
    metric=json.loads((ROOT/"summary/primary_metric_20261017/metric_lock.json").read_text());assert not metric["option_mapping"]
    sources["analysis/answer_f1.py"]=metric["metric_sha256"];assert digest(ROOT/"analysis/answer_f1.py")==metric["metric_sha256"]
    path=ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    assert digest(path)==parent["manifest_sha256"];manifest=json.loads(path.read_text())
    files={k:manifest["tasks"][3]["files"][k] for k in ["train","test"]}
    for f in files.values():assert digest(ROOT/f["path"])==f["sha256"]
    train={r["id"]:r for r in map(json.loads,(ROOT/files["train"]["path"]).read_text().splitlines())}
    full=[r["id"] for r in map(json.loads,(ROOT/files["test"]["path"]).read_text().splitlines())]
    assert full==parent["final_SeqFT_full_anchor_ids"]["4"] and len(full)==507
    probe=parent["anchor_probe_ids"]["4"];assert len(probe)==32 and set(probe)<=set(full)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    checks={};image_sources={}
    for ident in parent["TRAIN_ids"]["4"]:
        row=train[ident];assert not row["question"].count("<image>")
        images=[]
        for p in row["images"]:
            image_sources[p]=digest(ROOT/p)
            with Image.open(ROOT/p) as im:images.append(im.convert("RGB"))
        item={};original_pixels=None;original_sizes=None
        for view in VIEWS:
            text=manifest["tasks"][3]["instruction"]+PROMPTS[view]
            if row.get("context"):text+="\nContext: "+row["context"]
            text+="\nQuestion: "+row["question"]
            content=[{"type":"image"} for _ in images]+[{"type":"text","text":text}]
            prompt=processor.apply_chat_template([{"role":"user","content":content}],add_generation_prompt=True,tokenize=False)
            assert row["question"] in prompt and prompt.count("<image>")==len(images)
            batch=processor(text=prompt,images=images,return_tensors="pt")
            pixels=batch["pixel_values"].to(torch.bfloat16)
            if original_pixels is None:original_pixels=pixels;original_sizes=batch["image_sizes"]
            else:assert torch.equal(pixels,original_pixels) and torch.equal(batch["image_sizes"],original_sizes)
            item[view]={"input_ids_sha256":tensor_sha(batch["input_ids"]),"pixel_values_bf16_sha256":tensor_sha(pixels),
                        "image_sizes_sha256":tensor_sha(batch["image_sizes"]),"prompt_tokens":batch["input_ids"].shape[1]}
        assert len({v["input_ids_sha256"] for v in item.values()})==4,"Prompt change not effective"
        checks[ident]=item
    states={}
    for seq in SEQUENCES:
        p=BASE/f"T04_{seq}__SeqFT_LoRA_complete.json";proof=json.loads(p.read_text())
        assert proof["complete"] and proof["stage1_exact"] and proof["spec"]==parent
        rel="adapters/stage_05.pt";state=ROOT/proof["run"]/rel
        assert digest(state)==proof["artifact_sha256"][rel]
        states[seq]={"parent_completion":str(p.relative_to(ROOT)),"parent_completion_sha256":digest(p),
                     "run":proof["run"],"state_sha256":digest(state)}
    return {"parent_spec":parent,"source_sha256":sources,"trained_states":states,"views":VIEWS,
            "prompts":{v:PROMPTS[v] for v in VIEWS},"files":files,"TRAIN_input_gate":checks,
            "TRAIN_image_sha256":image_sources,"probe_ids":probe,"full_ids":full,
            "model_processor_path":str(MODEL),"optimizer_updates":0,
            "probe_condition_observations":256,"full_condition_observations":4056,
            "full_unique_QA":507,"full_all8_cells_required_regardless_probe_results":True,
            "inference_prompt_views_only_not_new_CL_method":True,"no_gold_language_routing":True}


def worker(sequence,scope,spec):
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="1"
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    for p,h in spec["source_sha256"].items():assert digest(ROOT/p)==h
    for f in spec["files"].values():assert digest(ROOT/f["path"])==f["sha256"]
    for p,h in spec["TRAIN_image_sha256"].items():assert digest(ROOT/p)==h
    state_info=spec["trained_states"][sequence];parent_path=ROOT/state_info["parent_completion"]
    assert digest(parent_path)==state_info["parent_completion_sha256"]
    parent=json.loads(parent_path.read_text());assert parent["spec"]==spec["parent_spec"]
    state_path=ROOT/state_info["run"]/"adapters/stage_05.pt";assert digest(state_path)==state_info["state_sha256"]
    state=checked_checkpoint(state_path)["adapter"]
    name=f"T04_{sequence}_{scope}";assert not (REPORT/(name+"_complete.json")).exists()
    pointer=REPORT/(name+"_attempt.json")
    if pointer.exists():
        previous=json.loads(pointer.read_text())
        write(REPORT/(name+"_interrupted_"+str(time.time_ns())+".json"),previous)
    run=ROOT/"exp/onevision_language_cue_20261017"/name/time.strftime("%Y%m%d_%H%M%S");run.mkdir(parents=True)
    write(pointer,{"run":str(run.relative_to(ROOT)),"spec":spec})
    write(run/"protocol.json",spec)
    manifest,data,*_=freeze();selected=spec["probe_ids"] if scope=="probe" else spec["full_ids"]
    by_id={r["id"]:r for r in data[4,"test"]};rows=[by_id[i] for i in selected]
    train={r["id"]:r for r in data[4,"train"]}
    started=time.time();x=Mechanism(manifest,run);x.process_started=started;x.boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    x.configure(state,finetune_shared=True);x.model.eval();torch.cuda.reset_peak_memory_stats();x.monitor.snapshot("inference_before",model=x.model)
    for view in VIEWS:
        x.prompt(view)
        for ident,values in spec["TRAIN_input_gate"].items():
            batch=x.inputs(train[ident],retain_pixels=True);expected=values[view]
            assert tensor_sha(batch["input_ids"])==expected["input_ids_sha256"]
            assert tensor_sha(batch["pixel_values"])==expected["pixel_values_bf16_sha256"]
            assert tensor_sha(batch["image_sizes"])==expected["image_sizes_sha256"]
        write(run/"input_gates"/(view+".json"),{"passed":True,"view":view,"TRAIN_rows":8,"CPU_input_ids_pixels_image_sizes_exact":True})
        write(REPORT/"status.json",{"phase":"inference","sequence":sequence,"scope":scope,"view":view,"pid":os.getpid(),"run":str(run.relative_to(ROOT))})
        x.evaluate(view,{"anchor":rows},{"anchor":rows})
        after=x.snapshot();assert all(torch.equal(after[n][f],state[n][f]) for n in state for f in ["A","B"])
        write(run/"parameter_gates"/(view+".json"),{"passed":True,"view":view,"all_A_B_exact":True,"original_state_sha256":state_info["state_sha256"]})
        if view=="original":
            rel="evaluations/anchor_after5/anchor_image.jsonl" if scope=="probe" else "evaluations/final_native_anchor/anchor_image.jsonl"
            original=ROOT/parent["run"]/rel;assert digest(original)==parent["artifact_sha256"][rel]
            old=[json.loads(s) for s in original.read_text().splitlines()]
            new=[json.loads(s) for s in (run/"evaluations/original/anchor_image.jsonl").read_text().splitlines()]
            fields=["id","references","prediction","scores","nll"]
            assert [[r[k] for k in fields] for r in old]==[[r[k] for k in fields] for r in new],"Original prompt reload failed; no tolerance"
            write(run/"original_exact_gate.json",{"passed":True,"QA":len(rows),"no_tolerance":True,"reference_sha256":digest(original)})
    for p,h in spec["source_sha256"].items():assert digest(ROOT/p)==h
    x.monitor.snapshot("inference_after",model=x.model)
    finish_resources(x,run,"inference_before","inference_after","fixed-weight4prompt views, no optimizer/training; includes32TRAIN input gates")
    files=[*run.rglob("*.json"),*run.rglob("*.jsonl")]
    write(REPORT/(name+"_complete.json"),{"complete":True,"sequence":sequence,"scope":scope,"spec":spec,
        "run":str(run.relative_to(ROOT)),"optimizer_updates":0,"parameters_unchanged":True,"QA_per_view":len(rows),
        "views":VIEWS,"state_sha256":state_info["state_sha256"],"artifact_sha256":{str(p.relative_to(run)):digest(p) for p in files}})


def main():
    p=argparse.ArgumentParser();p.add_argument("--check",action="store_true");p.add_argument("--sequence",choices=SEQUENCES);p.add_argument("--scope",choices=["probe","full"]);args=p.parse_args();REPORT.mkdir(parents=True,exist_ok=True)
    if args.check:
        spec=make_plan();assert not torch.cuda.is_initialized()
        write(REPORT/"CPU_PLAN_CHECK.json",{"passed":True,"CPU_only":True,"spec":spec,"real_GPU_input_and_reload_gates":"pending"})
        print(json.dumps({"CPU_only":True,"TRAIN_rows":8,"input_views":4,"changed_token_checks":32,"GPU_started":False}));return
    assert args.sequence and args.scope
    gate=json.loads((REPORT/"CPU_PLAN_CHECK.json").read_text());assert gate["passed"]
    worker(args.sequence,args.scope,gate["spec"])


if __name__=="__main__":
    try:main()
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True);write(REPORT/"needs_attention.json",{"pid":os.getpid(),"traceback":traceback.format_exc()});raise
