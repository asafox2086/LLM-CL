"""Read original 10% Qwen runs and compare acquisition and final F1 on the same questions."""
import argparse
import gzip
import hashlib
import json
import math
import struct
from collections import defaultdict
from pathlib import Path
from statistics import fmean

SCOPE = Path("summary/parameter_efficiency_retention_20261118")
MANIFEST = Path("data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json")
RUNS = {
    "small": "exp/two_backbone_pct10_20261025/private_old_archive/exp/two_backbone_pct10_native_20261025/runs/qwen/pct10/SeqFT_LoRA/training_20261025_222614",
    "expert": "exp/two_backbone_pct10_native_checkpoint_20261027/runs/qwen/pct10/CL_MoE/fresh_cachedS0_20261104",
}

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_bytes())

def save(path, value, compressed=False):
    data = (json.dumps(value, indent=None if compressed else 2, allow_nan=False) + "\n").encode()
    if compressed:
        data = gzip.compress(data, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(data)

def expert_parameters(root, run, config):
    adapter = run / "runner_source/clmoe_adapter.py"
    assert sha(adapter) == config["execution_source_sha256"][adapter.name] == "33946eddfc6424638ded0675ded68860273859c2f71993a6a60bd22556341be9"
    base = Path(config["base_model"])
    target = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
    added = layers = 0
    for shard in sorted(base.glob("*.safetensors")):
        with shard.open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for key, meta in header.items():
            parts = key.split(".")
            if key.startswith("model.layers.") and parts[-1] == "weight" and parts[-2] in target:
                output_size, input_size = meta["shape"]
                added += 8 * 8 * (input_size + output_size) + 8 * input_size
                layers += 1
    resource_path = run / "resources.jsonl"
    records = [json.loads(l) for l in resource_path.read_text().splitlines()]
    record, = [d for d in records if d["stage"] == 19 and d["phase"] == "stage_training_end_persistent"]
    assert layers == 196 and added == record["trainable_parameters"] == 77930496
    assert added * 4 == record["trainable_parameter_bytes"]
    return dict(added_parameters=added, trainable_parameters=record["trainable_parameters"],
                layers=layers, per_layer="8 rank-8 expert pairs and one 8-output bias-free router",
                resources_sha256=sha(resource_path), adapter_sha256=sha(adapter))

def collect(root, which):
    run = root / RUNS[which]
    manifest_path = root / MANIFEST
    manifest = read(manifest_path)
    cfg = read(run/"config.json")
    assert cfg["manifest_sha256"] == sha(manifest_path) == "02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7"
    assert (cfg["condition"],cfg["backbone"],cfg["seed"],cfg["method"]) == ("pct10","qwen",42,{"small":"SeqFT_LoRA","expert":"CL_MoE"}[which])
    status = read(run/"run_status.json")
    assert status["status"] == "completed" and status["completed_stage"] == 19
    params = {}
    provenance = {}
    if which == "expert":
        end = root/"summary/qwen_cl_baseline_20261104/EXIT.json"
        ident = read(end.with_name("IDENTITY.json"))
        done = read(end)
        assert done["returncode"] == 0 and done["pid"] == ident["pid"]
        proc = Path("/proc")/str(done["pid"])/"stat"
        assert not proc.exists() or proc.read_text().split(") ",1)[1].split()[19] != ident["stat"].split(") ",1)[1].split()[19]
        meta = read(run/"checkpoints/latest.json")
        assert meta["stage"] == 19 and meta["sha256"] == sha(run/meta["path"])
        assert all(read(run/f"validity/stage_{i:02d}.json")["passed"] for i in range(1,20))
        params = expert_parameters(root,run,cfg)
        provenance = dict(exit_sha256=sha(end),checkpoint_sha256=meta["sha256"],pid=done["pid"])
    files, scores, tasks = {}, {}, {}
    for task in manifest["tasks"]:
        i = task["task"]
        desc = task["files"]["test"]
        truth_file = root/desc["path"]
        assert sha(truth_file) == desc["sha256"]
        truth = [json.loads(l) for l in truth_file.read_bytes().splitlines()]
        assert len(truth) == desc["count"] == len({r["id"] for r in truth})
        entry = {}
        for stage in sorted({i,19}):
            p = run/f"evaluations/stage_{stage:02d}/T{i:02d}_test_predictions.jsonl"
            native = p.with_suffix(".scores.json")
            vals = []
            with p.open("rb") as f:
                for index,line in enumerate(f):
                    t, r = truth[index], json.loads(line)
                    assert r["id"] == t["id"] and r["references"] == t["answers"] and r["generation_cap"] == task["max_new_tokens"]
                    v = r["scores"]["token_f1"]
                    assert math.isfinite(v) and 0 <= v <= 100
                    vals.append(v)
            n = read(native)
            assert len(vals) == n["count"] == len(truth) and abs(fmean(vals)-n["token_f1"]) < 1e-8
            for path in (p,native):
                files[str(path.relative_to(run))] = sha(path)
            entry[stage] = vals
        scores[i] = [entry[i],entry[19]]
        tasks[i] = dict(QA=len(truth),domain=task["domain"],test_sha256=desc["sha256"],final=fmean(entry[19]))
    assert len(tasks)==19 and sum(t["QA"] for t in tasks.values())==21454
    domains = {}
    for k,native in read(run/"domain_results.json")["token_f1"].items():
        cohort = [t for t in tasks.values() if t["domain"]==k]
        count = sum(t["QA"] for t in cohort)
        score = sum(t["QA"]*t["final"] for t in cohort)/count
        assert count == native["test_count"] and abs(score-native["final"])<1e-8
        domains[k]=score
    proof = dict(run=RUNS[which],source_sha256=sha(Path(__file__)),config_sha256=sha(run/"config.json"),
                 status_sha256=sha(run/"run_status.json"),manifest_sha256=sha(manifest_path),domains=domains,
                 macro_final=fmean(domains.values()),tasks=tasks,parameters=params,files_sha256=files,**provenance)
    out=root/SCOPE/f"qwen10_{which}_columns.json.gz"
    save(out,dict(scores=scores,proof=proof),True)
    print(json.dumps(dict(sha256=sha(out),macro_final=proof["macro_final"],parameters=params)))

def compare(root):
    import numpy as np
    from parameter_efficiency_retention_20261118 import summarize
    m=read(root/MANIFEST)
    exposure=root/"summary/budget_matched_20261117/IMAGE_EXPOSURE_AUDIT.json"
    assert sha(exposure)=="0e2f8d769d50738df96b28c1b673c16a96edee9fa6a1b18ae45de4a185c24683"
    images=read(exposure)["image_content_hashes"]
    inputs = [json.loads(gzip.decompress((root/SCOPE/f"qwen10_{k}_columns.json.gz").read_bytes())) for k in ("small","expert")]
    groups=defaultdict(list)
    for task in m["tasks"]:
        i=str(task["task"]);desc=task["files"]["test"];p=root/desc["path"]
        assert sha(p)==desc["sha256"] and all(d["proof"]["tasks"][i]["test_sha256"]==desc["sha256"] for d in inputs)
        a,b=inputs
        cols=np.asarray(a["scores"][i]+b["scores"][i]).T
        rows=[json.loads(l) for l in p.read_bytes().splitlines()]
        assert cols.shape==(len(rows),4)
        for row,v in zip(rows,cols):
            groups[task["domain"]].append(([images[j] for j in row["images"]],v))
    rng=np.random.default_rng(20261118); means=[];draws=[];reports={}; seen=set()
    for k,rows in sorted(groups.items()):
        parent=list(range(len(rows))); first={}
        def find(i):
            while i!=parent[i]:
                parent[i]=parent[parent[i]]
                i=parent[i]
            return i
        for j,(imgs,_) in enumerate(rows):
            assert imgs
            for h in imgs:
                parent[find(j)]=find(first.setdefault(h,j))
        assert seen.isdisjoint(first);seen.update(first)
        buckets=defaultdict(lambda:np.zeros(5))
        for j,(_,v) in enumerate(rows):
            buckets[find(j)]+=np.r_[1.,v]
        sums=np.asarray(list(buckets.values()));n=len(sums)
        point=sums.sum(0)[1:]/len(rows);parts=[]
        for _ in range(50):
            rep=rng.multinomial(n,np.full(n,1/n),size=100)@sums
            parts.append(rep[:,1:]/rep[:,:1])
        sample=np.concatenate(parts)
        reports[k]=dict(**summarize(point,sample),QA=len(rows),image_clusters=n)
        means.append(point);draws.append(sample)
    macro=summarize(np.mean(means,axis=0),np.mean(draws,axis=0))
    for key,inp in zip(("small_end","expert_end"),inputs):
        assert abs(macro[key]-inp["proof"]["macro_final"])<1e-8
    assert abs(macro["small_end"]-52.854)<0.00051
    report=dict(macro=macro,domains=reports,small=inputs[0]["proof"],expert=inputs[1]["proof"],
                analysis="All 19 tasks, five equally weighted domains; 5000 paired image-component bootstrap draws conditional on these models. Algorithm comparison; parameter count is not independently varied.")
    out=root/SCOPE/"QWEN_PCT10_RETENTION.json";save(out,report)
    print(json.dumps(dict(macro=macro,domains={k:v["gap_change_during_later_tasks"] for k,v in reports.items()},sha256=sha(out))))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path);parser.add_argument("mode",choices=["small","expert","compare"])
    args=parser.parse_args()
    if args.mode=="compare":
        compare(args.root)
    else:
        collect(args.root,args.mode)
