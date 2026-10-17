"""CPU-only sealed-cell audit of V2 baseline language acquisition/retention.

Read original runs, never evaluate on GPU or edit any active framework files.
No EM/option parsing, translation, prompt selection or semantic rescoring.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FRAMEWORK=ROOT/"exp/two_backbone_native_cl_20261017_v2"
REPORT=ROOT/"summary/onevision_diagnostics_20261016/baseline_language_profile"
STAGES=[0,3,4,19]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def han(text):return any("\u3400"<=c<="\u9fff" for c in text)


def paired(values,images):
    assert values and len(values)==len(images) and all(math.isfinite(v) for v in values)
    groups=collections.defaultdict(list)
    for value,image in zip(values,images):groups[tuple(image)].append(value)
    clusters=list(groups.values());rng=random.Random(42);boot=[]
    for _ in range(2000):
        chosen=[clusters[rng.randrange(len(clusters))] for _ in clusters]
        boot.append(sum(map(sum,chosen))/sum(map(len,chosen)))
    boot.sort()
    return {"mean":sum(values)/len(values),"image_cluster_95CI":[boot[49],boot[1949]],
            "QA":len(values),"image_clusters":len(clusters)}


def build(paths,host):
    sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    def checked(p,expected=None):
        h=sha(p);assert expected is None or h==expected,str(p)
        sources[str(p.relative_to(ROOT))]=h;return p
    lock=json.loads(checked(ROOT/"summary/primary_metric_20261017/metric_lock.json").read_text())
    assert lock["option_mapping"] is False and lock["additional_prefix_or_answer_extraction"] is False
    module=checked(ROOT/"analysis/answer_f1.py",lock["metric_sha256"])
    s=importlib.util.spec_from_file_location("locked_answer_f1",module);metric=importlib.util.module_from_spec(s);s.loader.exec_module(metric)
    manifest_path=checked(ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json")
    manifest=json.loads(manifest_path.read_text());task=manifest["tasks"][3]
    assert task["native_label"]=="Organ" and task["task"]==4 and task["files"]["test"]["count"]==507
    data={}
    for split in ["train","test"]:
        f=task["files"][split]
        data[split]=[json.loads(s) for s in checked(ROOT/f["path"],f["sha256"]).read_text().splitlines()]
        assert len(data[split])==f["count"]
    gold=data["test"];ids=[r["id"] for r in gold];assert len(set(ids))==507
    groups={"all":list(range(len(ids)))}
    for key,values in [("q_lang",["en","zh"]),("answer_type",["OPEN","CLOSED"])]:
        for value in values:groups[f"{key}={value}"]=[i for i,r in enumerate(gold) if r["metadata"][key]==value]
    for lang in ["en","zh"]:
        for typ in ["OPEN","CLOSED"]:groups[f"q_lang={lang}/answer_type={typ}"]=[i for i,r in enumerate(gold) if r["metadata"]["q_lang"]==lang and r["metadata"]["answer_type"]==typ]
    runs=[];event_prefixes=[]
    for relative in paths:
        run=(ROOT/relative).resolve();assert run.is_relative_to(FRAMEWORK/"runs")
        config=json.loads(checked(run/"config.json").read_text())
        assert config["experiment_id"]==FRAMEWORK.name and config["condition"]=="pct01"
        assert config["framework"]=="shared-v2-uniform-input-and-cache" and config["image_area_cap"]==401408
        assert config["manifest_sha256"]==sha(manifest_path) and config["seed"]==42
        for name,h in config["execution_source_sha256"].items():
            assert Path(name).name==name;checked(run/"runner_source"/name,h);checked(FRAMEWORK/name,h)
        cells=[];observations={};runtime=json.loads((run/"run_status.json").read_text())
        log=run/"logs/events.jsonl";events=[]
        if log.exists():
            lines=log.read_bytes().splitlines(keepends=True)
            if lines and not lines[-1].endswith(b"\n"):lines=lines[:-1]
            prefix=b"".join(lines);events=[json.loads(s) for s in lines]
            event_prefixes.append({"path":str(log.relative_to(ROOT)),"bytes":len(prefix),"sha256":hashlib.sha256(prefix).hexdigest()})
        for stage in STAGES:
            p=run/f"evaluations/stage_{stage:02d}/T04_test_predictions.jsonl";sp=p.with_suffix(".scores.json")
            if not (p.exists() and sp.exists()):cells.append({"stage":stage,"phase":"pending","scores":None});continue
            # A nonzero cell requires every prior stage to be durably complete,
            # with actual positive-lr parameter changes, not just a sealed output.
            for prior in range(1,stage+1):
                stage_path=run/f"stages/stage_{prior:02d}.json"
                if not stage_path.exists():break
                proof=json.loads(checked(stage_path).read_text());assert proof["complete"] and proof["stage"]==prior
                validity=json.loads(checked(run/f"validity/stage_{prior:02d}.json").read_text())
                assert validity["passed"] and validity["positive_lr_updates"]>0 and validity["actual_parameters_changed"]
                assert validity["parameters_before_sha256"]!=validity["parameters_after_sha256"]
                updates=[e for e in events if e.get("event")=="train_update" and e["stage"]==prior]
                expected=math.ceil(manifest["tasks"][prior-1]["files"]["train"]["count"]/16)
                assert len(updates)==validity["updates"]==validity["positive_lr_updates"]==expected
                assert all(math.isfinite(e["loss"]) and math.isfinite(e["gradient_l2_preclip"]) and e["gradient_l2_preclip"]>0 for e in updates)
                assert all(any(lr>0 for lr in e["learning_rate_used"]) for e in updates)
            else:prior=None
            if stage and prior is not None:
                cells.append({"stage":stage,"phase":"waiting_durable_stage_completion","scores":None});continue
            rows=[json.loads(s) for s in checked(p).read_text().splitlines()];aggregate=json.loads(checked(sp).read_text())
            assert [r["id"] for r in rows]==ids and aggregate["count"]==507
            for a,b in zip(rows,gold):
                assert a["references"]==b["answers"] and a["metadata"]==b["metadata"]
                assert metric.answer_f1(a["prediction"],b["answers"])==a["scores"]["token_f1"]
            assert abs(aggregate["token_f1"]-sum(r["scores"]["token_f1"] for r in rows)/507)<1e-10
            observations[stage]=rows;strata={}
            for group,inds in groups.items():
                rr=[rows[i] for i in inds];counts=collections.Counter(r["prediction"] for r in rr)
                strata[group]={"QA":len(rr),"Answer_F1":math.fsum(r["scores"]["token_f1"] for r in rr)/len(rr),
                    "predictions_contain_Han":sum(han(r["prediction"]) for r in rr),
                    "empty_predictions":sum(not r["prediction"].strip() for r in rr),
                    "literal_prefix_hits":sum(r["prediction"].lower().startswith(("the answer is","answer:")) for r in rr),
                    "raw_top6_predictions":counts.most_common(6)}
            semantic=[[r[k] for k in ["id","prediction","references","scores"]] for r in rows]
            cells.append({"stage":stage,"phase":"verified_complete_cell","scores":strata,
                "prediction_sha256":sha(p),"semantic_fields_sha256":hashlib.sha256(json.dumps(semantic,ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
        changes=[]
        for before,after,label in [(0,3,"prior_tasks"),(3,4,"current_Organ_acquisition"),(4,19,"post_acquisition_retention"),(0,19,"total_change")]:
            if before not in observations or after not in observations:continue
            a,b=observations[before],observations[after]
            entry={"before":before,"after":after,"label":label,"groups":{}}
            for group in ["all","q_lang=en","q_lang=zh","answer_type=OPEN","answer_type=CLOSED"]:
                inds=groups[group];delta=[b[i]["scores"]["token_f1"]-a[i]["scores"]["token_f1"] for i in inds]
                entry["groups"][group]={"delta_F1":paired(delta,[gold[i]["images"] for i in inds]),
                    "increased":sum(d>0 for d in delta),"decreased":sum(d<0 for d in delta),"unchanged":sum(d==0 for d in delta)}
            changes.append(entry)
        runs.append({"run":str(run.relative_to(ROOT)),"method":config["method"],"backbone":config["backbone"],
            "model_dtype":config["model_dtype"],"image_area_cap":config["image_area_cap"],"runtime_snapshot":runtime,
            "config_primary_score_legacy_field":config["primary_score"],"reported_metric":"locked raw Answer F1",
            "stage4_training_events":[{k:e[k] for k in ["stage","step","qa_seen","total_qa","loss","gradient_l2_preclip","learning_rate_used","warmup_updates"]}
                                      for e in events if e.get("event")=="train_update" and e["stage"]==4],
            "cells":cells,"transitions":changes})
    for p,h in sources.items():assert sha(ROOT/p)==h,"Sealed source changed while reading"
    for item in event_prefixes:
        assert hashlib.sha256((ROOT/item["path"]).read_bytes()[:item["bytes"]]).hexdigest()==item["sha256"]
    train=data["train"]
    return {"CPU_only":True,"host_label":host,"snapshot_unix":time.time(),"runs":runs,"source_sha256":sources,"training_event_prefixes":event_prefixes,
        "Organ_TRAIN_QA":len(train),"Organ_TRAIN_author_language_counts":dict(collections.Counter(r["metadata"].get("q_lang","not_annotated") for r in train)),
        "Organ_TEST_QA":len(gold),"all_author_strata_included":True,"stages_requested":STAGES,
        "limits":["Same frozen formal V2 runs; not nativeAnyRes/BF16 research states, never fill its shared S1.",
            "Pending cells remain null; a sealed task cell is not full21454 completion or a method ranking.",
            "S3->S4 isolates observed current-stage learning along this curriculum, not a no-update randomized intervention.",
            "S4->S19 combines within-medical and later cross-domain learning; cannot uniquely attribute domain count.",
            "Language groups are not verified translation pairs; Han counts are literal flags, not correctness.",
            "Paired image intervals condition on fixed single-seed fits; no multi-seed or causal knowledge sufficiency claim.",
            "Only raw token_f1 is reported; legacy rougeL config field does not change this locked scoring or select models."]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--host",required=True,choices=["old2080","new3090"]);p.add_argument("--run",action="append",required=True);args=p.parse_args()
    result=build(args.run,args.host);REPORT.mkdir(parents=True,exist_ok=True)
    path=REPORT/(args.host+"_snapshot.json")
    if path.exists():
        history=REPORT/"previous_CPU_snapshots";history.mkdir(exist_ok=True)
        (history/(str(time.time_ns())+"_"+path.name)).write_bytes(path.read_bytes())
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n");tmp.replace(path)
    print(json.dumps({"CPU_only":True,"runs":len(result["runs"]),"sealed_cells":sum(c["scores"] is not None for r in result["runs"] for c in r["cells"]),"report":str(path.relative_to(ROOT))}))


if __name__=="__main__":main()
