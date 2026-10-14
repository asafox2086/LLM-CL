import json,csv,io,decimal,random,collections,hashlib,zipfile,shutil,time,re
from pathlib import Path
from PIL import Image
R=Path(__file__).resolve().parents[2];D=R/"data";B=D/"TenTaskCL/equal_budget_100_0_50_geo_fin_seed42_20261014";B.mkdir(exist_ok=True)
old=D/"TenTaskCL/equal_budget_100_0_50_seed42/manifest.json";previous=json.loads(old.read_text());newrows={};sampling={}
# Published GeoBenchmark Appendix B evaluates count/population free numeric QA.
g=D/"GeoQuestions1089";questions=json.loads((g/"raw/GeoQuestions1089.json").read_text());answers=json.loads((g/"raw/GeoQuestions1089_answers.json").read_text());pool=[]
for key,q in questions.items():
 if int(key)>1017:continue
 v=answers[key]
 if not isinstance(v,str) or len(v)>10000:continue
 a=list(csv.reader(io.StringIO(v)))
 if len(a)!=2 or len(a[1])!=1:continue
 try:n=decimal.Decimal(a[1][0]);assert n.is_finite()
 except Exception:continue
 lower=q["Question"].lower()
 if "how many" not in lower and "population" not in lower:continue
 pool.append({"id":"GeoQuestions1089:"+key,"source_id":key,"dataset":"GeoQuestions1089","question":q["Question"],"answer":a[1][0],"answers":[a[1][0]],"context":"","images":[],"metadata":{"source_question_id":key,"source_category":q["Category"],"original_answer_csv":v,"answer_serialization":"original single CSV value; no numerical conversion","subset_basis":"GeoBenchmark LREC2026 Appendix B count/population QA"},"source_split":"published_evaluation_pool_research_holdout"})
assert len(pool)>=150
rng=random.Random(42);rng.shuffle(pool);newrows[5]={"train":pool[:100],"test":pool[100:150]};sampling["T05"]={"available_pool":len(pool),"counts_question_how_many":sum("how many" in x["question"].lower() for x in pool),"counts_question_population":sum("population" in x["question"].lower() for x in pool),"paper_reported_counts":{"count":143,"population":28},"exact_author_ids_reproduced":False,"release_note":"Actual original release yields 175 union QA, paper reports 171 total; named subset criteria match but exact author IDs unavailable. No count forced, no question/answer changed.","split":"seed42 question-ID holdout; this research, not author official split"}
# Selected GeoChat QA retains original answer, image and source turn.
g=D/"GeoChat_Instruct";s=json.loads((g/"selected_original_records.json").read_text());newrows[6]={"train":[],"test":[]}
for x in s:
 record=x["original"];turn=x["retained_turn"];human=record["conversations"][2*turn]["value"];answer=record["conversations"][2*turn+1]["value"];question=human.replace("<image>","").strip()
 assert not any(mark in question for mark in ["[identify]","[refer]","{<"])
 assert not any(mark in question.lower() for mark in ["scene category","classify","land-use classification"])
 row={"id":"GeoChat_Instruct:"+str(x["source_index"])+":"+str(turn),"source_id":str(x["source_index"])+":"+str(turn),"dataset":"GeoChat_Instruct","question":question,"answer":answer,"answers":[answer],"context":"","images":[str((g/x["image_file"]).relative_to(R))],"source_split":"published_instruction_pool_research_holdout","metadata":{"original_source_index":x["source_index"],"original_turn":turn,"original_human_value":human,"original_image":record["image"],"archive_member":x["archive_member"],"annotation_origin":"original GeoChat mixture including synthetic conversations","input_serialization":"remove structural <image> marker and edge whitespace only"}}
 newrows[6][x["split"]].append(row)
sampling["T06"]={"split":"seed42 one original QA per image, image-disjoint 100/50; research holdout from published instruction mixture","official_split":False,"raw_annotations_unchanged":True,"semantic_scan":"all 150 retained turns are counts/spatial descriptions/natural questions; explicit classification/referring task markers absent","upstream_unsupported_candidate_rows_excluded":22}
# MME-Finance official chart categories, PC style, native open QA capabilities.
f=D/"MME_Finance";original=list(csv.DictReader((f/"raw/MMfin.tsv").open(),delimiter="\t"));allowed_types={"candlestick","indicator","statistic"};allowed_tasks={"Image Caption","Spatial Awareness","Accurate Numerical Calculation","Numerical Calculation","Investment Advice","Explain Reason","Risk Warning"}
pool=[x for x in original if x["image_type"] in allowed_types and x["task_category"] in allowed_tasks and x["image_style"]=="PC"]
assert len(pool)==153
images=f/"raw/selected_images";images.mkdir(exist_ok=True);image_sha={}
with zipfile.ZipFile(f/"raw/MMfin.zip") as z:
 for path in sorted({x["image_path"] for x in pool}):
  assert ".." not in Path(path).parts
  p=images/path;p.parent.mkdir(parents=True,exist_ok=True);blob=z.read("MMfin/"+path);p.write_bytes(blob)
  with Image.open(p) as im:im.verify()
  image_sha[path]=hashlib.sha256(blob).hexdigest()
# Group all variants/questions of the same chart; merge any byte-identical charts too.
groups=collections.defaultdict(list)
for x in pool:groups[str(Path(x["image_path"]).parent)].append(x)
parent={k:k for k in groups}
def find(k):
 while parent[k]!=k:k=parent[k]
 return k
seen={}
for k,rows in groups.items():
 for row in rows:
  h=image_sha[row["image_path"]]
  if h in seen:parent[find(k)]=find(seen[h])
  else:seen[h]=k
merged=collections.defaultdict(list)
for k,rows in groups.items():merged[find(k)].extend(rows)
keys=sorted(merged);random.Random(42).shuffle(keys);dp={0:[]}
for k in keys:
 for total,chosen in list(dp.items()):
  n=total+len(merged[k])
  if n<=100 and n not in dp:dp[n]=chosen+[k]
assert 100 in dp,"No chart-group-disjoint 100/50 allocation possible"
train_keys=set(dp[100]);train=[x for k in keys if k in train_keys for x in merged[k]];test=[x for k in keys if k not in train_keys for x in merged[k]];assert len(train)==100 and len(test)>=50
rng=random.Random(42);rng.shuffle(train);rng.shuffle(test);test=test[:50];newrows[8]={"train":[],"test":[]}
for split,rows in [("train",train),("test",test)]:
 for x in rows:
  row={"id":"MME_Finance:"+x["index"],"source_id":x["index"],"dataset":"MME_Finance","question":x["question"],"answer":x["answer"],"answers":[x["answer"]],"context":x["background"],"images":[str((images/x["image_path"]).relative_to(R))],"source_split":"official_english_evaluation_pool_research_holdout","metadata":{"source_index":x["index"],"original_image_path":x["image_path"],"image_type":x["image_type"],"image_style":x["image_style"],"task_category":x["task_category"],"chart_group":str(Path(x["image_path"]).parent),"image_sha256":image_sha[x["image_path"]]}}
  newrows[8][split].append(row)
sampling["T08"]={"official_source_rows":1171,"eligible_pool":153,"image_types":sorted(allowed_types),"excluded_image_types":["document","table","mixed"],"image_style":"PC","capabilities":sorted(allowed_tasks),"split":"seed42 shuffled chart groups, deterministic whole-group 100-QA allocation, 50 test QA from remaining chart groups; research split, not official","train_chart_groups":len(train_keys),"test_chart_groups":len({x["metadata"]["chart_group"] for x in newrows[8]["test"]}),"image_sha_overlap":0,"original_qa_modified":False}
# Construct new manifest; retain the other seven tasks byte-for-byte.
manifest={"protocol":"Published-task subsets; original annotations/assets unchanged; explicit research holdouts for new evaluation-only/instruction pools","sampling_seed":42,"seed":42,"counts_per_task":{"train":100,"test":50},"validation_enabled":False,"formal_ready":True,"assets_pending":[],"qa_role_overlap":0,"previous_manifest_sha256":hashlib.sha256(old.read_bytes()).hexdigest(),"dataset_revision":"geo_fin_20261014","initial_evaluation_cache":None,"sampling_details":sampling,"image_overlap_note":"New T6/T8 have no chart-group/image-hash cross-role overlap. Existing medical tasks retain native official QA splits, which may share images, as previously documented.","tasks":[]}
instructions={5:"Answer the geographic question. Give the numeric answer directly.",6:"Answer the question using the remote-sensing image. Give the answer directly.",8:"Answer the financial question using the chart image. Give the answer directly."}
audit={"status":"passed","new_tasks":{},"other_seven_tasks_unchanged":True,"raw_annotations_unchanged":True,"training_started":False}
for oldtask in previous["tasks"]:
 num=oldtask["task"];task={k:oldtask[k] for k in ["task","task_id","domain","modality","instruction"]};task["files"]={};task["sample_ids"]={}
 if num in [5,6]:task["domain"]="geography"
 if num in instructions:task["instruction"]=instructions[num]
 for split in ["train","test"]:
  if num in newrows:
   rows=newrows[num][split];p=B/f"T{num:02d}_{split}.jsonl";p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows));info={"path":str(p.relative_to(R)),"count":len(rows),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"dataset_counts":dict(collections.Counter(x["dataset"] for x in rows)),"selected_qa_ids":[x["id"] for x in rows]}
  else:
   info=oldtask["files"][split].copy();p=R/info["path"];assert hashlib.sha256(p.read_bytes()).hexdigest()==info["sha256"];rows=[json.loads(line) for line in p.read_text().splitlines()]
  assert len(rows)==(100 if split=="train" else 50)
  for x in rows:
   for name in x.get("images",[]):assert (R/name).is_file()
  task["files"][split]=info;task["sample_ids"][split]=[x["id"] for x in rows]
 assert not (set(task["sample_ids"]["train"])&set(task["sample_ids"]["test"]))
 if num in newrows:
  if num in [6,8]:
   sets=[{hashlib.sha256((R/p).read_bytes()).hexdigest() for row in newrows[num][split] for p in row["images"]} for split in ["train","test"]];assert not sets[0]&sets[1]
  audit["new_tasks"][str(num)]={"train":100,"test":50,"qa_id_overlap":0,"image_hash_overlap":0 if num in [6,8] else None}
 manifest["tasks"].append(task)
(B/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n");(D/"TenTaskCL/current_small_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n");(D/"TenTaskCL/dataset_revision_verification_20261014.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
p=R/"exp/ten_task_cl_20261012/execution_policy.json";policy=json.loads(p.read_text());policy.update(manifest=str((B/"manifest.json").relative_to(R)),experiment_budget_id="100_0_50_seed42_geo_fin_20261014",reuse_initial_base_predictions_only=False,datasets_verified=True,training_enabled=False,pending_dataset_revision={},training_block_reason="datasets verified; remaining paper method adapters and new-task GPU preflight still pending");p.write_text(json.dumps(policy,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({"status":"datasets_verified","new_manifest":policy["manifest"],"geo_text_pool":len(newrows[5]["train"])+len(newrows[5]["test"]),"sampling":sampling},ensure_ascii=False,indent=2))
