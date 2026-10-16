# Five datasets, 19 native tasks, sparse evaluation

User-confirmed revision: remove AQUA, remove Hyper-LLaVA, add ER-LoRA and CL-MoE (ten baselines). Rule: [017](../../rules/017_five_domain_native_sparse_two_hosts.md).

Backbone Qwen2-VL-2B-Instruct; one epoch; seed42; no validation. Full training131019 QA; nested1%1313 QA; identical full test21454 QA.

`prepare_native.py` partitions existing frozen parent records without rewriting them or resampling. `freeze_splits.py` writes immutable manifest/ID/file SHA256 ledgers. These scripts must run once before any formal worker. Neither host nor method may regenerate them. Native labels do not imply an author-defined task order; the explicit order is preregistered for this study.

TextVQA1 + SLAKE10 content classes + RSVQA-LR4 question tasks + FinMME1 + CODA-LM3 tasks =19 stages. FinMME single/multiple/numerical formats and SLAKE OPEN/CLOSED are reporting dimensions, not additional overlapping stages.

Initial evaluation covers all19 tasks. Intermediate stages evaluate current and next tasks only. Final stage evaluates all19. This is74 test groups/80316 predictions, not a full20×19 matrix. Absent cells are null. AP/FWT/BWT use required cells only; historical-peak ForgetRate is unavailable. Domain final results are QA-weighted within each domain; the native-task AP is equal-weight over19 tasks, and the five-domain macro is reported separately.

SeqFT-LoRA saves complete method graph at all19 stages in both conditions. Every other method saves only its final stage19 graph. The same final checkpoint is reused for downstream evaluation, not duplicated in a second artifact. Processor/config/RNG/SHA256 accompany checkpoints. Stage boundaries use newly created optimizers. `--resume` is enabled only for SeqFT with a matching committed checkpoint. Other interrupted workers require fresh runs.

Preflight checks every native task on the real2B model, then serializes learned method weights. A separate process must reload the graph and reproduce a deterministic answer loss plus finite backward gradients before formal training can start. Diagnostic weights are discarded after verification. Native method mechanisms are retained; Hyper-LLaVA is excluded; CL-MoE retains eight experts with paper dual routing and momentum (see METHOD_ADAPTATIONS.md), and SpaRTA has19 task slots×8 components, adapting capacity to19 stages rather than silently stopping at the inherited ten-task capacity.

`LLMCL_HOST=new3090`: full GPU3, 1% GPU2; GPU0/1 reserved. `LLMCL_HOST=old2080`: only1% on GPUs0–3. Disjoint method ownership is in `execution_policy.json`. Both user services use `pipeline.py`; status/figures/reports are under `summary/five_domain_native_cl_20261016_sparse/`. Old completed results are collected over LAN and their final checkpoint hashes checked. Runtime logs/checkpoints stay out of Git.

Raw EM/Token F1/ROUGE-L do not replace native benchmark scoring, particularly FinMME numerical tolerance and CODA-LM judge scores. No claim that external judge scores have already been produced.
