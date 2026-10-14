# Research context and current experiment

- The research setting is **small models, small datasets, lightweight continual learning**. This was explicitly confirmed by the user on 2026-10-14.
- Main backbone: Qwen2-VL-2B-Instruct. Five independent domains in this order: natural/general, medicine, geography/remote sensing, finance, art/culture. Each domain has text then image QA, for ten tasks.
- Current fixed budget is **100 training / no validation / 50 test QA per task**, one epoch, **seed 42 only**. Every method uses the same frozen sample IDs. Do not revert to 500 or 2000 training examples.
- Preserve original questions, answers, image assets and annotation semantics. Use the fixed source splits and source-stratified subsets. Do not add classification or fake-image tasks.
- **Do not save model, optimizer or recovery checkpoints for the current experiment.** Retain training logs, per-example predictions and references, stage/task/dataset score tables, full evaluation matrices and final figures.
- Evaluate all ten test tasks before training and after each completed training stage. Record normalized exact-match accuracy (including correct/total), token F1 and ROUGE-L; report native benchmark metric limitations. Derive AP, ForgetRate, FWT and BWT from complete 11×10 matrices.
- The eight selected paper comparisons are SMoLoRA, PASs-MoE, Hyper-LLaVA, PaRSP, OSFT, SpaRTA, FV-guided Training and ALER. Only the unchanged initial model is retained as a reference; SeqLoRA is excluded by user instruction. Exclude replay methods and historical-QA replay. DGG remains in the paper table but is excluded from this experiment.
- Preserve author mechanisms. Do not label downloaded code, unverified ports, or plain LoRA renamed after a paper as running methods. Report trainable parameters and auxiliary compute, particularly for intrinsically heavier full-tuning references.
- Run methods on idle GPUs; do not stop other users' jobs. Current code and reports are under exp/ten_task_cl_20261012 and summary/ten_task_cl_20261012. Current protocol: rules/012_small_data_results_only.md; execution policy: exp/ten_task_cl_20261012/execution_policy.json.
- The authoritative Git checkout is /data1/liyapeng/program/LLMCL on the new server (liyapeng@100.87.141.22, SSH port 8666; LAN 10.254.25.46). The local Windows workspace mirrors documents and scripts. Preserve unrelated remote changes when committing.

Later explicit user instructions override this context.

- Latest domain revision (2026-10-14): user rejected CUAD PDF-page visual QA and TAT-DQA document visual QA. Law domain may be replaced, prioritizing geography/remote sensing; art remains. Verify both text and genuine image tasks and multiple published uses. Training is disabled until these replacements are verified. Keep old GeoChat raw data available while evaluating candidates; do not bulk-transfer it yet.

- Dataset revision verified on 2026-10-14: T5 GeoQuestions1089 count/population numeric QA; T6 original GeoChat open QA; T8 MME-Finance PC candlestick/indicator/statistic open QA. All 100/0/50. These are explicit research holdouts, not author official splits. FinChart-Bench not selected due insufficient published reuse. Data audit: data/DATASET_VERIFICATION_20261014.md. Training remains disabled for remaining method adapters and new-task GPU preflight, not dataset verification. Old initial cache is incompatible with revised tasks; do not reuse it.
