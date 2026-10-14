# Research context and current experiment

- The research setting is **small models, small datasets, lightweight continual learning**. This was explicitly confirmed by the user on 2026-10-14.
- Main backbone: Qwen2-VL-2B-Instruct. Five independent domains in this order: natural/general, medicine, law, finance, art/culture. Each domain has text then image QA, for ten tasks.
- Current fixed budget is **100 training / 50 validation / 50 test QA per task**, one epoch, **seed 42 only**. Every method uses the same frozen sample IDs. Do not revert to 500 or 2000 training examples.
- Preserve original questions, answers, image assets and annotation semantics. Use the fixed source splits and source-stratified subsets. Do not add classification or fake-image tasks.
- **Do not save model, optimizer or recovery checkpoints for the current experiment.** Retain training logs, per-example predictions and references, stage/task/dataset score tables, full evaluation matrices and final figures.
- Evaluate all ten test tasks before training and after each completed training stage. Record normalized exact-match accuracy (including correct/total), token F1 and ROUGE-L; report native benchmark metric limitations. Derive AP, ForgetRate, FWT and BWT from complete 11×10 matrices.
- The eight selected paper comparisons are SMoLoRA, PASs-MoE, Hyper-LLaVA, PaRSP, OSFT, SpaRTA, FV-guided Training and ALER. Frozen and SeqLoRA are extra controls. Exclude replay methods and historical-QA replay. DGG remains in the paper table but is excluded from this experiment.
- Preserve author mechanisms. Do not label downloaded code, unverified ports, or plain LoRA renamed after a paper as running methods. Report trainable parameters and auxiliary compute, particularly for intrinsically heavier full-tuning references.
- Run methods on idle GPUs; do not stop other users' jobs. Current code and reports are under exp/ten_task_cl_20261012 and summary/ten_task_cl_20261012. Current protocol: rules/012_small_data_results_only.md; execution policy: exp/ten_task_cl_20261012/execution_policy.json.
- The authoritative Git checkout is /data2/liyapeng_grp/program/LLMCL on the SSH server. The local Windows workspace mirrors documents and scripts. Preserve unrelated remote changes when committing.

Later explicit user instructions override this context.
