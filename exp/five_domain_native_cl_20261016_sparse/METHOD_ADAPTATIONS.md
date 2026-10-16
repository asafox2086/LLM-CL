# New baseline revision, 2026-10-16

Remove Hyper-LLaVA from the experimental queue; retain its bibliography entry. Add ER-LoRA and CL-MoE, for ten baselines. Existing frozen datasets, task order and 1%/test hashes remain unchanged.

## CL-MoE

CVPR2025 main paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.pdf

Author source: https://github.com/ECNU-ICALK/CL-MoE , revision a9eff1c35a0cea13aa48e7233f5bbc8dd7b09471. Author expert classes are extracted from the SHA256-verified source. This is an explicitly adapted Qwen2-VL reproduction, not an exact LLaVA checkpoint reproduction.

- Paper defaults: eight experts, total rank64 (eight per expert), alpha128, topK2, beta0.5, gamma0.7, AdamW lr2e-4. Language q/k/v/o and gate/up/down projections, frozen base/vision, dropout0.05. Paper defaults take precedence over the example shell script's four-expert rank32 setting.
- Train with the instance router. At stage end, one additional no-grad pass over every current TRAIN example computes mean token probabilities per example, then equal-example task routing weights (equation6). Top2 identifies task-related experts; preserve their union across stages.
- Momentum follows equations10–12: a current-only expert keeps0.3 old/0.7 new; shared and other experts keep0.7 old/0.3 new. First task has no prior trained expert snapshot to merge. Router parameters remain current; averaging applies to experts.
- Task inference uses nearest TRAIN feature centroid, never ground-truth evaluation task IDs. Qwen has no encoder CLS: replace that feature with the last question-token state of the frozen Qwen language decoder, with adapters temporarily disabled. Inputs contain question text only, no answers or dataset/task labels. This is a backbone-dependent feature adaptation. Questions are tokenized to512 tokens for the auxiliary index; original training/evaluation inputs are unchanged. Task centroids are computed before momentum from the whole current train set. Evaluation uses beta0.5 local/global router mixing. No anchors at initial evaluation means instance-only routing; zero-initialized LoRA B makes the untrained reference unchanged.
- Released source has placeholder filesystem paths, timing-dependent router logs, commented inference mixing, and an external merge script whose prefix matching misses complete parameter keys. This implementation replaces these with deterministic in-memory statistics, the published equations, and layer/expert identities. These changes are disclosed rather than claiming an unchanged upstream trainer.
- Per-stage auxiliary files record real train counts, chosen experts and feature definition. Current-stage expert snapshots, task weights, seen expert sets, centroids and full model graph are retained in the final checkpoint. Resource monitoring covers all extra passes and snapshots.

## ER-LoRA

Standard experience replay, also compared in CL-MoE section5, not a newly proposed CVPR2025 algorithm. Memory capacity5000 follows that paper; q/v rank8 alpha32 dropout0.1 and 1:1 current/replay loss are explicit common-backbone protocol adaptations. Only previous stages' frozen training rows are eligible. The same deterministic seed42 bottom-hash reservoir and hash-index replay draw are used on both hosts. With1% training, the actual memory may be smaller than capacity; record its actual size and IDs. Test data are never accessed by replay. Current new-data traversal and its hash are unchanged. Replay adds compute/sample exposures, reported separately, without pretending to use the same total examples as non-replay controls.

## Gates and execution

Every method must pass19 real2B task backward/optimizer checks, full-graph serialization, and an independent-process reload/output/backward check. CL-MoE additionally exercises task statistics and momentum on two diagnostic stage subsets; those diagnostic weights are discarded. ER preflight exercises past-sample replay after its first diagnostic task. No proof is inferred from code downloads or compilation alone.

Both hosts run identical Python3.10.21 and all55 installed package distribution versions, source files and policy hashes. SMoLoRA uses its author BF16 configuration on3090; old2080Ti does not support native BF16 and is excluded from that method's ownership. New GPU0/1 remain reserved; newGPU2 handles1%, GPU3 full; oldGPU0–3 handle1%. SeqFT saves every stage; all other methods save final only.
