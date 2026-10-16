# Launch report

Checked at 2026-10-16T16:05:25.142470+08:00. This is a launch snapshot, not final experimental results.

Ten baselines; Hyper-LLaVA excluded, ER-LoRA and CVPR2025 CL-MoE added. New GPU0/1 reserved; old four GPUs and newGPU2/3 assigned to baselines. All methods share pre-frozen1% IDs and full test hashes.

| Condition | Method | Host | GPU | Queue phase | Actual worker phase |
|---|---|---|---:|---|---|
| full | SeqFT_LoRA | new3090 | 3 | training | initial_evaluation |
| full | ER_LoRA | new3090 | - | waiting | - |
| full | CL_MoE | new3090 | - | waiting | - |
| full | SMoLoRA | new3090 | - | waiting | - |
| full | PASs_MoE | new3090 | - | waiting | - |
| full | PaRSP | new3090 | - | waiting | - |
| full | OSFT | new3090 | - | waiting | - |
| full | SpaRTA | new3090 | - | waiting | - |
| full | FV_guided_Training | new3090 | - | waiting | - |
| full | ALER | new3090 | - | waiting | - |
| pct01 | CL_MoE | new3090 | 2 | training | initial_evaluation |
| pct01 | SMoLoRA | new3090 | - | waiting | - |
| pct01 | OSFT | new3090 | - | waiting | - |
| pct01 | SpaRTA | new3090 | - | waiting | - |
| pct01 | ALER | new3090 | - | waiting | - |
| pct01 | SeqFT_LoRA | old2080 | 0 | training | see old host live state |
| pct01 | ER_LoRA | old2080 | 1 | training | see old host live state |
| pct01 | PASs_MoE | old2080 | 2 | training | see old host live state |
| pct01 | PaRSP | old2080 | 3 | training | see old host live state |
| pct01 | FV_guided_Training | old2080 | - | waiting | see old host live state |

CL-MoE passed19 native2B optimizer/backward cases and fresh-process reload/finite-backward gate. Diagnostic task centroids and momentum were exercised before serialization. Diagnostic weights were discarded; formal workers start from base. CPU checks verified dual-router equation9 and all three expert momentum branches.

No formal training stage is claimed complete in this launch report. Queue phase training means the formal runner process, which first evaluates the unchanged initial model on all19 full tests.

Live progress: [progress.md](progress.md). Runtime/code/environment identity: [two_host_runtime_integrity.json](two_host_runtime_integrity.json). Exact CL-MoE/ER adaptations: [METHOD_ADAPTATIONS.md](../../exp/five_domain_native_cl_20261016_sparse/METHOD_ADAPTATIONS.md).
