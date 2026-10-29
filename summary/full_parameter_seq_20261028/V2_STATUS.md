# FullFT v2: LLaVA GPU preflight and independent reload passed; formal results missing

The existing terminal, rather than the earlier activation prose, is authoritative. `preflight_v2_once/COMPLETE.json` SHA256 `9d4be1fd21ec693c436cf1864132c1b514dffc07f70846fc7dd4706099b364c3` records actual preflight exit 0 (294.474035s, 19 original TRAIN QA) and a separate reload exit 0 (57.589630s, one already-used TRAIN QA). Both identities were absent at release; SLOT is complete/released/no_retry. Do not run the preflight again.

## What the successful preflight establishes

The LLaVA model has 893675552 unique parameters, with no LoRA or frozen visual feature cache. The preflight detected nonzero gradients and actual parameter changes in language, vision, projector and image-newline groups. Independent reload checked exact input and parameter digests, identical answer loss, and finite backward through all four groups. This is evidence for this LLaVA preflight's GPU/update/checkpoint path, not formal quality or universal memory feasibility.

Original assets remain at `exp/full_parameter_seq_20261028/v2/runs/onevision/pct01/preflight_once_20261028/`. Its checkpoint SHA256 is `325647cca57da556ddbf0dc8e7f13193dfbf1031ca48f096ccdf3a72ec3e194d`, 3582083716 bytes. `checkpoint_reload_proof.json` SHA256 is `def8a67e3b86377538a88f15adae57cfd0f34ce6fbd2012943ad30834ce9597e`. The checkpoint follows 19 diagnostic updates across tasks; despite its stage_01 filename it is NOT a formal S1 model and must not initialize a formal experiment.

## What remains missing

There are no formal FullFT TEST/F1, AP, FWT or BWT results. There is no Qwen preflight proof from this run. Four formal condition/backbone results remain missing.

The unchanged v2 runner has an explicit fresh formal path when `--preflight-only` is absent: original manifests, seed42, epoch1, batch16/micro1, LR1e-4, full-model FP32 parameters with BF16 AMP, native HostAdamW with CPU FP32 moments and GPU arithmetic, and per-stage complete checkpoints. S19 casts once to BF16 before evaluating and saving the same final graph. Training activation offload is explicit; deployment memory is not established by this preflight.

As audited on 2026-10-29, `summary/storage_recovery_second_20261029/owner_recovery.py` only offers the already-completed preflight/reload slot; it does not submit formal FullFT. The separate reviewed LLaVA 1% formal offer is now active in the same owner chain: `formal_pct01_once_20261029/ACTIVATION.json` records owner3338330/start62316253, original capacity-state adoption, and unchanged existing GPU workers. At activation no formal SLOT/model had started. GPU1/3 remain subject to original idle/held/other-UID guards. This is a queued formal offer, not a formal quality result; do not replay `activate_once.py`. The original failed v1 attempt, its failed_closed receipt, all diagnostics and the successful v2 assets remain retained.
