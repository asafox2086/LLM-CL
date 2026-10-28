# 2026-10-28 NEW storage interruption

At unix 1793196687, workers 8265 / 2840219 / 2154888 were absent from /proc. Their actual supervisor logs record ENOSPC or PytorchStreamWriter write failure at unix 1793196523–6533. Original queue JSON was stale. The sole capacity owner PID 122374 also exited; its systemd unit is failed, exit 1.

OLD TaskGradReuse worker 2187244/start782053901 remains live, completed S14 and evaluating S15. OLD three running baselines retain their actual identities. This incident is not a research result or a model failure.

Cleanup executed once: completed legacy Seq run S01–S08 payloads only, 35,438,364,080 bytes retired. Final S19 was fully SHA-verified, final stat and all non-weight file hashes remained unchanged. These retired early weights are intentionally not backed up; all existing main-table records, answers, final weights and current recovery checkpoints remain.

Recovery being prepared: Qwen OSFT from committed S02; LLaVA SMoLoRA from committed S13. Failed S03/S14 output remains in original runs and must not contaminate the resumed canonical prefix. Old Qwen SMoLoRA has no checkpoint and requires a fresh run, not checkpoint resumption. Two workers have now been launched via the original pipeline under its seven locks: OSFT PID 782824/start54065154 (GPU2) and LLaVA SMoLoRA PID 782837/start54065162 (GPU1). Original capacity unit restarted as PID783300 and adopted both; actual native state reached S3/S14 training. Original cudaMallocAsync environment was explicitly preserved. Recovery must use the sole original capacity owner and preserve GPU0.

Preparation provenance note: loaded source was 29bb1d41, but its file was edited to 59489 while it ran, so INTENT self-file SHA does not identify loaded code. Exact old source and independent SOURCE_VERSION_NOTE are retained, with original receipts unchanged. Resource logs were copied in a separate audited step; they span processes and are not a single-process memory-growth measurement.

FullFT preflight made its first actual attempt on GPU3, then failed before model creation: run.py AST lookup for pause_at_boundary raised StopIteration. This is an implementation/preflight failure, not evidence that full fine-tuning cannot fit. Original failed attempt remains immutable; no automatic retry.

Qwen SMo has been explicitly requeued fresh (not resumed); independent review found 22 computation sources byte-identical and main training AST identical across the three I/O file changes. Original owner is now PID789874 after CPU-only requeue handoff. Its normal queue launched Qwen PAS on GPU3 as PID790980/start54097728; Qwen SMo waits. GPU0 is not assigned. Archive service789877 and original seven-baseline observer789879 are active again; seven-baseline observer does not by itself verify checkpoint-frame branches. No full new F1 or deployment measurement was produced by recovery.
