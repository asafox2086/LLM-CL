# Six-domain multimodal continual fine-tuning

This experiment supersedes the five-domain MME-Finance run. The sixth domain is traffic/autonomous driving. See [the current rule](../../rules/016_six_multimodal_finmme_coda_20261016.md) and [source and reuse evidence](../../data/MULTIMODAL_SIX_DOMAIN_AUDIT_20261016.md).

Sequence: TextVQA → SLAKE → RSVQA-LR → FinMME → AQUA → CODA-LM. The author English CODA-LM Train/Test configuration is used unchanged; regional red-box images are author-supplied. FinMME uses a disclosed image-group research holdout, with two missing-gold records audited separately. Shared fixed IDs/order/test hashes for all methods and both conditions.

Qwen2-VL-2B-Instruct; seed42; one epoch; no validation. GPU0 reserved, GPU3 full group, GPU1/2 nested1% group. Initial and all six post-task evaluations cover every full test set: 7×6 matrices. Never reuse old predictions. Preserve all native question types. Uniform lexical scores do not reproduce FinMME FinScore or CODA-LM LLM judging.

`prepare_sources.py` downloads pinned author Parquet files with direct HTTPS, validates original LFS SHA256, and records source metadata. `prepare_data.py` verifies exact counts, decodes original image bytes, creates immutable task manifests and refreshes the root README. `pipeline.py` waits for complete sources, prepares manifests and requires a fresh real six-task backward preflight per method/condition before formal training. `run.py` performs the instrumented training/evaluation and exports final full model/method state once.

No stage, optimizer or recovery checkpoint. Retain final parameters after stage6, per-example predictions, complete matrices, loss, real pre/postclip gradient norms and concrete CPU/GPU before/after/peak measurements. Final parameter serialization occurs after the training-after memory snapshot. Final files require custom method reconstruction and fresh-process inference-equivalence verification before later evaluation.

As inherited from the existing implementation, training target token cap is512; decoding caps are configured in `runtime_utils.evaluate`. Original long answers remain intact in the manifests and predictions' references. CODA-LM has long descriptions; truncation/budget effects must be disclosed and considered when interpreting lexical scores. Every method and both conditions use identical budgets.

Source files, decoded images, predictions and parameters stay on the server and are excluded from Git. Author source metadata, checksums, fixed manifest metadata and documentation are committed.
