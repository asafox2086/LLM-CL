# Continual fine-tuning benchmark data

## Full Super-NaturalInstructions (SuperNI)

- `SuperNI_full/`: complete official repository snapshot, including `tasks/`, upstream splits, documentation and licenses; ignored by Git via the root `.gitignore`.
- Source: https://github.com/allenai/natural-instructions
- Downloaded on 2026-10-06; upstream revision: `55a365637381ce7f3748fa2eac7aef1a113bbb82`.
- The snapshot contains 1,613 task JSON files. The user has selected seven generation tasks, with 1000 train / 200 dev / 500 test instances per task, drawn directly from this complete snapshot.
- Download metadata and integrity results are stored locally as `upstream_commit.json`, `upstream_tree.json`, and `verification.json` inside `SuperNI_full/`. Verification compares file contents against the official Git blob hashes and parses every task JSON.
- Upstream task-level splits differ from this project's instance-level continual-learning splits. The deterministic algorithm and verified fingerprints are documented in [the experiment protocol](../rules/001_experiment_protocol.md) and [data manifest](../rules/001_data_manifest.json). SAPT's instance splits are reference material only.

## Existing paper benchmark subsets

- `SAPT_CL_Benchmark/Long_Sequence/`: Long Sequence Benchmark task data (15 tasks).
- `SAPT_CL_Benchmark/SuperNI/`: SuperNI/SuperNatural Instructions task data used by SAPT.
- `Long_Sequence_configs/` and `SuperNI_configs/`: task split/configuration files.

These files are copied from the official SAPT repository:
https://github.com/circle-hit/SAPT
- `MAC_datasets/`: StreamingQA test set and SQuAD test set/configs from the official MAC repository. The MAC README links the full StreamingQA download; `archivalqa.yaml` is included as its loader configuration.
