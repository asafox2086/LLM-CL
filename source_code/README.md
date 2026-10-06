# Source archive provenance

The original project `code/` directory was renamed to `source_code/` without changing its contents.

During implementation, `O-LoRA-src/README.md` was found to describe **Online-LoRA: Task-free Online Continual Learning via Low Rank Adaptation** (WACV 2025), a vision method. It does not correspond to the language-model O-LoRA PDF in `Paper/`. The original archive is retained for traceability.

The correct language-model implementation is now archived at `O-LoRA-language-src/`:

- Repository: https://github.com/cmnfriend/O-LoRA
- Commit: `07117e1fc4a5f5ad9308a815a42cee8f46502dc8`
- Paper: Orthogonal Subspace Learning for Language Model Continual Learning.
- Retrieved: 2026-10-06; all 426 upstream blob hashes verified against that commit.
- Upstream generated `logs_and_outputs/` and `logs_and_outputs_llama/` are retained locally but ignored by Git to avoid committing generated predictions. Source, configurations and reference data retain their original contents.

The unified implementation is `../code/olora.py`; execution settings and differences from the original experiments are documented in `../rules/001_experiment_protocol.md`, section 13.
