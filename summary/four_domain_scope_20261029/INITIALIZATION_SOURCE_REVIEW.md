# Four-domain initialization and S16 reuse boundary

This is a source-only review, not acceptance of any reused checkpoint. Two new input manifests retain exactly original native T01–T16; no QA, prediction, checkpoint payload or score was read for this review. Existing five-domain sources/results remain unchanged. The independent scope protocol belongs to root.

## Dependence on future task count/instructions

| Method | Actual source behavior | Consequence for an old five-domain S16 state |
|---|---|---|
| Seq-LoRA | `methods.Method` SeqLoRA ignores the instruction values, fixed q/v-r8; no task-count bank found. | Potential model-prefix reuse only; this is not a complete runner/RNG/evaluation equivalence proof. |
| SMoLoRA | `methods.py:118–128` encodes **all** supplied instructions jointly using MiniLM; stores the resulting table; `AuthorSMoLinear` keeps that table and uses task index. Eight experts/rank128 do not themselves shrink with 19→16. | Removing future instructions changes the initialization input/table and possibly padded encoding arithmetic. Do not uniformly certify the existing 19-row method object as a four-domain initialization. |
| SpaRTA | `Router` allocates six banks with `8 * task_count` rows; constructor receives `len(instructions)` and later Gram–Schmidt operates on those banks. | 19→16 changes persistent shapes and random draws, potentially all earlier learning. An old 19-task S16 state is not the four-domain method merely by deleting rows or relabeling. |
| CL-MoE | Constructor does not consume `instructions`; eight expert mechanism, anchors/task weights are collected from actually visited TRAIN stages. `NativeIO` still holds the whole supplied manifest/data. | No future instruction/count initialization dependency found in this adapter. Possible model-prefix candidate, but old full method/context and complete state equivalence are not certified. |
| PAS | Constructor ignores `instructions`; initial expert_num=2/rank128; learner receives stages as visited and retains importance/reference state. | No total-19 allocation found in this adapter. Possible causal model prefix, not blanket authorization to reuse the old full learner/context or old quality. |
| PaRSP | Constructor ignores instructions; stage-local localization/covariance and inherited history use current/past rows through `NativeIO`. | No total-19 initialization dependency found here. Must retain localization/history, not just base weights, and bind the four-domain context if reuse is later accepted. |
| OSFT | Initialization is matrix-shape SVD with ratio .25, instructions unused; refactor depends on visited stage. | No total-19 allocation found. Possible model-prefix candidate; function/state/deployment and RNG provenance remain separate. |
| FOREVER | Uses SeqLoRA initialization, but 1% `bind_context` asserts 19 tasks / 1313 QA; 10% asserts 19 and its preflight view also assumes 19. Full TRAIN task list is retained. | Existing runner cannot directly consume the new manifest. Future input/context removal and helper/count adaptation must be explicit; old full method object is not automatically a legal four-domain final state. This is not a claim that replay mathematics needs to change. |

## Why S16 is not automatically a four-domain final result

The native evaluator's old S16 evaluates only tasks 16 and 17. It does **not** contain R16,t for all T01–T16. A four-domain final table needs one legitimate S16 final graph evaluated on all 19,331 TEST rows; neither old S19 four-column slicing nor mixing S16 finance with other S19 domains satisfies this.

Even for count-independent methods, the old run performed initial evaluation of all 19 tasks and carried its old method/context/cache/RNG. Source inspection that a constructor ignores `instructions` does not prove every earlier execution state is identical. Reuse, if later accepted, must separately establish the relevant model/method-state prefix, permitted inputs and final graph. Missing S16 payload/state cannot be filled by an S19 checkpoint. No universal reuse is authorized by this document.

Current TaskGradReuse has stage-19 final fold/cleanup semantics; four-domain finalization must occur at S16 and be evaluated on that actual deployed graph. New FullFT `core.py` also explicitly finalizes/casts at 19; its FP32 S16 intermediate graph is not the BF16 four-domain final graph. Mathematical fold/cast availability does not transfer old quality to the resulting graph.

## Input lineage

`pct01_manifest.json`: 1108 TRAIN / 19331 TEST / 78 planned main updates.
`pct10_manifest.json`: 11055 TRAIN / 19331 TEST / 698 planned main updates.
Both preserve full task objects, IDs, original paths/file SHA, instruction, caps, order SHA and seed42. The inherited `formal_ready` describes already-frozen input metadata, not runner or GPU acceptance. `MANIFEST_SOURCE_RECEIPT.json` binds both original manifests unchanged and both new manifests; original source-pool details remain parent lineage, not a claim of new sampling.

Under the inherited sparse-cell convention, a future 16-task experiment has 62 cells / 70,091 score rows. Those are planned counts only; no evaluation or new F1 was performed.

## Sources actually reviewed

- `exp/two_backbone_native_checkpoint_20261027/methods.py` SHA `e5bd8f51690b006274ed5adcfa4ef3b1a2c3b0f1d79476d948b71d1cfec41c5b`
- `exp/two_backbone_native_checkpoint_20261027/clmoe_adapter.py` SHA `33946eddfc6424638ded0675ded68860273859c2f71993a6a60bd22556341be9`
- `exp/two_backbone_native_checkpoint_20261027/pass_adapter.py` SHA `c8e23c85a93e97130d464b4b6d87110470035c5af37f9575fce705923c80c7c4`
- `exp/two_backbone_native_checkpoint_20261027/parsp_adapter.py` SHA `d1ebe1831e6d089d434bdde34b59db64eff17dc74d8d77f9ca4f080a51da8366`
- `exp/two_backbone_native_checkpoint_20261027/osft_adapter.py` SHA `60b05f3b59f3878e65ee9543657b54a5aa5da13c14e139123e1c3591f3007b34`
- `exp/two_backbone_native_checkpoint_20261027/sparta_adapter.py` SHA `8a00505b40663e5a233ddceb34e295d9b7b73d55f0679ac3d2b93318e77d7a44`
- `exp/two_backbone_native_checkpoint_20261027/run.py` SHA `ea347e655c8f3a20d07c86568b351ab1615868bcf9b88d209756c45d1cef1280`
- `exp/two_backbone_native_checkpoint_20261027/sparse_results.py` SHA `07a7808beff8bc3330e92dc5b0fb5443401486846db764c912a5a0f57558fa3f`
- `exp/two_backbone_pct01_forever_checkpoint_20261027/forever_adapter.py` SHA `a9fd1fc2750117b94dfc712392ef616dca9cdb586b738e0b28ec5408097f710d`
- `exp/two_backbone_pct10_forever_checkpoint_20261027/forever_adapter.py` SHA `68a911e46c1e6db9d8da8125e9243f6480032e98bbbdc1f8cd9e49db9c5327cb`
- `exp/full_parameter_seq_20261028/v2/core.py` SHA `67ec0cc0ba660201b24f842556f8291afa4153d1fee17ee1ccb86f7255a7a2d4`
