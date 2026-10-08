# Experiment results

Each row is a separate attempt, not a multi-seed aggregate. Smoke runs are excluded.
Compare only runs with matching evaluation protocols, experimental settings and model fingerprints. Missing metrics are N/A.

| run | protocol | evaluation_protocol | model_id | precision | method | order | seed | epochs | status | AP | F.Rate | FWT | BWT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| superni_generation7_v2/migu_lora_first_order1_seed42_epoch1/20261006T112422Z | superni_generation7_v2 | cl_standard_fwt_v1 | meta-llama/Llama-2-7b-hf | nf4_fp16 | migu_lora | order_1 | 42 | 1 | stopped_model_switch | N/A | N/A | N/A | N/A |
| superni_generation7_v2/migu_lora_t5large_order1_seed42_epoch1/20261007T013544Z | superni_generation7_v2 | cl_standard_fwt_v1 | google-t5/t5-large | fp32_cuda | migu_lora | order_1 | 42 | 1 | completed | 23.0844 | 1.9441 | -2.8644 | 1.2129 |
| superni_generation7_v2/olora_first_order1_seed42_epoch1/20261006T092908Z | superni_generation7_v2 | legacy_sapt_fwt | meta-llama/Llama-2-7b-hf | nf4_fp16 | olora | order_1 | 42 | 1 | data_prepared | N/A | N/A | N/A | N/A |
| superni_generation7_v2/olora_first_order1_seed42_epoch1/20261006T092949Z | superni_generation7_v2 | legacy_sapt_fwt | meta-llama/Llama-2-7b-hf | nf4_fp16 | olora | order_1 | 42 | 1 | blocked_model | N/A | N/A | N/A | N/A |
| superni_generation7_v2/olora_first_order1_seed42_epoch1/20261006T104244Z | superni_generation7_v2 | legacy_sapt_fwt | meta-llama/Llama-2-7b-hf | nf4_fp16 | olora | order_1 | 42 | 1 | failed | N/A | N/A | N/A | N/A |
| superni_generation7_v2/olora_first_order1_seed42_epoch1/20261006T110916Z | superni_generation7_v2 | cl_standard_fwt_v1 | meta-llama/Llama-2-7b-hf | nf4_fp16 | olora | order_1 | 42 | 1 | stopped_model_switch | N/A | N/A | N/A | N/A |
| superni_generation7_v2/olora_t5large_order1_seed42_epoch1/20261007T013544Z | superni_generation7_v2 | cl_standard_fwt_v1 | google-t5/t5-large | fp32_cuda | olora | order_1 | 42 | 1 | completed | 21.9257 | 0.2969 | -3.8816 | 4.6029 |
| superni_generation7_v2/sapt_lora_first_order1_seed42_epoch1/20261006T112422Z | superni_generation7_v2 | cl_standard_fwt_v1 | meta-llama/Llama-2-7b-hf | nf4_fp16 | sapt_lora | order_1 | 42 | 1 | stopped_model_switch | N/A | N/A | N/A | N/A |
| superni_generation7_v2/sapt_lora_t5large_order1_seed42_epoch1/20261007T014202Z | superni_generation7_v2 | cl_standard_fwt_v1 | google-t5/t5-large | fp32_cuda | sapt_lora | order_1 | 42 | 1 | completed | 12.0302 | 0.8099 | -0.6518 | 0.9687 |
| superni_generation7_v2/seq_lora_first_order1_seed42_epoch1/20261006T152743Z | superni_generation7_v2 | cl_standard_fwt_v1 | meta-llama/Llama-2-7b-hf | nf4_fp16 | seq_lora | order_1 | 42 | 1 | stopped_model_switch | N/A | N/A | N/A | N/A |
| superni_generation7_v2/seq_lora_t5large_order1_seed42_epoch1/20261007T013544Z | superni_generation7_v2 | cl_standard_fwt_v1 | google-t5/t5-large | fp32_cuda | seq_lora | order_1 | 42 | 1 | completed | 25.6343 | 5.3524 | 2.3395 | -4.9429 |
