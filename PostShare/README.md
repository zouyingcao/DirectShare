## Take Notes:

please ``cd LLaMA-Factory/src/llmtuner/train/pt/workflow.py`` to set the specific training settings before using LLaMA-Factory repository for PostShare.

| Name                     | Meaning                                           | Example        |
| -------------------------| ------------------------------------------------- | -------------- |
| "weight_loss"            | the method choosen for pushing the weights closer | norm           |
| "match_indexs"           | about candidate attention head pairs              | 1e-4saved_npy/llama-2-7b-hf-qk/match_qk_indexs_llama-2-7b-hf_cos.npy    |
| "similarity_values"      | about matching function values (cosine similarity)| saved_npy/llama-2-7b-hf-qk/match_qk_values_llama-2-7b-hf_cos.npy        |
| "share_rate"             | the ratio for weight sharing                      | 0.30           |
