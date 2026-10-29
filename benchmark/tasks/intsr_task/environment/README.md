# INTSR evaluation task

Dataset: `Amazon`. Hidden-test metric: macro NDCG@10 over recommendation and search.

Retain IntSR query-driven behavior and dual-scenario fusion, including session masks. Use the supplied frozen precomputed Qwen features. Preserve each query’s 100 candidate item identities and their order. Select by the unweighted mean of recommendation and search validation NDCG@10. Return sample_ids, task_ids (0=rec, 1=src), candidate_items, and finite scores.

From this environment directory, run:

```bash
python3 -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 RESEARCH_SEED=2024 bash run_baseline.sh Amazon
```

Public inputs belong in `datasets/Amazon/input/`. The command writes `datasets/Amazon/working/submission.npz`. Use public training and validation data for fitting and model selection. Submit predictions to the hosted evaluator for hidden-test scoring; hidden labels are not distributed.
