# TIGER evaluation task

Dataset: `beauty`. Hidden-test metric: NDCG@10.

Retain TIGER semantic-ID generative retrieval and the supplied Beauty item mapping. The starter uses a pure-PyTorch T5 and supplied semantic IDs; it does not download pretrained weights. Fit only on training sequences. Select by full public-validation NDCG@10. Append validation interactions only for final test inference. Return public test user_ids and ten distinct, valid, unseen top_items per user.

From this environment directory, run:

```bash
python3 -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 RESEARCH_SEED=2024 bash run_baseline.sh beauty
```

Public inputs belong in `datasets/beauty/input/`. The command writes `datasets/beauty/working/submission.npz`. Use public training and validation data for fitting and model selection. Submit predictions to the hosted evaluator for hidden-test scoring; hidden labels are not distributed.
