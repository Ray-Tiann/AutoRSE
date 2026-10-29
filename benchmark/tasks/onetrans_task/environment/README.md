# ONETRANS evaluation task

Dataset: `kuaisar_v2`. Hidden-test metric: click AUC.

Retain unified OneTrans sequence/non-sequence processing and pyramid compression. Use strictly past-only histories. The public starter uses one visible GPU and selects by click AUC on the fixed first 204800 public validation rows. Test click/like zeros are loader placeholders. Return sample_id,prediction CSV in public test order with finite probabilities.

From this environment directory, run:

```bash
python3 -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 RESEARCH_SEED=2024 bash run_baseline.sh kuaisar_v2
```

Public inputs belong in `datasets/kuaisar_v2/input/`. The command writes `datasets/kuaisar_v2/working/submission.csv`. Use public training and validation data for fitting and model selection. Submit predictions to the hosted evaluator for hidden-test scoring; hidden labels are not distributed.
