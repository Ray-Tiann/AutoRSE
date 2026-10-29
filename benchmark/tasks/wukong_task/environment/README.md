# WUKONG evaluation task

Dataset: `movielenslatest`. Hidden-test metric: AUC.

Retain WuKong FMB/LCB feature interactions. Fit feature processors only on training rows and preserve their artifacts with the checkpoint. Select by full public-validation AUC. Return sample_id,score in a gzip CSV in public test order.

From this environment directory, run:

```bash
python3 -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 RESEARCH_SEED=2024 bash run_baseline.sh movielenslatest
```

Public inputs belong in `datasets/movielenslatest/input/`. The command writes `datasets/movielenslatest/working/submission.csv.gz`. Use public training and validation data for fitting and model selection. Submit predictions to the hosted evaluator for hidden-test scoring; hidden labels are not distributed.
