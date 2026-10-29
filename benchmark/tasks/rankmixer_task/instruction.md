# RANKMIXER evaluation task

Dataset: `criteo5m`. Hidden-test metric: AUC.

Retain RankMixer feature tokenization, token mixing, and per-token feed-forward networks. Preserve fixed splits, feature vocabularies, and row order. The public starter uses one visible GPU. Select by full public-validation AUC. Return int64 sample_ids and finite probability scores in submission.npz.

From this environment directory, run:

```bash
python3 -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 RESEARCH_SEED=2024 bash run_baseline.sh criteo5m
```

Public inputs belong in `datasets/criteo5m/input/`. The command writes `datasets/criteo5m/working/submission.npz`. Use public training and validation data for fitting and model selection. Submit predictions to the hosted evaluator for hidden-test scoring; hidden labels are not distributed.


Final selected source artifact: before ending, write `/workspace/artifact/`
containing the selected code, necessary helpers/configuration, `README.md` with
exact training and inference commands and public-validation selection, and
`SHA256SUMS`. Run `python3 export_artifact.py --command 'EXACT COMMANDS'
--validation 'PUBLIC VALIDATION RESULT AND RATIONALE'`, adding `--config` for
necessary text configuration and `--helper` for source helpers outside baseline. Use `--overwrite` to replace an
earlier selected artifact. Exclude weights, input data, caches, credentials, and
historical experiments. Harbor collects this directory separately from the
prediction file. The local verifier reports format only; hidden-test scores come
from the hosted evaluator.
