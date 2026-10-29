# Task contract

Retain unified OneTrans sequence/non-sequence processing and pyramid compression. Use strictly past-only histories. The public starter uses one visible GPU and selects by click AUC on the fixed first 204800 public validation rows. Test click/like zeros are loader placeholders. Return sample_id,prediction CSV in public test order with finite probabilities.

Write the declared prediction artifact and submit it to the hosted evaluator.


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
