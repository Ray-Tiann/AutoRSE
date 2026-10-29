# Task contract

Retain TIGER semantic-ID generative retrieval and the supplied Beauty item mapping. The starter uses a pure-PyTorch T5 and supplied semantic IDs; it does not download pretrained weights. Fit only on training sequences. Select by full public-validation NDCG@10. Append validation interactions only for final test inference. Return public test user_ids and ten distinct, valid, unseen top_items per user.

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
