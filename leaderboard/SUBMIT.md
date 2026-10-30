# Submit to the AutoRSE leaderboard

We welcome new agents, harnesses and recommendation research workflows. Submit a reproducible six-task result for maintainer review; approved results are published automatically to the [Hugging Face leaderboard](https://huggingface.co/spaces/Ray0311/AutoRSE-Leaderboard) and embedded on the project website.

## 1. Run the complete benchmark

Evaluate the same versioned agent configuration on **HSTU, IntSR, TIGER, OneTrans, RankMixer and WuKong**, with three independent trials for every task. This means **18 research runs**. Each run must use the fixed **12-hour research window and one GPU**, with no resource or timeout overrides. Finalization may take at most 300 additional seconds. Record three distinct seeds per task and preserve the complete execution evidence.

Use the public data revision and bundle pinned in [`autorse_leaderboard/protocol.py`](autorse_leaderboard/protocol.py). The submission template includes these values. Keep the agent, harness, model versions and framework commit explicit. A protocol/data update requires a separate comparable cohort; results from different releases must not be pooled with the historical paper table.

For each independent trial, submit the six-task prediction ZIP to the official evaluator and download the **signed, complete six-task receipt**. Repeat for all three trials, producing three different evaluation IDs. Every receipt must belong to the same participant account. A receipt binds scores to prediction hashes; it does not certify how the agent produced those predictions.

Demo/synthetic submissions and receipts marked for testing are useful for checking the evaluator. They are **never eligible for the public ranking**, even if a test succeeds. Do not rename a test receipt or manufacture benchmark evidence.

## 2. Publish the run evidence

Upload each run's evidence to a **public, ungated Hugging Face dataset repository**. Supply a full 40-character commit SHA, relative path, SHA256 and byte size for every referenced file. Branch names such as `main` are not immutable revisions. Each file must be nonempty and at most 160 MiB.

Each of the 18 run records requires these five artifacts:

| Artifact | Required content |
| --- | --- |
| `code` | The evaluated implementation, such as a source archive, pinned to the recorded code commit. |
| `report` | Readable UTF-8 text describing the run, changes and results. |
| `trajectory` | UTF-8 JSON containing a nonempty array of structured steps, messages or events. |
| `config` | UTF-8 JSON with the actual command arguments, model, harness, seed, one-GPU allocation and fixed budget. |
| `provenance` | UTF-8 JSON binding the evaluated code commit to the exact prediction SHA256 in the signed receipt. |

A minimal configuration record has this structure; replace the model, harness, seed and command with the actual values:

```json
{
  "harness": "YOUR-HARNESS",
  "model": "YOUR-MODEL",
  "seed": 2024,
  "research_budget_seconds": 43200,
  "minimum_runtime_sec": 43200,
  "finalization_grace_sec": 300,
  "gpus": 1,
  "command": ["YOUR-ACTUAL-EXECUTABLE", "YOUR-ACTUAL-ARGUMENT"]
}
```

The provenance record must include `evaluated_code_commit` and `prediction.sha256`. If it also includes a `run` object, its `run_id`, `seed` and `code_commit` must agree with the submission. For example:

```json
{
  "evaluated_code_commit": "FULL_40_CHARACTER_CODE_COMMIT",
  "prediction": {"sha256": "FULL_64_CHARACTER_PREDICTION_SHA256"},
  "run": {"run_id": "hstu-trial-1", "seed": 2024}
}
```

Use hashes of the actual uploaded bytes. After uploading evidence, this helper creates a hash/size reference for one file (it does not upload the file):

```bash
PYTHONPATH=leaderboard python -m autorse_leaderboard.cli artifact-ref \
  --file /path/to/local/report.md \
  --repo-id YOUR-HF-USERNAME/YOUR-EVIDENCE-DATASET \
  --revision FULL_40_CHARACTER_DATASET_COMMIT \
  --path hstu-trial-1/report.md --output /tmp/report-ref.json
```

Use hashes of the actual uploaded bytes. The same evidence file can be referenced by multiple records only when it truthfully covers those runs; generic placeholder logs do not demonstrate independent trials. Keep the dataset public after submission. Evidence is fetched again on every publication. If any approved entry's evidence becomes unavailable or fails integrity checks, publication stops and the last published snapshot remains visible. The leaderboard mirrors submission records and the review ledger, **not a benchmark-owned copy of all evidence binaries**.

## 3. Prepare and validate the submission JSON

From the root of your AutoRSE checkout, use Python 3.12:

```bash
python -m venv .venv-leaderboard
source .venv-leaderboard/bin/activate
python -m pip install -r leaderboard/requirements.txt
PYTHONPATH=leaderboard python -m autorse_leaderboard.cli template \
  --output /tmp/autorse-submission-metadata.json
```

Edit the generated metadata. Set `id` to a unique lowercase slug of at most 80 characters, with single hyphens separating alphanumeric segments, such as `my-agent-model-v1`. Fill the configuration and all 18 run records. The `prepare` command fills `receipt_evaluation_id` and `prediction_sha256` from the corresponding signed receipt (receipt order is trial 1, 2, 3); conflicting nonempty metadata is rejected. Each trial index must use one complete receipt for its six tasks; use all three receipts exactly once this way. Record timezone-aware start/finish timestamps and the actual code commit. Read the two acknowledgements and set them to `true` only when accurate.

Combine the metadata and the three receipts, then validate the public evidence:

```bash
PYTHONPATH=leaderboard python -m autorse_leaderboard.cli prepare \
  --metadata /tmp/autorse-submission-metadata.json \
  --receipts /path/to/trial-1-receipt.json /path/to/trial-2-receipt.json /path/to/trial-3-receipt.json \
  --output /tmp/my-agent-model-v1.json

PYTHONPATH=leaderboard python -m autorse_leaderboard.cli validate \
  --submission /tmp/my-agent-model-v1.json \
  --keys leaderboard/trusted-keys.json \
  --resolve-artifacts \
  --report /tmp/my-agent-model-v1-validation.json
```

A successful command exits with status 0. Inspect any failed checks in the report and correct the underlying evidence. Metadata-only validation cannot authorize publication. Passing automatic validation makes the result eligible for human review; it does not prove real GPU use, trial independence, originality or compliance with hidden-label access rules.

## 4. Open one submission pull request

Fork [Ray-Tiann/AutoRSE](https://github.com/Ray-Tiann/AutoRSE) and create a branch. Add exactly one new file:

```text
leaderboard/submissions/my-agent-model-v1.json
```

The filename stem must exactly match `id`. The file must be at most 2 MiB. **The submission PR may only add this one JSON file**: no code, documentation, key changes, edits to existing submissions or renamed files. Propose maintenance changes separately. Keep source archives and execution artifacts in the public dataset rather than adding them to this PR.

Open the PR against `main`, describe your configuration and link the public evidence dataset. The CI reads the JSON as data using the trusted base-branch validator; it never runs participant code. If you change the submission after review, request approval of the new head commit.

## 5. Review, merge and publication

A maintainer checks reproducibility, whether the recorded protocol was actually followed, full trajectories, prediction/code provenance and possible test-label leakage or score manipulation. Automated checks validate signatures, coverage, declarations and artifact integrity; the reviewer remains responsible for assessing the experiment.

Publication requires an actual GitHub `APPROVED` review on the current PR head from a **different human account** with current repository `write`, `maintain` or `admin` permission. Self-approval, bot approvals, stale or dismissed approvals and a JSON field claiming approval do not qualify. Maintainers merge only after successful CI and review.

After merge, the publisher verifies the merged bytes and review evidence again, revalidates all approved submissions and uploads a new leaderboard snapshot through a main-branch-only trusted publisher. A direct push or an unreviewed submission does not create an eligible row. The website receives the updated Hugging Face data; no separate website edit is required for each result.

For each task, the leaderboard averages the three signed scores. Higher scores rank first. Scores rounded to four decimals with decimal half-up are tied and receive midranks. The final order uses **mean rank across all six tasks**, with lower being better; equal rank sums share an overall competition rank. Filters do not recalculate ranks. Historical paper results remain separate from the reviewed public-data cohort.
