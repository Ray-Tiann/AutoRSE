# Hosted evaluation on Hugging Face

## Official live service

The official evaluator is [Ray0311/AutoRSE-Evaluator](https://huggingface.co/spaces/Ray0311/AutoRSE-Evaluator).
It uses free ZeroGPU for the fixed test metrics and accepts one complete six-task
prediction ZIP per evaluation. Download its signed receipt; collect three
independent complete trial sets for a formal submission. Demo receipts are
excluded from the community ranking. See the [submission guide](../leaderboard/SUBMIT.md)
for the ZIP/run evidence, review and automatic publication process.

The current deployed application source is public in the
[evaluator Space](https://huggingface.co/spaces/Ray0311/AutoRSE-Evaluator/tree/main).
The private evaluator remains private. The receipt verification public keys are
in [trusted-keys.json](../leaderboard/trusted-keys.json); no signing private key
belongs in GitHub or a participant submission.

The current hosted service is separate from the self-hosted CPU reference below.
The CPU template does not issue official signed community leaderboard receipts.
Public input hosting is a separate release step: the installer examples below
are not evidence that their example dataset repository has been populated.

## Self-hosted CPU reference setup

Participants run models/agents locally, select configurations using public validation
data, then upload **only the prediction file** to the AutoRSE Space. Hidden test
labels remain in a private Dataset owned by the maintainers. The public validator
checks formats; it does not compute a hidden-test score.

## Three separate repositories

| Repository | Visibility | Contents |
|---|---|---|
| `RayTian/AutoRSE` (Dataset) | Public | Reviewed public input ZIP, manifest, dataset card |
| `RayTian/AutoRSE-private` (Dataset) | **Private** | Twelve opaque evaluator files and `private-manifest.json` |
| `RayTian/AutoRSE` (Space, Docker) | Public or protected | Application source, no labels or tokens |

Dataset and Space may have the same name; they are different repository types.
These names are reference setup examples; they do not identify the live service above.

## 1. Upload public inputs

Upload the contents of the delivered `huggingface-dataset/` folder to the public
Dataset. Keep the ZIP filename unchanged; the public installer pins its SHA256.
Do not upload the old v1 bundle: its OneTrans event table and IntSR session/history
tables exposed held-out outcomes. See the v2 dataset card for the redaction scope.
The corrected public inputs are a new data revision, not a claim of reproducing
historical paper scores.

## 2. Provision private evaluator inputs

On a trusted maintainer machine which already holds the original evaluator files:

```bash
python tools/package_private_evaluator.py \
  --source-benchmark /trusted/path/to/benchmark \
  --intsr-identities /path/to/public-hf-folder/intsr-test-identity.npz \
  --private-output /trusted/private-evaluator-upload
```

This copies only fixed evaluator filenames and hashes their bytes without parsing
their values. IntSR's identity-only expected array comes from the reviewed v2
public inputs, so its candidate order matches the newly randomized test requests;
the original hidden positive-label file stays unchanged. The public identity NPZ
contains no targets and is published beside the data ZIP. Upload the generated
folder to the **private** Dataset. Record its exact
40-character commit ID and the printed manifest SHA256. Never include the folder
in the public Dataset, GitHub repository, or Space source tree.

## 3. Deploy Space source

```bash
python tools/prepare_space.py --output /tmp/autorse-space-upload
```

Create a Docker Space named `RayTian/AutoRSE`, then upload the **contents** of
the generated directory. Its root README enables HF OAuth and its root Dockerfile
starts the application. No GPU is needed for scoring prediction files.

Set these in **Space Settings → Variables and secrets**:

| Name | Kind | Value |
|---|---|---|
| `HF_TOKEN` | **Secret** | Fine-grained read token for the private Dataset only |
| `AUTORSE_PRIVATE_REPO` | Variable | `RayTian/AutoRSE-private` |
| `AUTORSE_PRIVATE_REVISION` | Variable | Exact private Dataset commit ID (40 hex characters) |
| `AUTORSE_PRIVATE_MANIFEST_SHA256` | Variable | SHA256 printed by the packager |
| `AUTORSE_STATE_DIR` | Variable, optional | `/data/autorse-state` when persistent storage is enabled |
| `AUTORSE_DAILY_LIMIT` | Variable, optional | Default `5` attempts per account/task/UTC day |

Without persistent storage the quota database resets when the Space restarts.
The source checks that the evaluator repository is private, pins its revision,
checks every private file's size/SHA and its pairing to the public data revision,
and blocks the private cache from Gradio's
file-serving endpoint. Submitted files are bounded, checked without pickle, and
scored in a fresh subprocess. The subprocess does not inherit the HF token.
Only aggregate metrics and the submission SHA are returned; errors do not expose
label values or internal file paths. Invalid submissions consume an attempt too.

After startup, sign in and test one known prediction per task against the trusted
original scorer before announcing the endpoint. Local synthetic tests cover all
six metric paths and malformed inputs, but cannot validate your HF secrets or
private Dataset upload. A numeric score is not a certification of experimental
protocol, source completeness, or full-duration research completion.

## Submission formats

| Task | File | Required contents | Main metric |
|---|---|---|---|
| HSTU / TIGER | `submission.npz` | `user_ids`, integer `top_items[N,10]` | NDCG@10 |
| IntSR | `submission.npz` | `sample_ids`, `task_ids`, unchanged `candidate_items[N,100]`, finite `scores[N,100]` | Mean rec/src NDCG@10 |
| WuKong | `submission.csv.gz` | Header `sample_id,score` | AUC |
| RankMixer | `submission.npz` | int64 `sample_ids`, probability `scores` | AUC |
| OneTrans | `submission.csv` | Header `sample_id,prediction` | Click AUC |

Keep public test IDs and row order unchanged. Upload predictions; never upload
checkpoints, Python code, executables, or pickled objects. Local public-validation
metrics are available from each baseline; official hidden metrics come from Space.

Primary deployment references: [HF Spaces OAuth](https://huggingface.co/docs/hub/spaces-oauth),
[HF Spaces configuration](https://huggingface.co/docs/hub/spaces-config-reference),
[Gradio file access](https://www.gradio.app/guides/file-access).
