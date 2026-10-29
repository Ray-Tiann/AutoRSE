# AutoRSE

Evaluating LLM Agents as Recommendation Engineers for Self-Evolving Recommender Systems.

AutoRSE evaluates recommendation research performed by an LLM agent. A participant
runs a model–harness combination on a task, uses **public validation data** to
choose the final model, and uploads its **test predictions** to the hosted evaluator.
Hidden test labels stay on the evaluator server.

**Evaluation workflow:** download public inputs → run the agent → select by public
validation → export predictions and source → check prediction format → upload
predictions to Hugging Face → receive aggregate test metrics.

The repository includes six editable baselines, Claude Code / Codex / Terminus-2
integrations, a local format validator, and the Hugging Face scoring application.
It excludes hidden labels, API credentials, trained checkpoints, historical runs,
paper knowledge bases, and experiment cards.

## Evaluation links and availability

| Service | Address | Purpose |
|---|---|---|
| Public Dataset | [RayTian/AutoRSE](https://huggingface.co/datasets/RayTian/AutoRSE) | Download the checksum-pinned public v2 inputs |
| Scoring Space | [RayTian/AutoRSE](https://huggingface.co/spaces/RayTian/AutoRSE) | Sign in and upload predictions for hidden-test scoring |

**Deployment requirement:** the maintainer must upload the public data and deploy
and configure the scoring Space before participants can complete this workflow.
The code release alone does not activate either service. If the Dataset is
inaccessible or the Space says its evaluator is not configured, contact the
maintainer; local format checks cannot substitute for an official score.
Maintainer setup is documented in [Hosted evaluation](docs/HOSTED_EVALUATION.md).
Participants do **not** need access to the private evaluator Dataset or its token.

## 1. Prepare the runtime

For GPU experiments, use **Linux**, an NVIDIA GPU, a CUDA-12.1-compatible driver,
Docker, and NVIDIA Container Toolkit. Use **Python 3.12+** on the host; the optional
agent runner is pinned to Harbor 0.18.0. Docker Desktop on macOS cannot run this
NVIDIA GPU protocol. Data installation and format validation do not require a GPU.

```bash
git clone https://github.com/Ray-Tiann/AutoRSE.git
cd AutoRSE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash tools/build_images.sh
```

The shared image installs the task training dependencies and agent CLIs. GPU
memory needs vary by task and model configuration. The Harbor task files currently
request one visible GPU, 8 CPUs, `memory_mb=65536`, and `storage_mb=102400` for the
agent environment; the separate format verifier requests 2 CPUs and 4096 MiB.
These are the public starter settings, not a claim of matching historical runs.

## 2. Download the public data

Install one task, for example HSTU:

```bash
python tools/install_public_data.py --repo-id RayTian/AutoRSE --task hstu
```

Install all six tasks by omitting `--task`:

```bash
python tools/install_public_data.py --repo-id RayTian/AutoRSE
```

If the maintainer supplied the archive directly, install it locally instead:

```bash
python tools/install_public_data.py \
  --archive /path/to/recagent-data-public-v2.zip --task hstu
```

The installer pins the complete archive SHA256, verifies each installed file,
and refuses unsafe paths or replacement of different existing files. A selective
install still downloads the complete ZIP, then extracts only the selected tasks.
All task inputs together need approximately 2.1 GB; training outputs need additional
space. Use a new checkout if old v1 inputs conflict. See the
[dataset card](docs/DATASET.md) for split counts, v2 changes, and upstream data terms.

## 3. Run the experiment

### Evaluate an LLM agent

Install the host orchestration dependency:

```bash
python -m pip install -r requirements-harbor.txt
```

Configure your own provider credentials in a private environment file **outside
this repository**. The variable names and model identifier must match the chosen
Harbor harness/provider. Replace `YOUR_MODEL_ID` and the environment-file path
below with your actual configuration; the example does not include credentials.
A baseline-only run, described below, does not need model API access.

First inspect the launch command without starting an agent, using a GPU, or
calling an API:

```bash
python tools/run_agent.py --task hstu --agent codex \
  --model YOUR_MODEL_ID --gpu 0 --seed 2024 --dry-run
```

Then launch one HSTU research trial:

```bash
python tools/run_agent.py \
  --task hstu \
  --agent codex \
  --model YOUR_MODEL_ID \
  --api-env-file "$HOME/.config/autorse/provider.env" \
  --gpu 0 --seed 2024 \
  --jobs-dir ./runs --job-name hstu-codex-s2024
```

- `--task`: `hstu`, `tiger`, `intsr`, `wukong`, `rankmixer`, or `onetrans`.
- `--agent`: `claude-code`, `codex`, or `terminus-2`. Configure a compatible model
  and provider credential file for each harness.
- The default research budget is **43,200 seconds (12 hours)**, followed by at most
  **300 seconds** for finalization. Early completion resumes the same native session.
- Each invocation runs one task and one attempt. Use a new `--job-name` for each
  task, model, harness, and seed; run jobs sequentially when sharing one GPU.
- Short infrastructure checks may use `--minimum-runtime-sec 60
  --finalization-grace-sec 30`; label these as smoke runs, not 12-hour evaluations.

Use only public training inputs for fitting and public validation for selection.
Do not use hidden-test feedback, external answer sources, knowledge banks, or
historical experiment cards to select models or ideas. The supplied agent
containers have public networking for API access; these data-access rules are
not enforced by complete network isolation. Details, provider configuration
references, and runtime constraints are in [Runtime instructions](benchmark/RUNTIME.md).

### Optional: run a baseline without an LLM agent

To check the training/prediction pipeline or establish a baseline, run one of the
following commands after installing its task data. Each command trains from scratch
and writes a test prediction file; it does **not** obtain a hidden-test score.

```bash
python tools/run_baseline.py --task hstu      --gpu 0 --seed 2024
python tools/run_baseline.py --task tiger     --gpu 0 --seed 2024
python tools/run_baseline.py --task intsr     --gpu 0 --seed 2024
python tools/run_baseline.py --task wukong    --gpu 0 --seed 2024
python tools/run_baseline.py --task rankmixer --gpu 0 --seed 2024
python tools/run_baseline.py --task onetrans  --gpu 0 --seed 2024
```

Run these separately or sequentially on a shared GPU. For a host-managed CUDA
installation, install the selected task's `environment/requirements.txt` and add
`--local`. Each task's `environment/README.md` explains its validation rule.
A constant/example submission is useful for format checks, but is not a trained
baseline or a research result.

## 4. Locate and validate the final predictions

For a **baseline run**, outputs remain under
`benchmark/tasks/<task>_task/environment/` at the relative paths in this table.
For an **agent run**, Harbor collects predictions under
`runs/<job-name>/<trial>/artifacts/workspace/` at the same relative paths.

| Task | Relative prediction path | Test rows | Required contents | Main hosted metric |
|---|---|---:|---|---|
| HSTU | `datasets/beauty/working/submission.npz` | 22,363 | `user_ids[N]`, integer `top_items[N,10]` | `ndcg_at_10` |
| TIGER | `datasets/beauty/working/submission.npz` | 22,363 | `user_ids[N]`, integer `top_items[N,10]` | `ndcg_at_10` |
| IntSR | `datasets/Amazon/working/submission.npz` | 133,391 | `sample_ids[N]`, `task_ids[N]`, `candidate_items[N,100]`, finite `scores[N,100]` | `macro_ndcg_at_10` |
| WuKong | `datasets/movielenslatest/working/submission.csv.gz` | 200,686 | CSV header `sample_id,score` | `auc` |
| RankMixer | `datasets/criteo5m/working/submission.npz` | 500,000 | int64 `sample_ids[N]`, probability `scores[N]` | `auc` |
| OneTrans | `datasets/kuaisar_v2/working/submission.csv` | 1,460,572 | CSV header `sample_id,prediction` | `auc` |

Keep test IDs and row order unchanged. HSTU/TIGER require ten distinct catalog
items per user, ranked best first. IntSR uses `task_ids` 0 for recommendation and
1 for search; preserve the **v2 candidate identities and order** in every row.
CTR predictions must be finite probabilities in `[0,1]`.
NPZ submissions must contain only the listed named arrays. CSV headers must match
the listed names and order; WuKong's `.csv.gz` must actually be gzip-compressed.

For example, check a baseline HSTU prediction:

```bash
python tools/validate_submission.py --task hstu \
  --submission benchmark/tasks/hstu_task/environment/datasets/beauty/working/submission.npz
```

For an agent trial, find the collected prediction, then pass that exact path to
the same validator. Do not assume the editable baseline directory contains the
agent's final output:

```bash
find runs/hstu-codex-s2024 -type f -path '*/artifacts/workspace/datasets/*/working/submission.*'
python tools/validate_submission.py --task hstu \
  --submission /path/to/the/collected/submission.npz
```

Agents must also write their selected code, necessary helpers/configuration,
`README.md`, and `SHA256SUMS` to `/workspace/artifact/`. Harbor collects this under
the trial's `artifacts/workspace/artifact/` directory. See
[the source export command](benchmark/RUNTIME.md) for `export_artifact.py` usage.
Keep this source artifact for reproducibility; the scoring Space accepts the
**prediction file**, not the source folder or a checkpoint.

**Local validation is not official evaluation.** `format_valid=true` or Harbor
`format_valid=1` means only that the prediction format passed. It does not check
all hidden identities, compute NDCG/AUC, or establish that the research completed
successfully. Review trial logs and artifact collection before submitting.

## 5. Upload to Hugging Face and read the score

1. Open the [AutoRSE scoring Space](https://huggingface.co/spaces/RayTian/AutoRSE).
2. Sign in using the **Hugging Face login** button.
3. Select the task corresponding to your prediction file.
4. Upload the single `.npz`, `.csv`, or `.csv.gz` file from step 4, then click
   **Submit for scoring**. Do not upload data, weights, executables, source code,
   or the outer data/source ZIP.
5. Read the returned JSON. `metric` names the main metric; its value is in
   `metrics[metric]`. Keep `task`, `data_revision`, `public_bundle_sha256`,
   `submission_sha256`, and the returned metrics with your experiment record.

All six main metrics are **higher is better**. IntSR's main score is the unweighted
mean of recommendation and search NDCG@10; the two components are also returned.
CTR tasks return AUC with auxiliary logloss (lower is better); OneTrans also
returns user AUC. Hidden labels and per-example correctness are never returned.

The current application defaults to **160 MiB per file** and **five attempts per
account/task/UTC day**; invalid attempts count. Check formats before uploading.
A deployment may change the daily limit. If the Space is unavailable, unconfigured,
or reports a scoring error, there is no completed official score for that attempt;
do not substitute the local format reward or treat the error as a model score of 0.
For ID/order errors, verify the task, v2 data revision, row count, and candidate order.

## Reporting a comparable result

Report each task's hosted score together with the exact model/backend identifier,
harness and CLI versions, source commit, container image digest, public-data
SHA/revision, seed, GPU/resources, research and
finalization budgets, public-validation selection, prediction SHA, and selected
source artifact. Keep resource budgets and v2 inputs consistent across comparisons;
report tasks and repeated seeds separately. A hosted numeric score alone does not
certify source completeness or compliance with the research protocol.

The public **v2** inputs remove held-out outcomes exposed by the old OneTrans and
IntSR tables and repair IntSR's predictable candidate ordering. IntSR public split
sizes and history context changed, and test candidate order was randomized.
**Historical paper scores are not v2 re-evaluations.** Details and source data
terms are in the [dataset card](docs/DATASET.md).

## Maintainers and validation scope

To deploy the Dataset/Space and configure the separate private evaluator, follow
[Hosted evaluation](docs/HOSTED_EVALUATION.md). Never upload private scoring inputs
to this GitHub repository, the public Dataset, or the Space source tree.

Release checks cover source syntax, task contracts, checksum-verified data
installation, six scorer paths with synthetic labels, upload boundaries, baseline
CLI imports, and small CPU model checks. Full six-task CUDA training and live HF
end-to-end deployment are not established by those offline checks.

```bash
python -m unittest discover -s tests -v
# Install requirements-harbor.txt and space/requirements.txt for optional checks.
```

Code: Apache-2.0, with [third-party notices](THIRD_PARTY_NOTICES.md). Dataset terms
remain with their original sources; the code license does not relicense the data.
