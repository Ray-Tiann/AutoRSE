# AutoRSE

Evaluating LLM Agents as Recommendation Engineers for Self-Evolving Recommender Systems.

This repository contains the minimal six-task runtime, editable recommendation
baselines, three agent-harness integrations, prediction-format checks, and a
Hugging Face scoring Space. Historical runs, checkpoints, paper knowledge bases,
API credentials, and hidden test labels are excluded.

**Run locally, score on Hugging Face.** Participants train and choose models with
public train/validation inputs, generate predictions, and submit those predictions
to the hosted evaluator. Local format checks never compute the hidden-test score.

| Task | Public dataset | Prediction file | Hosted metric |
|---|---|---|---|
| HSTU | Amazon Beauty | `submission.npz` | Full-catalog NDCG@10 |
| TIGER | Amazon Beauty | `submission.npz` | Full-catalog NDCG@10 |
| IntSR | Amazon Kindle | `submission.npz` | Mean recommendation/search NDCG@10 |
| WuKong | MovieLens Latest | `submission.csv.gz` | AUC |
| RankMixer | Criteo-5M | `submission.npz` | AUC |
| OneTrans | KuaiSAR v2 | `submission.csv` | Click AUC |

## Quick start

Use Linux with an NVIDIA GPU and Docker + NVIDIA Container Toolkit for the
container runner. Public data installation and file checks also work without a GPU.
Install Python 3.11+ on the host. Download the corrected public v2 input bundle
from the public Dataset **after the maintainer has uploaded it**:

```bash
git clone https://github.com/Ray-Tiann/AutoRSE.git
cd AutoRSE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/install_public_data.py --repo-id RayTian/AutoRSE --task hstu
```

To use the delivered archive before the HF upload:

```bash
python tools/install_public_data.py --archive /path/to/recagent-data-public-v2.zip --task hstu
```

The installer pins the whole archive SHA256, checks every selected file, and
refuses unsafe paths or overwriting different existing inputs.

Build the shared GPU image and run a baseline from scratch:

```bash
bash tools/build_images.sh
python tools/run_baseline.py --task hstu --gpu 0
python tools/validate_submission.py --task hstu \
  --submission benchmark/tasks/hstu_task/environment/datasets/beauty/working/submission.npz
```

Run `python tools/run_baseline.py --help` for the other tasks and direct local
training. Each task's `environment/README.md` describes its model, validation rule,
and output format. A format-valid example/oracle submission is not a trained model
or an official test score.

## Run an agent

Install the pinned Harbor host dependency and configure your own provider credentials.
The repository never supplies API keys. The public runner supports Claude Code,
Codex, and Terminus-2; public validation drives research and final model selection.

```bash
pip install -r requirements-harbor.txt
python tools/run_agent.py --help
```

See [runtime instructions](benchmark/RUNTIME.md) for exact commands, time budgets, and
what the public Harbor verifier checks. Agents must not use hidden evaluator
feedback to select models or ideas. Paper retrieval, external knowledge banks,
historical experiment cards, and previous results are not bundled.

## Submit for a hidden-test score

The intended scoring Space is [RayTian/AutoRSE](https://huggingface.co/spaces/RayTian/AutoRSE).
It becomes available after the owner deploys the provided Space source and private
evaluator inputs. Sign in, choose a task, upload the prediction file, and receive
aggregate metrics. Do not upload weights, source code, or training data.

Maintainers: [deploy the Space and private evaluator](docs/HOSTED_EVALUATION.md).
The public Dataset is [RayTian/AutoRSE](https://huggingface.co/datasets/RayTian/AutoRSE).
The Dataset and Space are separate repository types with the same name.

## Data revision and validation scope

The public **v2** release removes held-out outcomes exposed by the old OneTrans
event table and IntSR cross-domain history/session tables. IntSR public training
and validation rows at or beyond each user's first test timestamp are removed;
test IDs and candidate sets are retained, while IntSR candidate order is freshly
randomized to remove the old seed-based answer leak. Preserve the shipped v2
candidate order in predictions. Read the dataset card for exact
counts and audit checks. **Old paper scores do not establish results on v2.**

Release checks cover source syntax, task contracts, installer integrity, six
scoring paths using synthetic labels, upload validation, and app access boundaries.
Full six-task CUDA training and live HF deployment require the recipient's GPU,
provider configuration, and private evaluator setup; they are not claimed by the
offline checks.

```bash
python -m unittest discover -s tests -v
# Install space/requirements.txt for the optional app/OAuth boundary checks.
```

Code: Apache-2.0, with [third-party notices](THIRD_PARTY_NOTICES.md). Dataset terms
remain with the original sources; no blanket code license is applied to the data.
