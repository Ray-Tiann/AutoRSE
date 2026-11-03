# Run the recommendation baselines

Train the supplied models and write test predictions without launching an LLM
agent. Run every host command below from the **repository root**.

For agent experiments, start with the [root quickstart](../../README.md) or the
[custom-agent guide](../custom_agents/README.md). The
[benchmark runtime guide](../README.md) covers resources, Git experiments and
source export.

## 1. Prepare the runtime

Use Python 3.12 on the host, Linux, an NVIDIA GPU, a CUDA-12.1-compatible driver,
Docker and NVIDIA Container Toolkit. Docker Desktop on macOS cannot run this
NVIDIA GPU workflow.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash tools/build_images.sh
```

The shared image supplies the training dependencies. Baseline commands do not
invoke Harbor and need no model API credentials.

## 2. Install the public data

Install all six tasks:

```bash
python tools/install_public_data.py --repo-id Ray0311/AutoRSE
```

Add `--task hstu` to install only HSTU, or use a maintainer-provided v2 archive:

```bash
python tools/install_public_data.py --archive /path/to/recagent-data-public-v2.zip
```

The installer verifies the pinned archive and file checksums. See the
[dataset card](../../docs/DATASET.md) for input layouts, revisions and data terms.

## 3. Run a baseline

Choose one command, or run them sequentially when sharing GPU 0. Each trains from
scratch and writes the task's test prediction.

```bash
python tools/run_baseline.py --task hstu      --gpu 0 --seed 2024
python tools/run_baseline.py --task tiger     --gpu 0 --seed 2024
python tools/run_baseline.py --task intsr     --gpu 0 --seed 2024
python tools/run_baseline.py --task wukong    --gpu 0 --seed 2024
python tools/run_baseline.py --task rankmixer --gpu 0 --seed 2024
python tools/run_baseline.py --task onetrans  --gpu 0 --seed 2024
```

Add `--dry-run` to inspect a launch without Docker, training, data or API access.
Pass task-specific training options after `--`; each linked task guide defines
its architecture, validation rule and prediction contract.

## 4. Use an existing host CUDA environment

For a compatible host CUDA/PyTorch installation, install the selected task's
requirements and add `--local`. For example, for HSTU:

```bash
python -m pip install -r benchmark/tasks/hstu_task/environment/requirements.txt
python tools/run_baseline.py --task hstu --gpu 0 --seed 2024 --local
```

Use the corresponding `environment/requirements.txt` for another task. Record
your actual dependency versions when they differ from the shared image.

## 5. Locate and check the predictions

Baseline outputs persist in the selected task's `environment/datasets/` directory
on the host. They are separate from Harbor agent outputs under `runs/`.
All prediction paths below are relative to the repository root.

| Task instructions | Public-validation selection | Test prediction |
| --- | --- | --- |
| [HSTU](hstu_task/environment/README.md) | Full validation NDCG@10 | `benchmark/tasks/hstu_task/environment/datasets/beauty/working/submission.npz` |
| [TIGER](tiger_task/environment/README.md) | Full validation NDCG@10 | `benchmark/tasks/tiger_task/environment/datasets/beauty/working/submission.npz` |
| [IntSR](intsr_task/environment/README.md) | Unweighted mean of recommendation and search NDCG@10 | `benchmark/tasks/intsr_task/environment/datasets/Amazon/working/submission.npz` |
| [WuKong](wukong_task/environment/README.md) | Full validation AUC | `benchmark/tasks/wukong_task/environment/datasets/movielenslatest/working/submission.csv.gz` |
| [RankMixer](rankmixer_task/environment/README.md) | Full validation AUC | `benchmark/tasks/rankmixer_task/environment/datasets/criteo5m/working/submission.npz` |
| [OneTrans](onetrans_task/environment/README.md) | Click AUC on the fixed first 204,800 validation rows | `benchmark/tasks/onetrans_task/environment/datasets/kuaisar_v2/working/submission.csv` |

For example, validate the HSTU prediction format:

```bash
python tools/validate_submission.py --task hstu \
  --submission benchmark/tasks/hstu_task/environment/datasets/beauty/working/submission.npz
```

A passing local format check does not compute an official score. Hosted scoring
requires one six-task ZIP with members `<task>/<submission filename>`, using the
filenames above. Collect baseline files from these environment paths; the root
quickstart's automatic packager reads Harbor agent runs. Retain code and run
evidence separately from the prediction ZIP.

Fit only on public training inputs and select models using the task's public
validation rule. Hidden-test feedback must not guide selection. Report the public
data revision, source commit, seed, resources and dependency versions with a result.
Historical paper scores are separate from evaluations on the current public v2
inputs; running these starters does not establish reproduction of those scores.
