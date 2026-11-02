<div align="center">

<img src="docs/assets/banner.svg" width="100%" alt="AutoRSE — A framework for evaluating autonomous recommendation research agents"/>

<p>
  <a href="https://arxiv.org/abs/2610.37228"><img src="https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Paper"/></a>
  <a href="https://ray-tiann.github.io/AutoRSE_WEB/"><img src="https://img.shields.io/badge/Homepage-FFD21E?style=for-the-badge" alt="Homepage"/></a>
  <a href="https://huggingface.co/spaces/Ray0311/AutoRSE-Leaderboard"><img src="https://img.shields.io/badge/Leaderboard-2563EB?style=for-the-badge&amp;logo=huggingface&amp;logoColor=white" alt="Leaderboard"/></a>
</p>

</div>

Built upon [Harbor](https://harborframework.com/), AutoRSE evaluates the capability of LLM agents to autonomously optimize six recommendation baselines through iterative code modifications and empirical experimentation.

## 🧭 Workflow

1. **Task setup.** Each run starts with a baseline implementation, task instructions, and public training, validation, and unlabeled test inputs. The agent works in a container with file-editing tools, shell access, dependency installation, and one visible GPU.

2. **Research loop (12 hours).** The agent explores the code and data, proposes improvements, implements changes, trains, and analyzes validation results to plan the next experiment. Each logical experiment uses a Git branch; both successful and failed outcomes are recorded.

3. **Final selection and export.** The agent selects its best candidate using public validation and restores the matching test predictions. It exports the selected code, helpers, configuration, Git history, and prediction hashes to `/workspace/artifact/`; the agent trajectory is retained separately.

4. **Hosted evaluation.** After completing all six tasks, package their predictions into one ZIP and upload it to the Hugging Face evaluator. The service returns task metrics such as NDCG@10 or AUC and a signed receipt. Hidden labels stay private, and hidden-test feedback must not guide research or selection.

![AutoRSE workflow](docs/assets/workflow.png)

## 🚀 Quickstart

**Requirements:** Linux · one NVIDIA GPU · CUDA 12.1-compatible driver · Docker + NVIDIA Container Toolkit · Python 3.12 · Bash.

### 1. Install

```bash
git clone https://github.com/Ray-Tiann/AutoRSE.git
cd AutoRSE
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-harbor.txt
bash tools/build_images.sh
```

### 2. Download the data

```bash
python tools/install_public_data.py --repo-id Ray0311/AutoRSE
```

To install from a local copy of the reviewed v2 ZIP:

```bash
python tools/install_public_data.py --archive /path/to/recagent-data-public-v2.zip
```

### 3. Configure your model

Create a private credential file outside the repository:

```bash
mkdir -p "$HOME/.config/autorse"
${EDITOR:-vi} "$HOME/.config/autorse/provider.env"
chmod 600 "$HOME/.config/autorse/provider.env"
```

For an OpenAI-backed Terminus-2 run, put this in the file and replace the value:

```dotenv
OPENAI_API_KEY=YOUR_API_KEY
```

Set the agent's LLM model ID, credential-file path, and a run label:

```bash
export AUTORSE_MODEL="openai/YOUR_MODEL_ID"
export AUTORSE_API_ENV="$HOME/.config/autorse/provider.env"
export AUTORSE_RUN_LABEL="my-agent-v1"
export AUTORSE_SEED=2024
```

For other providers or harnesses, use their supported model IDs and credential variables.

### 4. Choose and run your agent

**Choose one** in the same Bash session:

```bash
# A. Use the supplied Terminus-2 harness
AUTORSE_AGENT_ARGS=(--agent terminus-2)
```

```bash
# B. Use your own agent; replace the shipped example with your module:Class
AUTORSE_AGENT_ARGS=(--agent-import-path custom_agents.evidence_agent:EvidenceAgent)
```

First check the launch command; this starts no agent or training:

```bash
python tools/run_agent.py --task hstu "${AUTORSE_AGENT_ARGS[@]}" \
  --model "$AUTORSE_MODEL" --api-env-file "$AUTORSE_API_ENV" \
  --gpu 0 --seed "$AUTORSE_SEED" --dry-run
```

Run all six tasks sequentially, using one GPU:

```bash
for AUTORSE_TASK in hstu intsr tiger onetrans rankmixer wukong; do
  python tools/run_agent.py --task "$AUTORSE_TASK" "${AUTORSE_AGENT_ARGS[@]}" \
    --model "$AUTORSE_MODEL" --api-env-file "$AUTORSE_API_ENV" \
    --gpu 0 --seed "$AUTORSE_SEED" \
    --jobs-dir ./runs --job-name "${AUTORSE_RUN_LABEL}-${AUTORSE_TASK}-s${AUTORSE_SEED}" || break
done
```

- Default: **12 hours of research + up to 300 seconds of finalization per task**.
- Change `AUTORSE_RUN_LABEL` for a new agent/configuration or rerun. To try one task first, replace the loop's task list with `hstu`.
- Custom harnesses must honor the same budget. Select on public validation; no hidden-test feedback, external knowledge banks, or historical experiment cards.

### 5. Validate and package the predictions

Validate the six collected predictions and create the evaluator ZIP:

```bash
python tools/package_predictions.py --runs-dir ./runs \
  --run-label "$AUTORSE_RUN_LABEL" --seed "$AUTORSE_SEED"
```

The command refuses missing or ambiguous trials. Upload the ZIP filename it prints.

Local validation checks **format only**. Keep the selected source `artifact/`, agent logs, and configuration under `runs/` for reproduction.

### 6. Get the official score

1. Open [AutoRSE Evaluator](https://huggingface.co/spaces/Ray0311/AutoRSE-Evaluator) and sign in.
2. Upload the ZIP printed by step 5 and click **Evaluate all six tasks**.
3. Download the signed JSON receipt and keep it with your run evidence.

Only predictions go in this ZIP. Source code, Git bundles, and logs stay in the separate evidence package.

## 🏆 Submit to the leaderboard

- Run **three independent trials per task** with three distinct seeds: **18 research runs**.
- For the next trial, change `AUTORSE_SEED` and repeat steps 4–6. Keep the same versioned agent/model configuration.
- Publish each run's evidence and follow the submission commands in [submission guide](https://github.com/Ray-Tiann/AutoRSE_WEB/blob/main/leaderboard/SUBMIT.md).

Code: Apache-2.0. Dataset and third-party terms remain with their original sources.
