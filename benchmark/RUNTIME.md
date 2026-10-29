# Public evaluation runtime

The repository supplies six editable baseline implementations, public training and
validation inputs, and unlabeled prediction schemas. Hidden-test scores come from
the hosted evaluator. The local Harbor metric `format_valid` checks prediction
format only; a value of 1 does not measure recommendation quality.

Build the shared Linux/NVIDIA image from the repository root:

```bash
bash tools/build_images.sh
python3 tools/run_baseline.py --task hstu --gpu 0
python3 tools/run_baseline.py --task tiger --gpu 0 -- --epochs 1 --max-users 1024
```

The image uses Ubuntu 22.04, CUDA 12.1, PyTorch 2.1.2, NumPy 1.26.4, Node 22,
and Ubuntu's tmux 3.2a. An NVIDIA driver compatible with CUDA 12.1 and NVIDIA
Container Toolkit are required. Docker GPU runs require Linux; Docker Desktop
on macOS cannot execute this GPU protocol. IntSR reads supplied frozen Qwen
features and requires no language-model checkpoint download. Baseline commands
mount the selected task environment at `/workspace`; outputs remain on the host.

For a host-managed CUDA environment, install that task's environment requirements
and use `tools/run_baseline.py --task TASK --local`. The Python CLI supports
`--dry-run` without Docker, PyTorch, data, or API credentials.

Optional Harbor orchestration uses a separate Python 3.12+ host environment:

```bash
python3 -m pip install -r requirements-harbor.txt
python3 tools/run_agent.py --task wukong --agent codex --model PROVIDER/MODEL --gpu 0
python3 tools/run_agent.py --task hstu --agent claude-code --model PROVIDER/MODEL --gpu 0
python3 tools/run_agent.py --task intsr --agent terminus-2 --model PROVIDER/MODEL --gpu 0
```

Supply provider credentials through the environment or `--api-env-file` pointing
outside the checkout. Credentials are never bundled. Use the provider/model
identifier accepted by the selected Harbor harness. Codex uses
`codex exec resume --last`; Claude Code uses `--continue`; Terminus-2 preserves
its chat and tmux session. These three persistence adapters default to 43,200
research seconds plus a bounded 300-second finalization phase. The supervisor
resumes early completions. Short infrastructure smokes may set
`--minimum-runtime-sec 60 --finalization-grace-sec 30`; record those settings.
`--no-persistence` invokes the unmodified Harbor harness.

The shared image installs current Codex and Claude Code CLI releases unless you
set `CODEX_VERSION` and `CLAUDE_CODE_VERSION` before building. Record resolved
CLI versions and the Docker image digest for reproducibility. Harbor is pinned
to 0.18.0 because the persistence adapters use its installed-harness and
Terminus-2 APIs. Re-run integration checks before changing that version.

Agent task images contain only their environment. The verifier runs in a
separate container with networking disabled and receives only declared
artifacts. The public verifier holds no hidden labels. Agent containers have
public network access for model API use; disabling Claude web tools does not
prevent network access through a shell. This release does not claim strict
network isolation for agents.

This public starter standardizes training on one visible GPU and omits the paper
corpus, historical experiment records, paper-specific idea gates, pretrained
baseline checkpoints, and the private offline verifier. It enables new baseline
training and agent experiments. Its settings, sanitized public data, and hosted
evaluation workflow must be reported separately from historical paper runs.
Static validation and CLI/configuration checks do not establish GPU-training
reproduction or reproduce the paper's performance numbers.

Each task runtime initializes a local Git repository at `/workspace/baseline`
before experiments, with the starter on branch `baseline` and immutable tag
`autorse-baseline`. Git is initialized in the running task; the public repository
does not contain a nested `.git` directory. Keep that initial reference and all
experiment branches for the final audit. Do not use submodules or push this local
repository to a remote.

Create one `exp/...` branch **before making each coherent experimental change**.
A branch represents a logical experiment, not each file save. Use `--from baseline`
for an independent idea or `--from exp/previous-name` to extend a previous idea.
Keep all experiment source, helpers, and configuration inside `baseline/`, and
commit the code before running it. Reserve `baseline/experiments/` for generated
JSON run records; put model code and configuration elsewhere under `baseline/`.
The workspace `run_baseline.sh` wrapper and `requirements.txt` stay unchanged; the export records hashes of required runtime
files outside the baseline in its provenance metadata.

Run experiments through `bash run_baseline.sh DATASET ...`. The wrapper captures
the experiment branch, evaluated code commit, command, exit status, and prediction
SHA256 in `/workspace/.autorse-runs/`, outside the baseline Git repository. After
every real run, `baseline_git.py record` uses that captured run to commit an
experiment record under `baseline/experiments/` on the same branch. Its metadata
commit does not change which code commit was evaluated. Record both successes and
failures, keep failed branches, and use only measured public validation for
selection. If a run fails before validation, say validation is unavailable and
record the error in `--notes`; do not invent a metric. Hidden-test scores must
never guide experiment selection.

For example, after installing HSTU's public data, use this workflow inside its
task container. Initialization is idempotent; it never resets an existing history:

```bash
cd /workspace
python3 baseline_git.py init
python3 baseline_git.py new exp/001-starter --from baseline
CUDA_VISIBLE_DEVICES=0 RESEARCH_SEED=2024 bash run_baseline.sh beauty
python3 baseline_git.py record \
  --validation 'full public-validation NDCG@10=<measured value>' \
  --notes 'Unchanged starter; record the observed outcome'
# After a successful run, retain its exact prediction outside Git.
mkdir -p datasets/beauty/working/experiments/exp/001-starter
cp datasets/beauty/working/submission.npz \
  datasets/beauty/working/experiments/exp/001-starter/submission.npz

python3 baseline_git.py new exp/002-hstu-change --from exp/001-starter
# Make one coherent change in baseline/model.py and any baseline/ helpers/config.
python3 baseline_git.py commit -m 'Describe the HSTU model change'
CUDA_VISIBLE_DEVICES=0 RESEARCH_SEED=2024 bash run_baseline.sh beauty
python3 baseline_git.py record \
  --validation 'full public-validation NDCG@10=<measured value>' \
  --notes 'Describe the change, observed outcome, and comparison with the starter'
# Save this candidate separately after a successful run.
mkdir -p datasets/beauty/working/experiments/exp/002-hstu-change
cp datasets/beauty/working/submission.npz \
  datasets/beauty/working/experiments/exp/002-hstu-change/submission.npz
```

Replace each validation placeholder with the result actually measured by that
run. Keep a unique prediction snapshot for each candidate/run you might select,
outside Git; the example uses one successful run per branch. Do not overwrite a
good snapshot after a failed run. Record a failure on its existing branch, for
example:

```bash
python3 baseline_git.py record --validation 'unavailable: run failed before validation' \
  --notes 'Describe the actual error and any partial public-validation result'
```

At finalization, select the best branch by public validation, ensure its source
and experiment records are committed, and restore its saved prediction. For
example, if `exp/001-starter` was selected:

```bash
git -C baseline switch exp/001-starter
git -C baseline status --short
cp datasets/beauty/working/experiments/exp/001-starter/submission.npz \
  datasets/beauty/working/submission.npz
python3 export_artifact.py \
  --command 'CUDA_VISIBLE_DEVICES=0 RESEARCH_SEED=2024 bash run_baseline.sh beauty' \
  --validation 'PUBLIC VALIDATION RESULT AND SELECTION RATIONALE'
```

The Git status must be empty. Predictions live outside Git: switching branches
does not restore them. Export verifies a recorded successful run on the selected
branch, the evaluated code and its ancestry, required runtime file hashes, and
the current prediction's exact bytes. This allows an earlier best experiment to
be selected without retraining during finalization, including after later failed
experiments, when its recorded code/runtime/prediction still match. A stale or
modified prediction, code, or required runtime file cannot reuse that receipt.
If no matching saved prediction/receipt is available, rerun the selected branch
through the wrapper and record its actual public-validation result before export.

Use the complete reproduction command, including nondefault environment
assignments and arguments, in `--command`. The captured run receipt separately
retains the executed trainer arguments. Add `--overwrite` when replacing a prior
selected artifact. Keep necessary editable helpers and configuration under
`baseline/` so their versions are captured by Git; exported external files are
identified as outside the baseline Git history in provenance metadata.

`/workspace/artifact/` contains the selected source/configuration, `README.md`,
`GIT_PROVENANCE.json`, `baseline.git.bundle`, and `SHA256SUMS`. The Git bundle
retains all experiment branches, their code history, and committed experiment
records, including failed attempts. It is a code/provenance artifact: weights,
input data, caches, and credentials must not enter Git or the export. The provenance
also identifies the selected branch, code commit, real run, current prediction
hash, and required runtime file hashes.

Harbor collects the complete artifact directory and prediction file separately.
Retain the whole artifact for reproduction and code audit. Hosted Hugging Face
scoring accepts **only the prediction file**; do not upload `baseline.git.bundle`
to the public Dataset or Space. Source export does not compute or expose a
hidden-test score.

The updated installed-harness adapter is specific to Harbor 0.18.0: it preserves
Codex's session directory between segments, delays credential/session cleanup
until finalization, and terminates only a timed-out segment's tagged processes.
The official [Codex non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode)
describes native session resumption. Harbor's [agent documentation](https://docs.harborframework.com/agents/pre-integrated-agents)
describes provider model selection and built-in harnesses.
