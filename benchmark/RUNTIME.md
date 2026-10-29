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

At finalization, agents must export their selected code, necessary helpers and
configuration to `/workspace/artifact/`, including `README.md` with the exact
commands and public-validation selection, and `SHA256SUMS`. From the task
environment, prepare that directory with:

```bash
python3 export_artifact.py --command 'bash run_baseline.sh DATASET' \
  --validation 'public validation result and selection rationale'
```

Add `--config relative/path.yaml` for configuration outside baseline and
`--helper relative/helper.py` for necessary source helpers outside baseline,
and `--overwrite` when replacing a prior selected artifact. Weights, input data,
and caches are excluded. Harbor collects the artifact directory and the prediction
file separately; source export does not expose or compute a hidden-test score.

The updated installed-harness adapter is specific to Harbor 0.18.0: it preserves
Codex's session directory between segments, delays credential/session cleanup
until finalization, and terminates only a timed-out segment's tagged processes.
The official [Codex non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode)
describes native session resumption. Harbor's [agent documentation](https://docs.harborframework.com/agents/pre-integrated-agents)
describes provider model selection and built-in harnesses.
