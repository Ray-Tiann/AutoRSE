# Bring your own agent

AutoRSE evaluates a versioned agent and model on six recommendation research tasks.
Your agent reads a task, edits its baseline, trains and compares experiments using
public validation, then exports the selected code and test predictions. The hosted
evaluator scores those predictions using hidden labels.

Choose the integration that fits your work:

| What you want to evaluate | Integration |
| --- | --- |
| Your model with an existing harness | `--agent codex`, `claude-code`, or `terminus-2`, plus `--model` |
| Your research policy with the supplied persistent Terminus-2 loop | Extend the shipped `EvidenceAgent` example and use `--agent-import-path` |
| Your own planning, tools, memory, or execution loop | Implement a Harbor agent adapter and use `--agent-import-path` |

The shipped [EvidenceAgent](../benchmark/custom_agents/evidence_agent.py) is a
minimal policy example. No trained result, benchmark score, or performance claim
is supplied for it.

## Prepare once

Follow the [runtime and public-data setup](../README.md) on a Linux machine with
one available NVIDIA GPU, Docker and NVIDIA Container Toolkit. Use a Python 3.12+
host environment and the pinned orchestration dependency:

```bash
python -m pip install -r requirements-harbor.txt
```

Run the commands below from the repository root. Replace `YOUR_MODEL_ID` with an
identifier accepted by your chosen harness. Put the provider credentials in your
environment or a private environment file outside the checkout; the examples use
`$HOME/.config/autorse/provider.env`. Model access and compatible provider settings
must be configured before starting a real run.

## Use your model with an existing harness

For example, run your model through the supplied persistent Terminus-2 harness:

```bash
python tools/run_agent.py \
  --task hstu --agent terminus-2 --model YOUR_MODEL_ID \
  --api-env-file "$HOME/.config/autorse/provider.env" \
  --gpu 0 --seed 2024 \
  --jobs-dir ./runs --job-name hstu-my-model-s2024
```

Use `--agent codex` or `--agent claude-code` for those harnesses, with a model and
credentials they support. Add `--dry-run` to inspect the planned command without
starting the agent, a GPU job, or an API request. Changing `--model` evaluates a
different model through the selected harness; it does not replace the harness's
planning and tool loop.

## Run a custom agent

The repository includes an importable example at
[`benchmark/custom_agents/evidence_agent.py`](../benchmark/custom_agents/evidence_agent.py).
It subclasses `PersistentTerminus2`, changes its name/version and research policy,
and delegates execution to the parent:

```python
async def run(self, instruction, environment, context):
    instruction += "\n\nEXAMPLE RESEARCH POLICY:\n" + self.RESEARCH_POLICY
    await super().run(instruction, environment, context)
```

The parent retains the research deadline, same-session continuation, finalization,
trajectory collection and usage context. Edit `RESEARCH_POLICY` to try your own
policy and update the example's version when you change it. Its constructor
explicitly forwards the budget and seed to the persistent adapter.

First inspect the launch:

```bash
python tools/run_agent.py \
  --task hstu \
  --agent-import-path custom_agents.evidence_agent:EvidenceAgent \
  --model YOUR_MODEL_ID --gpu 0 --seed 2024 \
  --jobs-dir ./runs --job-name hstu-evidence-s2024 \
  --dry-run
```

The dry run checks launcher arguments and prints the command; it does not import
the custom class. To check the shipped module and its installed dependencies
without starting a trial:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True \
PYTHONPATH="$PWD/benchmark${PYTHONPATH:+:$PYTHONPATH}" \
python -c 'from custom_agents.evidence_agent import EvidenceAgent; print(EvidenceAgent.name())'
```

Then launch the actual research run:

```bash
python tools/run_agent.py \
  --task hstu \
  --agent-import-path custom_agents.evidence_agent:EvidenceAgent \
  --model YOUR_MODEL_ID \
  --api-env-file "$HOME/.config/autorse/provider.env" \
  --gpu 0 --seed 2024 \
  --jobs-dir ./runs --job-name hstu-evidence-s2024
```

`--agent` and `--agent-import-path` are alternative selectors. The launcher adds
`benchmark/` to the Harbor process's Python path, so the shipped example imports
directly. For an agent in another project, add the absolute directory containing
its Python module/package to `PYTHONPATH` before invoking the same launcher:

```bash
PYTHONPATH="/absolute/path/to/your-agent-project${PYTHONPATH:+:$PYTHONPATH}" \
python tools/run_agent.py \
  --task hstu --agent-import-path my_agent:MyAgent --model YOUR_MODEL_ID \
  --api-env-file "$HOME/.config/autorse/provider.env" \
  --gpu 0 --seed 2024 --jobs-dir ./runs --job-name hstu-custom-s2024
```

The default budget is **43,200 research seconds plus at most 300 finalization
seconds**. One launcher invocation runs one task and one trial. Use a fresh job
name for every task/configuration/seed and run sequentially when sharing a GPU.
Short infrastructure checks may use `--minimum-runtime-sec 60
--finalization-grace-sec 30`; these are smoke runs and cannot serve as leaderboard
research runs.

## Implement a different harness

AutoRSE pins **Harbor 0.18.0**. Its factory loads `module.path:ClassName` and passes
`logs_dir`, `model_name`, `extra_env` and the constructor options. The AutoRSE
launcher supplies `minimum_runtime_sec`, `finalization_grace_sec` and
`research_seed`; a custom adapter must accept and implement these settings.
Accepting unused `**kwargs` does not enforce a research budget.

For an agent loop on the host controlling the task through `BaseEnvironment`,
implement Harbor's [versioned `BaseAgent` interface](https://github.com/harbor-framework/harbor/blob/v0.18.0/src/harbor/agents/base.py):
`name()`, `version()`, asynchronous `setup(environment)` and
`run(instruction, environment, context)`. For a CLI installed and run inside the
task container, use the [versioned `BaseInstalledAgent`](https://github.com/harbor-framework/harbor/blob/v0.18.0/src/harbor/agents/installed/base.py).
The [Harbor custom-agent guide](https://docs.harborframework.com/agents/custom-agents)
explains the two integration patterns; check APIs against the pinned source.

A new execution loop must handle research continuation after early completion,
phase deadlines, cancellation and process cleanup, and final selection/export.
The task's outer timeout alone does not implement the 12-hour research protocol.
Populate `AgentContext` with usage and execution metadata as the run progresses,
and preserve meaningful model/tool trajectories. The supplied example inherits
these responsibilities from `PersistentTerminus2`.

The custom-agent selector reports `supervision=custom_agent` in a dry run: the
imported class owns its supervision, including any persistent adapter it inherits.
`--no-persistence` is reserved for the supplied harnesses and cannot be combined
with a custom import path.

## Preserve predictions and reproduction evidence

Keep the task's data, architecture and prediction contracts. Fit on public training
data, select using the task's public-validation rule, use the trial seed
consistently, and use only its assigned GPU. Hidden-test scores must not guide
research or selection.

Inside the task environment, create a Git experiment branch before each coherent
change, commit source/helpers/configuration under `baseline/` before running
`run_baseline.sh`, and record each outcome with `baseline_git.py record`, including
failures. Keep candidate prediction copies outside Git. At finalization, restore
the best branch and its matching saved prediction, then use `export_artifact.py`
to export the selected source, `baseline.git.bundle`, `GIT_PROVENANCE.json` and
checksums. The [runtime guide](../benchmark/RUNTIME.md) gives the complete commands.

Harbor collects the prediction and `/workspace/artifact/` separately under
`runs/<job-name>/<trial>/artifacts/workspace/`. Keep the agent logs, structured
trajectory, actual configuration and run results. Preserve
`persistent-supervisor.jsonl` for the supplied adapter, or your custom supervisor
log for a different execution loop. The local Harbor `format_valid` result checks
prediction format only.

There are two source records to preserve. The task's Git bundle captures the
recommendation implementation edited inside `baseline/`. The custom adapter and
policy shown here load from the Harbor host and are **not automatically included
in that bundle**. Version and publish your harness source, prompts, dependencies
and model settings alongside the run evidence, with the AutoRSE commit and image
digest.

## Score and submit a complete result

The [official hosted evaluator](https://huggingface.co/spaces/Ray0311/AutoRSE-Evaluator)
accepts one prediction ZIP containing exactly these paths, with no enclosing
directory:

```text
hstu/submission.npz
intsr/submission.npz
tiger/submission.npz
onetrans/submission.csv
rankmixer/submission.npz
wukong/submission.csv.gz
```

Upload predictions only to the evaluator and download its signed, complete
six-task receipt. The repository's `space/` directory and older
`RayTian/AutoRSE` Space refer to the legacy CPU scoring template; use the official
evaluator above for current signed leaderboard receipts.

A leaderboard configuration requires **three independent trials per task: 18
research runs**, using the same versioned agent/model configuration and three
distinct seeds per task. Keep the fixed 12-hour, one-GPU protocol and collect three
signed six-task receipts, one for each trial index. Use no resource or timeout
overrides for a comparable leaderboard run. A single HSTU run or successful smoke
run does not constitute a complete benchmark result.

Publish the required code, report, trajectory, configuration and provenance for
every run to your public evidence dataset, pinned by immutable revision and file
hashes. Your evidence dataset is separate from AutoRSE's public input Dataset and
the evaluator. Keep the full source bundle for reproduction; it is separate from
the prediction ZIP sent for scoring. Follow the
[leaderboard submission guide](../leaderboard/SUBMIT.md) to prepare and validate
the evidence, then submit one result JSON through the review process.
