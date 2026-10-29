"""Terminus-2 adapter that keeps one chat and tmux session active for a fixed time."""

from __future__ import annotations

import asyncio
import json
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Step
from harbor.models.trial.paths import EnvironmentPaths


class _CompatibleTmuxSession(TmuxSession):
    """Start tmux without ``new-session -e``, unsupported by tmux 3.0a.

    Older images can ship tmux 3.0a. Harbor emits the
    newer ``-e KEY=value`` syntax, so pass the same variables through ``env``
    in the tmux shell command instead. This also works on newer tmux versions.
    """

    @property
    def _tmux_start_session(self) -> str:
        environment = " ".join(
            shlex.quote(f"{key}={value}") for key, value in self._extra_env.items()
        )
        shell_command = "env"
        if environment:
            shell_command += f" {environment}"
        shell_command += " bash --login"
        return (
            "export TERM=xterm-256color && "
            "export SHELL=/bin/bash && "
            f"tmux new-session -x {self._pane_width} -y {self._pane_height} "
            f"-d -s {shlex.quote(self._session_name)} "
            f"{shlex.quote(shell_command)} \\; "
            f"pipe-pane -t {shlex.quote(self._session_name)} "
            f"{shlex.quote(f'cat > {self._logging_path}')}"
        )


class PersistentTerminus2(Terminus2):
    """Re-enter the same Terminus-2 chat and terminal after an early completion."""

    def __init__(
        self,
        logs_dir: Path,
        minimum_runtime_sec: int | str = 43_200,
        finalization_grace_sec: int | str = 300,
        short_segment_backoff_sec: int | str = 10,
        research_seed: int | str = 2024,
        *args,
        **kwargs,
    ) -> None:
        self.minimum_runtime_sec = int(minimum_runtime_sec)
        self.finalization_grace_sec = int(finalization_grace_sec)
        self.short_segment_backoff_sec = int(short_segment_backoff_sec)
        self.research_seed = int(research_seed)
        if self.minimum_runtime_sec <= 0:
            raise ValueError("minimum_runtime_sec must be positive")
        if self.finalization_grace_sec <= 0:
            raise ValueError("finalization_grace_sec must be positive")
        self._supervisor_log = Path(logs_dir) / "persistent-supervisor.jsonl"
        self._persistent_original_instruction = ""
        self._persistent_total_episodes = 0
        # The trajectory and pane log remain available. Disabling asciinema
        # avoids a network package install during every fresh task container.
        kwargs.setdefault("record_terminal_session", False)
        super().__init__(logs_dir, *args, **kwargs)

    async def setup(self, environment: BaseEnvironment) -> None:
        if self._record_terminal_session:
            local_recording_path = environment.trial_paths.agent_dir / "recording.cast"
            remote_recording_path = EnvironmentPaths.agent_dir / "recording.cast"
        else:
            local_recording_path = None
            remote_recording_path = None

        self._session = _CompatibleTmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "terminus_2.pane",
            local_asciinema_recording_path=local_recording_path,
            remote_asciinema_recording_path=remote_recording_path,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
        )
        await self._session.start()

    def _record(self, event: str, **values: object) -> None:
        self._supervisor_log.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **values,
        }
        with self._supervisor_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _sync_context(self, context: AgentContext) -> None:
        """Preserve cumulative metrics after each continuation segment."""
        if self._chat is None:
            return
        context.rollout_details = (
            self._chat.rollout_details + self._subagent_rollout_details
        )
        context.n_input_tokens = (
            self._chat.total_input_tokens + self._subagent_metrics.total_prompt_tokens
        )
        context.n_output_tokens = (
            self._chat.total_output_tokens
            + self._subagent_metrics.total_completion_tokens
        )
        context.n_cache_tokens = (
            self._chat.total_cache_tokens + self._subagent_metrics.total_cached_tokens
        )
        total_cost = self._chat.total_cost + self._subagent_metrics.total_cost_usd
        context.cost_usd = total_cost if total_cost > 0 else None
        context.metadata = {
            "n_episodes": self._persistent_total_episodes,
            "api_request_times_msec": self._api_request_times,
            "summarization_count": self._summarization_count,
            "persistent_session": True,
        }
        if self._store_all_messages:
            context.metadata["all_messages"] = self._chat.messages

    async def _run_segment(
        self,
        prompt: str,
        environment: BaseEnvironment,
        context: AgentContext,
        *,
        resume: bool,
        segment: int,
    ) -> float:
        started = time.monotonic()
        self._record("segment_started", segment=segment, resume=resume)
        try:
            if not resume:
                await super().run(prompt, environment, context)
            else:
                if self._chat is None or self._session is None:
                    raise RuntimeError(
                        "Cannot resume Terminus-2 before its first segment"
                    )
                if not await self._session.is_session_alive():
                    raise RuntimeError(
                        "Cannot resume Terminus-2 because its tmux session ended"
                    )

                # An early return leaves the second completion marker set. Clear only
                # that marker; keep the full Chat history, terminal, and trajectory.
                self._pending_completion = False
                self._n_episodes = 0
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="user",
                        message=prompt,
                    )
                )
                await self._run_agent_loop(
                    initial_prompt=prompt,
                    chat=self._chat,
                    original_instruction=self._persistent_original_instruction,
                )
        finally:
            self._persistent_total_episodes += self._n_episodes
            self._sync_context(context)
            self._dump_trajectory()
            duration = time.monotonic() - started
            self._record(
                "segment_finished",
                segment=segment,
                resume=resume,
                duration_sec=duration,
                episodes=self._n_episodes,
            )
        return duration

    async def _run_bounded_segment(
        self,
        prompt: str,
        environment: BaseEnvironment,
        context: AgentContext,
        *,
        resume: bool,
        segment: int,
        timeout_sec: float,
        phase: str,
    ) -> tuple[float, bool]:
        """Run one segment, cancelling it when its phase budget expires."""
        timeout_sec = max(0.0, float(timeout_sec))
        if timeout_sec == 0:
            self._record(
                "segment_skipped_no_time",
                segment=segment,
                resume=resume,
                phase=phase,
            )
            return 0.0, True

        started = time.monotonic()
        try:
            duration = await asyncio.wait_for(
                self._run_segment(
                    prompt,
                    environment,
                    context,
                    resume=resume,
                    segment=segment,
                ),
                timeout=timeout_sec,
            )
            return duration, False
        except asyncio.TimeoutError:
            duration = time.monotonic() - started
            self._record(
                "segment_interrupted_at_deadline",
                segment=segment,
                resume=resume,
                phase=phase,
                timeout_sec=timeout_sec,
                duration_sec=duration,
            )
            return duration, True

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        started = time.monotonic()
        deadline = started + self.minimum_runtime_sec
        segment = 1
        initial_prompt = (
            instruction
            + "\n\n"
            + "PERSISTENT EXECUTION CONTRACT:\n"
            + "- The research phase lasts at least "
            + f"{self.minimum_runtime_sec} seconds.\n"
            + "- A valid submission or an apparent improvement does not end the run.\n"
            + "- Continue designing, running, and interpreting useful experiments.\n"
            + "- Keep the best validation-selected submission updated after every "
            + "successful experiment so a later timeout cannot lose it.\n"
            + "- Maintain baseline/.git: create a new exp/<name> branch for each "
            + "experimental change, commit source before run_baseline.sh, and use "
            + "baseline_git.py record after each outcome, including failures. Keep "
            + "experimental helpers/config inside baseline and save candidate "
            + "predictions outside Git. Preserve all branches and records.\n"
            + "- The research deadline is hard: the supervisor may interrupt the "
            + "current action when it arrives, then starts a separate finalization "
            + f"phase of at most {self.finalization_grace_sec} seconds.\n"
            + "- During the last 30 minutes, do not start work expected to cross the "
            + "research deadline and do not use blocking waits longer than 60 seconds. "
            + "Make sure a valid best-so-far submission already exists.\n"
            + "- If you return early, the supervisor will resume this same session.\n"
            + f"- This trial's fixed research seed is {self.research_seed}. Use it "
            + "for Python, NumPy, PyTorch, data loaders, and every comparable "
            + "training run; do not switch seeds within this trial.\n"
            + "- Use only the GPUs assigned and visible in this trial. First verify "
            + "the device count with nvidia-smi and do not launch work outside the "
            + "assigned GPUs.\n"
        )
        self._persistent_original_instruction = initial_prompt
        self._persistent_total_episodes = 0
        self._record(
            "supervisor_started",
            minimum_runtime_sec=self.minimum_runtime_sec,
            finalization_grace_sec=self.finalization_grace_sec,
            research_seed=self.research_seed,
        )
        _, research_deadline_hit = await self._run_bounded_segment(
            initial_prompt,
            environment,
            context,
            resume=False,
            segment=segment,
            timeout_sec=max(0.0, deadline - time.monotonic()),
            phase="research",
        )

        while not research_deadline_hit:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            segment += 1
            if remaining > 1800:
                phase_prompt = (
                    "The persistent research time budget is not exhausted. "
                    f"Approximately {int(remaining)} seconds remain. Continue the same "
                    "research process now. Inspect the existing experiment history, "
                    "derive the next evidence-driven modification on a new committed "
                    "exp/<name> branch, run it, record its outcome, compare validation "
                    "results, and preserve the best branch and prediction. Do not "
                    "finish or merely summarize while useful experiments remain."
                )
            else:
                phase_prompt = (
                    "The run is in its final research phase, with approximately "
                    f"{int(remaining)} seconds before mandatory finalization. Continue "
                    "working in the same session, but do not start training or a "
                    "blocking command expected to cross that deadline. Do not use "
                    "blocking waits longer than 60 seconds. Validate and preserve the "
                    "current best submission now, then use only short evidence-driven "
                    "checks. Do not exit yet."
                )
            duration, research_deadline_hit = await self._run_bounded_segment(
                phase_prompt,
                environment,
                context,
                resume=True,
                segment=segment,
                timeout_sec=max(0.0, remaining),
                phase="research",
            )
            if research_deadline_hit:
                break
            if duration < self.short_segment_backoff_sec:
                await asyncio.sleep(
                    min(float(self.short_segment_backoff_sec), max(0.0, remaining))
                )

        segment += 1
        elapsed = time.monotonic() - started
        self._record("research_deadline_reached", elapsed_sec=elapsed, segment=segment)
        final_prompt = (
            "The mandatory research phase is complete. You have at most "
            f"{self.finalization_grace_sec} seconds to finalize. Do not launch or wait "
            "for training. Stop outstanding background experiments, select strictly "
            "by existing public validation evidence, restore the selected branch "
            "and its matching saved prediction to the required path with the exact "
            "schema. Preserve all experiment branches and records. Run "
            "export_artifact.py with reproduction commands and validation rationale "
            "to create /workspace/artifact with the selected source, Git bundle, "
            "provenance and SHA256SUMS. Report the prediction path, selected branch "
            "and commit, artifact path and final conclusion, "
            "and return a concise final response immediately."
        )
        _, finalization_timed_out = await self._run_bounded_segment(
            final_prompt,
            environment,
            context,
            resume=True,
            segment=segment,
            timeout_sec=self.finalization_grace_sec,
            phase="finalization",
        )
        self._record(
            "supervisor_finished",
            elapsed_sec=time.monotonic() - started,
            finalization_timed_out=finalization_timed_out,
        )
