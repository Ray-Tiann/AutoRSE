"""Wall-clock persistence for the installed harness API in Harbor 0.18.0."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
import time
import uuid


class PersistentInstalled:
    """Mixin retaining one native CLI session through a bounded research run."""

    harness = ''

    def __init__(self, logs_dir: Path, minimum_runtime_sec=43200,
                 finalization_grace_sec=300, short_segment_backoff_sec=10,
                 research_seed=2024, *args, **kwargs):
        self.minimum_runtime_sec = int(minimum_runtime_sec)
        self.finalization_grace_sec = int(finalization_grace_sec)
        self.short_segment_backoff_sec = int(short_segment_backoff_sec)
        self.research_seed = int(research_seed)
        if min(self.minimum_runtime_sec, self.finalization_grace_sec) <= 0:
            raise ValueError('Research and finalization durations must be positive')
        self._supervisor_log = Path(logs_dir) / 'persistent-supervisor.jsonl'
        self._persistent_resume = False
        self._segment_token = ''
        super().__init__(logs_dir, *args, **kwargs)

    def _record(self, event, **values):
        self._supervisor_log.parent.mkdir(parents=True, exist_ok=True)
        with self._supervisor_log.open('a') as stream:
            stream.write(json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(),
                                     'event': event, **values}) + '\n')

    async def _exec(self, environment, command, user=None, env=None, cwd=None, timeout_sec=None):
        # Harbor 0.18.0 has no `_resume` field. Its Codex cleanup removes the
        # session home after every run; defer that cleanup until finalization.
        if self.harness == 'codex':
            if command.startswith('rm -rf ') and '"$CODEX_HOME"' in command:
                return SimpleNamespace(return_code=0, stdout='', stderr='')
            command = command.replace('cat >>"$CODEX_HOME/config.toml"',
                                      'cat >"$CODEX_HOME/config.toml"')
        active = ('codex exec ' in command if self.harness == 'codex'
                  else 'claude --verbose --output-format=stream-json ' in command)
        if active:
            if self._persistent_resume:
                if self.harness == 'codex':
                    command = command.replace('-- ', 'resume --last -- ', 1)
                else:
                    command = command.replace('claude --verbose ', 'claude --continue --verbose ', 1)
            command = command.replace('| tee ', '| tee -a ')
            env = {**(env or {}), 'AUTORSE_SEGMENT_TOKEN': self._segment_token}
        return await super()._exec(environment, command, user=user, env=env,
                                   cwd=cwd, timeout_sec=timeout_sec)

    async def _stop_segment(self, environment):
        # Cancelled docker-exec calls can leave CLI children alive. Match only
        # this adapter's random segment token, inherited by its process tree.
        source = '''import os, pathlib, signal, time
tag = b'AUTORSE_SEGMENT_TOKEN=' + __SEGMENT_VALUE__.encode()
pids = []
for entry in pathlib.Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid(): continue
    try:
        if tag in (entry / 'environ').read_bytes().split(b'\\0'): pids.append(int(entry.name))
    except (FileNotFoundError, PermissionError, ProcessLookupError): pass
for sig in (signal.SIGTERM, signal.SIGKILL):
    for pid in pids:
        try: os.kill(pid, sig)
        except ProcessLookupError: pass
    if sig == signal.SIGTERM: time.sleep(1)
'''.replace('__SEGMENT_VALUE__', repr(self._segment_token))
        await environment.exec(command='python3 -c ' + shlex.quote(source), timeout_sec=10)

    async def _segment(self, prompt, environment, context, seconds, phase):
        self._segment_token = uuid.uuid4().hex
        self._record('segment_started', phase=phase, resume=self._persistent_resume,
                     timeout_sec=seconds)
        try:
            await asyncio.wait_for(super().run(prompt, environment, context), timeout=seconds)
            return False
        except asyncio.TimeoutError:
            await self._stop_segment(environment)
            self._record('segment_deadline', phase=phase)
            return True
        finally:
            self._persistent_resume = True
            self._record('segment_finished', phase=phase)

    async def run(self, instruction, environment, context):
        started = time.monotonic()
        deadline = started + self.minimum_runtime_sec
        contract = (
            f'\n\nResearch budget: {self.minimum_runtime_sec} seconds, followed by at most '
            f'{self.finalization_grace_sec} seconds for finalization. Keep working until the '
            'research deadline; an early final answer resumes this same native session. '
            f'Use seed {self.research_seed} consistently and only the visible GPU. '
            'Fit only public training data and select on public validation. Maintain a valid '
            'best-so-far prediction artifact. Local format_valid is a schema check; official '
            'hidden-test scoring is hosted separately. Do not attempt to recover hidden targets. '
            'Maintain baseline/.git: create a new exp/<name> branch for each experimental change, '
            'commit source before running run_baseline.sh, and record each outcome with '
            'baseline_git.py record, including failures. Keep experimental helpers/config inside '
            'baseline and save each candidate prediction outside Git. '
            'No paper knowledge base or previous experiment records are supplied.\n'
        )
        prompt = instruction + contract
        self._record('supervisor_started', research_seconds=self.minimum_runtime_sec,
                     research_seed=self.research_seed)
        try:
            while (remaining := deadline - time.monotonic()) > 0:
                segment_started = time.monotonic()
                timed_out = await self._segment(prompt, environment, context, remaining, 'research')
                if timed_out:
                    break
                if time.monotonic() - segment_started < self.short_segment_backoff_sec:
                    await asyncio.sleep(min(self.short_segment_backoff_sec,
                                            max(0, deadline - time.monotonic())))
                prompt = ('Continue this same research session. Review public-validation evidence, '
                          'run a useful next experiment on a new committed exp/<name> branch, '
                          'record its outcome, and preserve the best branch and prediction. '
                          f'{max(0, int(deadline - time.monotonic()))} research seconds remain.')
            await self._segment('The research deadline has arrived. Stop new experiments. '
                                'Select the best branch using existing public validation evidence '
                                'and restore that branch plus its matching saved prediction to the '
                                'task’s declared path. Preserve all experiment branches and records. '
                                'Run export_artifact.py with reproduction commands and validation '
                                'rationale to create /workspace/artifact, including the selected '
                                'source, Git bundle, provenance and SHA256SUMS. Report the prediction '
                                'path, selected branch/commit and artifact path. '
                                'Do not claim a hidden-test score.',
                                environment, context, self.finalization_grace_sec, 'finalization')
        finally:
            if self.harness == 'codex':
                await environment.exec(command='rm -rf /tmp/codex-secrets /tmp/codex-home', timeout_sec=10)
            self._record('supervisor_finished', elapsed_sec=time.monotonic() - started)
