"""Infrastructure checks without Docker, GPUs, model APIs, or private inputs."""
from __future__ import annotations

import asyncio
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('hstu', 'tiger', 'intsr', 'wukong', 'rankmixer', 'onetrans')
os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')


class RuntimeCliTests(unittest.TestCase):
    def test_all_baselines_have_valid_dry_run(self):
        for task in TASKS:
            with self.subTest(task=task):
                result = subprocess.run([sys.executable, str(ROOT / 'tools/run_baseline.py'),
                                         '--task', task, '--dry-run'], capture_output=True,
                                        text=True, check=True)
                report = json.loads(result.stdout)
                self.assertFalse(report['training_started'])
                self.assertIn('target=/workspace', report['command'])

    def test_three_harnesses_are_explicitly_format_only(self):
        for agent in ('codex', 'claude-code', 'terminus-2'):
            with self.subTest(agent=agent):
                result = subprocess.run([sys.executable, str(ROOT / 'tools/run_agent.py'),
                                         '--task', 'tiger', '--agent', agent,
                                         '--model', 'provider/example', '--dry-run'],
                                        capture_output=True, text=True, check=True)
                report = json.loads(result.stdout)
                self.assertEqual(report['research_seconds'], 43200)
                self.assertEqual(report['local_verifier'], 'format_only_not_official_score')
                self.assertFalse(report['agent_started'])

    def test_multiple_gpus_rejected_for_single_gpu_protocol(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/run_baseline.py'),
                                 '--task', 'rankmixer', '--gpu', '0,1', '--dry-run'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_task_git_tooling_and_collection_wiring_matches_the_canonical_tools(self):
        for task in TASKS:
            with self.subTest(task=task):
                task_dir = ROOT / f'benchmark/tasks/{task}_task'
                environment = task_dir / 'environment'
                for name in ('baseline_git.py', 'export_artifact.py'):
                    self.assertEqual((ROOT / 'tools' / name).read_bytes(),
                                     (environment / name).read_bytes())
                wrapper = (environment / 'run_baseline.sh').read_text()
                self.assertIn('baseline_git.py', wrapper)
                self.assertIn(' run --workspace ', wrapper)
                self.assertIn(f'--task {task} ', wrapper)
                self.assertIn('train_baseline.sh', wrapper)
                self.assertIn('python3 /workspace/baseline_git.py init --workspace /workspace',
                              (environment / 'Dockerfile').read_text())
                self.assertNotRegex((task_dir / 'task.toml').read_text(),
                                    r'source\s*=\s*"/workspace/baseline"')

    def test_selected_artifact_records_source_config_and_checksums(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            (workspace / 'baseline').mkdir()
            (workspace / 'baseline/train.py').write_text('print("selected")\n')
            (workspace / 'baseline/checkpoint.pt').write_bytes(b'weights excluded')
            (workspace / 'run_baseline.sh').write_text('bash train_baseline.sh "$@"\n')
            (workspace / 'train_baseline.sh').write_text(
                '#!/usr/bin/env bash\nset -eu\n'
                'mkdir -p datasets/beauty/working\n'
                'cat baseline/train.py > datasets/beauty/working/submission.npz\n')
            (workspace / 'requirements.txt').write_text('numpy==1.26.4\n')
            (workspace / 'baseline/helper.py').write_text('print("helper")\n')
            (workspace / 'baseline/settings.yaml').write_text('batch_size: 2\n')
            def git_step(*command):
                subprocess.run([sys.executable, str(ROOT / 'tools/baseline_git.py'),
                                command[0], '--workspace', temp, *command[1:]],
                               check=True, capture_output=True)
            git_step('init')
            git_step('new', 'exp/selected', '--from', 'baseline')
            (workspace / 'baseline/train.py').write_text('print("selected experiment")\n')
            git_step('commit', '-m', 'Select fixture source')
            git_step('run', '--task', 'hstu', '--', 'bash', 'train_baseline.sh', 'beauty')
            git_step('record', '--validation', 'NDCG@10=0.1', '--notes', 'Selected fixture')
            subprocess.run([sys.executable, str(ROOT / 'tools/export_artifact.py'),
                            '--workspace', temp, '--command', 'bash run_baseline.sh beauty',
                            '--config', 'baseline/settings.yaml',
                            '--validation', 'NDCG@10=0.1'],
                           check=True, capture_output=True)
            artifact = workspace / 'artifact'
            self.assertTrue((artifact / 'source/baseline/train.py').is_file())
            self.assertFalse((artifact / 'source/baseline/checkpoint.pt').exists())
            self.assertTrue((artifact / 'config/baseline/settings.yaml').is_file())
            self.assertTrue((artifact / 'source/run_baseline.sh').is_file())
            self.assertTrue((artifact / 'source/requirements.txt').is_file())
            self.assertTrue((artifact / 'source/baseline/helper.py').is_file())
            self.assertTrue((artifact / 'baseline.git.bundle').is_file())
            self.assertTrue((artifact / 'GIT_PROVENANCE.json').is_file())
            self.assertIn('bash run_baseline.sh beauty', (artifact / 'README.md').read_text())
            for line in (artifact / 'SHA256SUMS').read_text().splitlines():
                digest, name = line.split('  ', 1)
                self.assertEqual(digest, hashlib.sha256((artifact / name).read_bytes()).hexdigest())


@unittest.skipUnless(importlib.util.find_spec('harbor'), 'Optional Harbor dependency is not installed')
class HarborCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / 'benchmark'))
        from harbor.models.task.config import TaskConfig
        from tools.persistent_claude_code import PersistentClaudeCode
        from tools.persistent_codex import PersistentCodex
        from tools.persistent_terminus_2 import PersistentTerminus2
        cls.TaskConfig = TaskConfig
        cls.codex_type = PersistentCodex
        cls.claude_type = PersistentClaudeCode
        cls.terminus_type = PersistentTerminus2

    def test_all_task_configs_load_in_pinned_harbor(self):
        for task in TASKS:
            path = ROOT / f'benchmark/tasks/{task}_task/task.toml'
            config = self.TaskConfig.model_validate_toml(path.read_text())
            self.assertEqual(config.verifier.network_mode, 'no-network')
            self.assertEqual(config.verifier.environment_mode, 'separate')
            self.assertEqual(config.metadata['local_metric'], 'format_valid_not_official_score')
            self.assertTrue(any(getattr(a, 'source', None) == '/workspace/artifact' for a in config.artifacts))

    async def test_codex_session_survives_segments_and_native_resume_is_used(self):
        from harbor.agents.installed.base import BaseInstalledAgent
        calls = []

        async def capture(base, environment, command, **kwargs):
            calls.append((command, kwargs))
            return None

        with tempfile.TemporaryDirectory() as temp:
            adapter = self.codex_type(Path(temp), model_name='openai/example')
            adapter._segment_token = 'example-segment'
            command = 'codex exec --json -- hello | tee /logs/agent/codex.txt'
            with patch.object(BaseInstalledAgent, '_exec', capture):
                await adapter._exec(None, command)
                adapter._persistent_resume = True
                await adapter._exec(None, command)
                await adapter._exec(None, 'rm -rf /tmp/codex-secrets "$CODEX_HOME"')
            self.assertEqual(len(calls), 2)
            self.assertNotIn('resume --last', calls[0][0])
            self.assertIn('resume --last -- hello', calls[1][0])
            self.assertEqual(calls[1][1]['env']['AUTORSE_SEGMENT_TOKEN'], 'example-segment')

    async def test_deadline_stops_tagged_processes_before_finalization(self):
        from harbor.agents.installed.codex import Codex
        calls = []

        class Environment:
            async def exec(self, command, **kwargs):
                calls.append(command)

        async def running(*args, **kwargs):
            await asyncio.sleep(60)

        with tempfile.TemporaryDirectory() as temp:
            adapter = self.codex_type(Path(temp), model_name='openai/example')
            with patch.object(Codex, 'run', running):
                timed_out = await adapter._segment('research', Environment(), None, 0.001, 'research')
            self.assertTrue(timed_out)
            self.assertTrue(adapter._persistent_resume)
            self.assertEqual(len(calls), 1)
            self.assertIn('AUTORSE_SEGMENT_TOKEN', calls[0])
            self.assertIn(adapter._segment_token, calls[0])

    async def test_claude_native_session_continues(self):
        from harbor.agents.installed.base import BaseInstalledAgent
        calls = []

        async def capture(base, environment, command, **kwargs):
            calls.append(command)

        with tempfile.TemporaryDirectory() as temp:
            adapter = self.claude_type(Path(temp), model_name='anthropic/example')
            adapter._persistent_resume = True
            with patch.object(BaseInstalledAgent, '_exec', capture):
                await adapter._exec(None, 'claude --verbose --output-format=stream-json --print -- hello')
            self.assertIn('claude --continue --verbose', calls[0])


if __name__ == '__main__':
    unittest.main()
