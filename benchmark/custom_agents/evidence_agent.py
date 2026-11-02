"""Example research policy adapter; no evaluated benchmark results are supplied."""

from __future__ import annotations

from pathlib import Path

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from tools.persistent_terminus_2 import PersistentTerminus2


class EvidenceAgent(PersistentTerminus2):
    """Customize the research policy while inheriting the persistent execution loop."""

    RESEARCH_POLICY = (
        "Before each experiment, state a testable hypothesis and the public-validation "
        "comparison that will support or reject it. Make one coherent change at a time. "
        "Interpret failed and successful runs before choosing the next experiment. "
        "Use the validation rule required by the task, and preserve the best branch "
        "and its matching prediction throughout the run."
    )

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        minimum_runtime_sec: int | str = 43_200,
        finalization_grace_sec: int | str = 300,
        research_seed: int | str = 2024,
        **kwargs,
    ) -> None:
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            minimum_runtime_sec=minimum_runtime_sec,
            finalization_grace_sec=finalization_grace_sec,
            research_seed=research_seed,
            **kwargs,
        )

    @staticmethod
    def name() -> str:
        return "evidence-agent"

    def version(self) -> str:
        return "0.1.0"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        instruction += "\n\nEXAMPLE RESEARCH POLICY:\n" + self.RESEARCH_POLICY
        await super().run(instruction, environment, context)
