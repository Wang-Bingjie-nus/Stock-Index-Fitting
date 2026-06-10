"""Repository interface placeholders."""

from typing import Protocol

from hedge_system.models.domain import StrategyRun


class StrategyRunRepository(Protocol):
    def save(self, run: StrategyRun) -> None: ...

