"""Core domain model placeholders."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StrategyRun:
    run_id: str
    as_of_time: datetime
    index_code: str
    futures_contract: str

