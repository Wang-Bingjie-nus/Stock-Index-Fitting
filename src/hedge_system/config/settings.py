"""Configuration model placeholder."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategySettings:
    index_code: str = "000016.SH"
    futures_product: str = "IH"
    hedge_ratio: float = 1.0

