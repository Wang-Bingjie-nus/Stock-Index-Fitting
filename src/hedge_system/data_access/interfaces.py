"""Data-provider interfaces."""

from typing import Protocol

import pandas as pd


class IndexDataProvider(Protocol):
    def get_constituents(self, index_code: str, as_of_time: str) -> pd.DataFrame: ...


class MarketDataProvider(Protocol):
    def get_snapshot(self, security_codes: list[str], as_of_time: str) -> pd.DataFrame: ...

