"""Validation placeholders."""

import pandas as pd


def validate_weight_sum(constituents: pd.DataFrame, tolerance: float = 0.001) -> bool:
    """Return whether decimal weights sum to one within tolerance."""
    return abs(float(constituents["weight"].sum()) - 1.0) <= tolerance

