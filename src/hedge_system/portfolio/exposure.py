"""Exposure calculation placeholders."""


def futures_notional(futures_price: float, multiplier: float, contracts: int) -> float:
    """Calculate futures notional value."""
    return futures_price * multiplier * contracts

