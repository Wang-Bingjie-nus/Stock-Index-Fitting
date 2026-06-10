import math

from hedge_system.portfolio.exposure import futures_notional


def test_futures_notional() -> None:
    assert math.isclose(futures_notional(2985.5048, 300, 14), 12_539_120.16)
