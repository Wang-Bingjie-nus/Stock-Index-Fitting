import math

from hedge_system.risk.metrics import active_share


def test_active_share() -> None:
    assert math.isclose(active_share([0.6, 0.4], [0.5, 0.5]), 0.1)
