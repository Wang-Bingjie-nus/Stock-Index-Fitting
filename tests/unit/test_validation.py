import pandas as pd

from hedge_system.validation.checks import validate_weight_sum


def test_validate_weight_sum() -> None:
    constituents = pd.DataFrame({"weight": [0.6, 0.4]})
    assert validate_weight_sum(constituents)

