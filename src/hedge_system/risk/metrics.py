"""Risk metric placeholders."""


def active_share(actual_weights: list[float], target_weights: list[float]) -> float:
    """Calculate active share from aligned decimal weight vectors."""
    if len(actual_weights) != len(target_weights):
        raise ValueError("Weight vectors must have the same length.")
    return 0.5 * sum(abs(actual - target) for actual, target in zip(actual_weights, target_weights))

