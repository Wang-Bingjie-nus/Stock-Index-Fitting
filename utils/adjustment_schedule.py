from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterable, Sequence

from .minute_tick_cache_v03 import (
    load_unavailable_minute_tick_cache,
    normalize_trade_date,
    unavailable_minute_tick_cache_path,
)


@dataclass(frozen=True)
class IntervalAdjustmentSchedule:
    interval_trading_days: int
    calendar_dates: tuple[str, ...]
    candidate_dates: tuple[str, ...]
    adjusting_dates: tuple[str, ...]
    shifted_dates: tuple[tuple[str, str], ...]
    omitted_dates: tuple[str, ...]


@dataclass(frozen=True)
class AdjustmentDateResolution:
    calendar_dates: tuple[str, ...]
    candidate_dates: tuple[str, ...]
    adjusting_dates: tuple[str, ...]
    shifted_dates: tuple[tuple[str, str], ...]
    omitted_dates: tuple[str, ...]


def adjustment_interval_from_config(adjusting_date_ls: Sequence[object]) -> int | None:
    """Return the positive interval encoded by ``[int]``, otherwise ``None``."""

    if not isinstance(adjusting_date_ls, (list, tuple)):
        raise ValueError("adjusting_date_ls must be a list or tuple.")
    if len(adjusting_date_ls) != 1:
        return None

    value = adjusting_date_ls[0]
    if isinstance(value, bool):
        raise ValueError("A boolean is not a valid adjusting-date interval.")
    if not isinstance(value, Integral):
        return None

    interval = int(value)
    if interval <= 0:
        raise ValueError("The adjusting-date interval must be a positive integer.")
    return interval


def find_unavailable_marker_dates(
    cache_dir: str | Path,
    trading_dates: Iterable[object],
) -> list[str]:
    """Return dates carrying a valid unavailable-cache marker."""

    cache_dir = Path(cache_dir)
    unavailable_dates: list[str] = []
    for value in trading_dates:
        trade_date = normalize_trade_date(value)
        marker_path = unavailable_minute_tick_cache_path(cache_dir, trade_date)
        if not marker_path.is_file():
            continue
        load_unavailable_minute_tick_cache(marker_path, trade_date=trade_date)
        unavailable_dates.append(trade_date)
    return unavailable_dates


def first_trading_dates_of_later_months(
    trading_dates: Iterable[object],
) -> tuple[str, ...]:
    """Return each later month's first trading date, excluding the start month."""

    calendar_dates = tuple(sorted({normalize_trade_date(value) for value in trading_dates}))
    if not calendar_dates:
        raise ValueError("trading_dates must not be empty.")

    first_month = calendar_dates[0][:6]
    first_date_by_month: dict[str, str] = {}
    for trade_date in calendar_dates:
        first_date_by_month.setdefault(trade_date[:6], trade_date)

    return tuple(
        trade_date
        for month, trade_date in first_date_by_month.items()
        if month != first_month
    )


def resolve_adjusting_date_candidates(
    candidate_dates: Iterable[object],
    trading_dates: Iterable[object],
    *,
    unavailable_dates: Iterable[object] = (),
) -> AdjustmentDateResolution:
    """Roll adjustment candidates forward past unavailable trading dates."""

    calendar_dates = tuple(sorted({normalize_trade_date(value) for value in trading_dates}))
    if not calendar_dates:
        raise ValueError("trading_dates must not be empty.")

    normalized_candidates = tuple(sorted({normalize_trade_date(value) for value in candidate_dates}))
    unavailable_set = {normalize_trade_date(value) for value in unavailable_dates}
    calendar_position = {trade_date: index for index, trade_date in enumerate(calendar_dates)}
    invalid_candidates = [date for date in normalized_candidates if date not in calendar_position]
    if invalid_candidates:
        raise ValueError(
            "Adjustment candidates must be trading dates inside the fitting range: "
            f"{invalid_candidates}"
        )

    adjusting_dates: list[str] = []
    shifted_dates: list[tuple[str, str]] = []
    omitted_dates: list[str] = []
    for candidate_date in normalized_candidates:
        resolved_position = calendar_position[candidate_date]
        while (
            resolved_position < len(calendar_dates)
            and calendar_dates[resolved_position] in unavailable_set
        ):
            resolved_position += 1

        if resolved_position >= len(calendar_dates):
            omitted_dates.append(candidate_date)
            continue

        resolved_date = calendar_dates[resolved_position]
        if resolved_date != candidate_date:
            shifted_dates.append((candidate_date, resolved_date))
        if resolved_date not in adjusting_dates:
            adjusting_dates.append(resolved_date)

    return AdjustmentDateResolution(
        calendar_dates=calendar_dates,
        candidate_dates=normalized_candidates,
        adjusting_dates=tuple(adjusting_dates),
        shifted_dates=tuple(shifted_dates),
        omitted_dates=tuple(omitted_dates),
    )


def resolve_interval_adjusting_dates(
    interval_trading_days: int,
    trading_dates: Iterable[object],
    *,
    unavailable_dates: Iterable[object] = (),
) -> IntervalAdjustmentSchedule:
    """Build an interval schedule and roll unavailable candidates forward.

    The first trading date represents the initial basket construction. Therefore,
    an interval ``N`` produces the first candidate at position ``N`` (zero based).
    Each candidate keeps its original cadence; rolling one candidate forward does
    not reset subsequent candidates. Duplicate rolled dates are collapsed.
    """

    if isinstance(interval_trading_days, bool) or not isinstance(interval_trading_days, Integral):
        raise ValueError("interval_trading_days must be a positive integer.")
    interval = int(interval_trading_days)
    if interval <= 0:
        raise ValueError("interval_trading_days must be a positive integer.")

    calendar_dates = tuple(sorted({normalize_trade_date(value) for value in trading_dates}))
    if not calendar_dates:
        raise ValueError("trading_dates must not be empty.")

    candidate_dates = tuple(calendar_dates[index] for index in range(interval, len(calendar_dates), interval))
    resolution = resolve_adjusting_date_candidates(
        candidate_dates,
        calendar_dates,
        unavailable_dates=unavailable_dates,
    )

    return IntervalAdjustmentSchedule(
        interval_trading_days=interval,
        calendar_dates=calendar_dates,
        candidate_dates=candidate_dates,
        adjusting_dates=resolution.adjusting_dates,
        shifted_dates=resolution.shifted_dates,
        omitted_dates=resolution.omitted_dates,
    )


__all__ = [
    "AdjustmentDateResolution",
    "IntervalAdjustmentSchedule",
    "adjustment_interval_from_config",
    "find_unavailable_marker_dates",
    "first_trading_dates_of_later_months",
    "resolve_adjusting_date_candidates",
    "resolve_interval_adjusting_dates",
]
