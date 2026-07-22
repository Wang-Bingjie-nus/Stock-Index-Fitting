from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.covariance import oas


MINUTES_PER_TRADING_DAY = 240
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ShrunkRiskModel:
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    returns: pd.DataFrame
    summary: dict


def _normalize_stock_code(value) -> str:
    raw = str(value).strip().upper()
    if raw.endswith((".SH", ".SZ", ".BJ")):
        return raw
    digits = raw.split(".")[0].zfill(6)
    if digits.startswith(("5", "6", "9")):
        return f"{digits}.SH"
    if digits.startswith(("0", "2", "3")):
        return f"{digits}.SZ"
    if digits.startswith(("4", "8")):
        return f"{digits}.BJ"
    return raw


def _extract_minute_price_series(frame: pd.DataFrame, price_col: str) -> pd.Series:
    if frame is None or frame.empty or price_col not in frame.columns:
        return pd.Series(dtype=float)

    df = frame.copy()
    if "time" in df.columns:
        raw_time = pd.to_numeric(df["time"], errors="coerce")
    else:
        raw_time = pd.Series(df.index, index=df.index)
        if np.issubdtype(raw_time.dtype, np.datetime64):
            raw_time = pd.to_datetime(raw_time).strftime("%H%M%S")
        raw_time = pd.to_numeric(raw_time, errors="coerce")

    price = pd.to_numeric(df[price_col], errors="coerce")
    valid = raw_time.notna() & price.notna() & price.gt(0)
    if not valid.any():
        return pd.Series(dtype=float)

    hhmmss = raw_time.loc[valid].astype("int64").map(lambda value: int(str(value)[-6:]))
    minute = (hhmmss // 100).astype(int)
    series = pd.Series(price.loc[valid].to_numpy(dtype=float), index=minute)
    in_session = series.index.map(lambda value: 930 <= value <= 1130 or 1300 <= value <= 1500)
    return series.loc[in_session].groupby(level=0).last().sort_index()


def build_intraday_log_return_matrix(
    tick_dict_by_date: dict[str, dict[str, pd.DataFrame | None]],
    stock_codes: list[str],
    *,
    price_col: str = "lastPrice",
) -> pd.DataFrame:
    """Build 1-minute log returns separately for each day.

    Returns never span two trading days, so overnight returns are excluded.
    """

    stock_codes = [_normalize_stock_code(code) for code in stock_codes]
    daily_returns = []

    for date_key, tick_dict in sorted(tick_dict_by_date.items()):
        price_columns = {}
        for code in stock_codes:
            raw = None if tick_dict is None else tick_dict.get(code)
            series = _extract_minute_price_series(raw, price_col)
            if not series.empty:
                price_columns[code] = series

        if not price_columns:
            continue

        prices = pd.DataFrame(price_columns).sort_index().reindex(columns=stock_codes).ffill()
        log_returns = np.log(prices).diff().iloc[1:].ffill()
        if log_returns.empty:
            continue
        log_returns.index = [f"{date_key}_{int(minute):04d}" for minute in log_returns.index]
        daily_returns.append(log_returns)

    if not daily_returns:
        return pd.DataFrame(columns=stock_codes)

    returns = pd.concat(daily_returns, axis=0).reindex(columns=stock_codes)
    missing = returns.isna().sum()
    if missing.any():
        details = missing.loc[missing.gt(0)].sort_values(ascending=False).head(10).to_dict()
        raise ValueError(f"Risk returns still contain missing values after ffill: {details}")
    return returns


def _day_decay_weights(index: pd.Index, half_life_days: float) -> tuple[np.ndarray, list[str]]:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive.")

    day_keys = pd.Index([str(item).split("_", 1)[0] for item in index])
    unique_days = sorted(day_keys.unique().tolist())
    day_age = {day: len(unique_days) - 1 - position for position, day in enumerate(unique_days)}
    raw = np.array([0.5 ** (day_age[day] / float(half_life_days)) for day in day_keys], dtype=float)
    return raw / raw.sum(), unique_days


def build_ew_oas_covariance(
    returns: pd.DataFrame,
    *,
    half_life_days: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Estimate an exponentially day-weighted OAS covariance matrix.

    OAS is fitted to weighted, centered observations. The transformation keeps
    the empirical covariance equal to the requested day-weighted covariance
    before shrinkage while allowing sklearn to choose shrinkage automatically.
    """

    if returns.empty:
        raise ValueError("returns is empty; cannot build risk matrix.")
    clean = returns.astype(float)
    if clean.isna().any().any():
        raise ValueError("returns contains missing values after within-day ffill.")

    sample_weights, unique_days = _day_decay_weights(clean.index, half_life_days)
    values = clean.to_numpy(dtype=float, copy=True)
    weighted_mean = np.sum(values * sample_weights[:, None], axis=0)
    centered = values - weighted_mean[None, :]

    # sklearn uses 1 / n_rows internally. This scaling reproduces the desired
    # weighted covariance before OAS shrinkage.
    scaled = centered * np.sqrt(sample_weights[:, None] * len(sample_weights))
    shrunk_values, shrinkage = oas(scaled, assume_centered=True)
    shrunk_values = (shrunk_values + shrunk_values.T) / 2.0

    codes = clean.columns.tolist()
    covariance = pd.DataFrame(shrunk_values, index=codes, columns=codes)
    vol = np.sqrt(np.clip(np.diag(shrunk_values), 0.0, None))
    denom = np.outer(vol, vol)
    correlation_values = np.divide(
        shrunk_values,
        denom,
        out=np.zeros_like(shrunk_values),
        where=denom > 0,
    )
    np.fill_diagonal(correlation_values, 1.0)
    correlation = pd.DataFrame(correlation_values, index=codes, columns=codes)

    effective_rows = float(1.0 / np.square(sample_weights).sum())
    summary = {
        "estimator": "ew_oas_covariance",
        "risk_dates": ",".join(unique_days),
        "risk_date_count": len(unique_days),
        "return_rows": int(len(clean)),
        "stock_count": int(len(codes)),
        "half_life_days": float(half_life_days),
        "effective_return_rows": effective_rows,
        "oas_shrinkage": float(shrinkage),
        "annualization_periods": MINUTES_PER_TRADING_DAY * TRADING_DAYS_PER_YEAR,
    }
    return covariance, correlation, summary


def build_shrunk_risk_model_from_tick_dicts(
    tick_dict_by_date: dict[str, dict[str, pd.DataFrame | None]],
    stock_codes: list[str],
    *,
    price_col: str = "lastPrice",
    half_life_days: float = 3.0,
) -> ShrunkRiskModel:
    returns = build_intraday_log_return_matrix(
        tick_dict_by_date,
        stock_codes,
        price_col=price_col,
    )
    covariance, correlation, summary = build_ew_oas_covariance(
        returns,
        half_life_days=half_life_days,
    )
    summary["price_col"] = price_col
    return ShrunkRiskModel(
        covariance=covariance,
        correlation=correlation,
        returns=returns,
        summary=summary,
    )


def build_shrunk_risk_model_from_daily_loader(
    risk_dates: list[str],
    stock_codes: list[str],
    daily_tick_loader: Callable[[str], dict[str, pd.DataFrame | None]],
    *,
    price_col: str = "lastPrice",
    half_life_days: float = 3.0,
) -> ShrunkRiskModel:
    """Build a risk model while keeping only one raw tick dictionary in memory."""

    daily_return_frames = []
    for risk_date in risk_dates:
        tick_dict = daily_tick_loader(risk_date)
        daily_returns = build_intraday_log_return_matrix(
            {str(risk_date): tick_dict},
            stock_codes,
            price_col=price_col,
        )
        if daily_returns.empty:
            raise ValueError(f"Risk returns are empty for {risk_date}.")
        daily_return_frames.append(daily_returns)
        del tick_dict

    returns = pd.concat(daily_return_frames, axis=0).reindex(
        columns=[_normalize_stock_code(code) for code in stock_codes]
    )
    covariance, correlation, summary = build_ew_oas_covariance(
        returns,
        half_life_days=half_life_days,
    )
    summary["price_col"] = price_col
    summary["raw_tick_loading"] = "daily_streaming"
    return ShrunkRiskModel(
        covariance=covariance,
        correlation=correlation,
        returns=returns,
        summary=summary,
    )
