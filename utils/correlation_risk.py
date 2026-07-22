from __future__ import annotations

import numpy as np
import pandas as pd


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
        tick_time_raw = pd.to_numeric(df["time"], errors="coerce")
    else:
        tick_time_raw = pd.Series(df.index, index=df.index)
        if np.issubdtype(tick_time_raw.dtype, np.datetime64):
            tick_time_raw = pd.to_datetime(tick_time_raw).strftime("%H%M%S")
        tick_time_raw = pd.to_numeric(tick_time_raw, errors="coerce")

    price = pd.to_numeric(df[price_col], errors="coerce")
    valid = tick_time_raw.notna() & price.notna() & (price > 0)
    if not valid.any():
        return pd.Series(dtype=float)

    hhmmss = tick_time_raw.loc[valid].astype("int64").map(lambda value: int(str(value)[-6:]))
    minute = (hhmmss // 100).astype(int)
    series = pd.Series(price.loc[valid].to_numpy(dtype=float), index=minute)
    series = series[series.index.map(lambda x: 930 <= x <= 1130 or 1300 <= x <= 1500)]
    if series.empty:
        return pd.Series(dtype=float)
    return series.groupby(level=0).last().sort_index()


def build_intraday_log_return_matrix(
    tick_dict_by_date: dict[str, dict[str, pd.DataFrame | None]],
    stock_codes: list[str],
    *,
    price_col: str = "lastPrice",
) -> pd.DataFrame:
    stock_codes = [_normalize_stock_code(code) for code in stock_codes]
    return_frames = []

    for date_key, tick_dict in sorted(tick_dict_by_date.items()):
        price_columns = {}
        for code in stock_codes:
            raw = None if tick_dict is None else tick_dict.get(code)
            series = _extract_minute_price_series(raw, price_col)
            if not series.empty:
                price_columns[code] = series

        if not price_columns:
            continue

        price_frame = pd.DataFrame(price_columns).sort_index()
        price_frame = price_frame.reindex(columns=stock_codes).ffill()
        log_return = np.log(price_frame).diff().iloc[1:]
        if not log_return.empty:
            log_return.index = [f"{date_key}_{int(item):04d}" for item in log_return.index]
            return_frames.append(log_return)

    if not return_frames:
        return pd.DataFrame(columns=stock_codes)

    returns = pd.concat(return_frames, axis=0)
    return returns.reindex(columns=stock_codes)


def build_risk_matrix_from_returns(
    returns: pd.DataFrame,
    *,
    mode: str = "covariance",
) -> pd.DataFrame:
    mode = str(mode).strip().lower()
    if mode not in {"covariance", "correlation"}:
        raise ValueError("mode must be 'covariance' or 'correlation'.")
    if returns.empty:
        raise ValueError("returns is empty; cannot build risk matrix.")

    clean_returns = returns.ffill()
    if mode == "covariance":
        matrix = clean_returns.cov()
    else:
        matrix = clean_returns.corr()

    matrix = matrix.reindex(index=returns.columns, columns=returns.columns)
    matrix = matrix.fillna(0.0)
    values = matrix.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(values, np.where(np.isfinite(np.diag(values)), np.diag(values), 0.0))
    if mode == "correlation":
        np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)


def build_risk_matrix_from_tick_dicts(
    tick_dict_by_date: dict[str, dict[str, pd.DataFrame | None]],
    stock_codes: list[str],
    *,
    price_col: str = "lastPrice",
    mode: str = "covariance",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = build_intraday_log_return_matrix(
        tick_dict_by_date,
        stock_codes,
        price_col=price_col,
    )
    matrix = build_risk_matrix_from_returns(returns, mode=mode)
    return matrix, returns
