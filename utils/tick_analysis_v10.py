from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


PRICE_COL_ALIASES = {
    "lastPrice": ("lastPrice", "last_price", "last"),
    "bid1": ("bid1", "bidPrice1", "bid_price1", "bid_price_1"),
    "ask1": ("ask1", "askPrice1", "ask_price1", "ask_price_1"),
}


@dataclass
class MinuteTrackingResult:
    times: pd.Index
    basket_base_amount: float
    real_index_preclose: float
    real_index_base_price: float
    real_index_price_caliber: str
    real_index_amount_mode: str
    real_index_amount: pd.Series
    basket_amount: dict[str, pd.Series]
    diff_amount: pd.Series
    diff_ratio: pd.Series
    stats: dict
    real_index_frame: pd.DataFrame
    corporate_action_enabled: bool = False
    quantity_adjustments: pd.DataFrame | None = None
    dividend_records: pd.DataFrame | None = None
    daily_dividend_summary: pd.DataFrame | None = None


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


def _normalize_date_key(value) -> str:
    raw = str(value).strip()
    if re.fullmatch(r"\d{8}", raw):
        return raw
    return pd.Timestamp(value).strftime("%Y%m%d")


def _empty_action_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "stock_code",
        "ex_date",
        "cash_dividend_per_share",
        "share_increase_ratio",
        "source_cash_col",
        "source_share_cols",
    ])


def _numeric_col(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def standardize_corporate_actions(actions: pd.DataFrame | None) -> pd.DataFrame:
    """
    Normalize corporate-action rows for basket tracking.

    The function prefers XtQuant per-share fields when available:
    - interest: cash dividend per share
    - stockBonus + stockGift: stock dividend / transfer shares per original share

    Go-Goal fallback fields are commonly stored per 10 shares, so they are
    divided by 10 when the XtQuant field is unavailable.
    """
    if actions is None or actions.empty:
        return _empty_action_frame()
    if "stock_code" not in actions.columns or "ex_date" not in actions.columns:
        raise ValueError("corporate actions must contain stock_code and ex_date columns.")

    frame = actions.copy()
    frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
    frame["ex_date"] = frame["ex_date"].map(lambda value: _normalize_date_key(value) if pd.notna(value) else np.nan)
    frame = frame.dropna(subset=["stock_code", "ex_date"])
    if frame.empty:
        return _empty_action_frame()

    xt_cash = _numeric_col(frame, "interest", np.nan)
    gogoal_cash = _numeric_col(frame, "beftax_maxcashdiv", 0.0) / 10.0
    if "beftax_maxcashdiv" not in frame.columns and "beftax_mincashdiv" in frame.columns:
        gogoal_cash = _numeric_col(frame, "beftax_mincashdiv", 0.0) / 10.0
    cash = xt_cash.where(xt_cash.notna(), gogoal_cash).fillna(0.0)
    cash = cash.clip(lower=0.0)

    xt_bonus = _numeric_col(frame, "stockBonus", np.nan)
    xt_gift = _numeric_col(frame, "stockGift", np.nan)
    xt_share = xt_bonus.fillna(0.0) + xt_gift.fillna(0.0)
    xt_has_share = xt_bonus.notna() | xt_gift.notna()
    gogoal_share = (
        _numeric_col(frame, "stockdiv_ratio", 0.0)
        + _numeric_col(frame, "trans_ratio", 0.0)
        + _numeric_col(frame, "bonus_ratio", 0.0)
    ) / 10.0
    share_ratio = xt_share.where(xt_has_share, gogoal_share).fillna(0.0)
    share_ratio = share_ratio.clip(lower=0.0)

    out = pd.DataFrame({
        "stock_code": frame["stock_code"],
        "ex_date": frame["ex_date"],
        "cash_dividend_per_share": cash,
        "share_increase_ratio": share_ratio,
        "source_cash_col": np.where(xt_cash.notna(), "interest", "beftax_maxcashdiv/10"),
        "source_share_cols": np.where(xt_has_share, "stockBonus+stockGift", "stockdiv_ratio+trans_ratio+bonus_ratio / 10"),
    })
    out = out.groupby(["stock_code", "ex_date"], as_index=False).agg({
        "cash_dividend_per_share": "sum",
        "share_increase_ratio": "sum",
        "source_cash_col": lambda s: "|".join(sorted(set(map(str, s)))),
        "source_share_cols": lambda s: "|".join(sorted(set(map(str, s)))),
    })
    return out.sort_values(["ex_date", "stock_code"]).reset_index(drop=True)


def build_corporate_action_quantity_schedule(
    basket_quantities: dict[str, float],
    corporate_actions: pd.DataFrame | None,
    fitting_dates: list[str] | tuple[str, ...],
    *,
    baseline_date: str | None = None,
) -> dict:
    """
    Build actual holdings and dividend records for each fitting date.

    Corporate actions are applied before the market opens on ex_date.

    When baseline_date is provided, every action with
    baseline_date < ex_date <= max(fitting_dates) is applied in chronological
    order, even if ex_date itself is not in fitting_dates. The returned
    actual_quantities_by_date only contains requested fitting dates, but each
    date's quantities include all earlier unplotted ex-right adjustments.
    """
    base_quantities = {
        _normalize_stock_code(code): float(qty)
        for code, qty in basket_quantities.items()
        if float(qty) > 0
    }
    if not base_quantities:
        raise RuntimeError("basket_quantities is empty.")

    date_keys = sorted({_normalize_date_key(item) for item in fitting_dates})
    if not date_keys:
        raise RuntimeError("fitting_dates is empty.")

    start_exclusive = _normalize_date_key(baseline_date) if baseline_date is not None else None
    end_inclusive = max(date_keys)

    actions = standardize_corporate_actions(corporate_actions)
    actions = actions[actions["stock_code"].isin(base_quantities)].copy()
    if start_exclusive is not None:
        actions = actions[actions["ex_date"].gt(start_exclusive)]
    else:
        actions = actions[actions["ex_date"].ge(min(date_keys))]
    actions = actions[actions["ex_date"].le(end_inclusive)].copy()
    actions = actions.sort_values(["ex_date", "stock_code"]).reset_index(drop=True)

    current_qty = dict(base_quantities)
    base_by_date = {date_key: dict(base_quantities) for date_key in date_keys}
    actual_by_date: dict[str, dict[str, float]] = {}
    quantity_rows = []
    dividend_rows = []
    cumulative_dividend = 0.0

    date_keys_set = set(date_keys)
    event_dates = sorted(set(actions["ex_date"].tolist()) | date_keys_set)
    for date_key in event_dates:
        day_actions = actions.loc[actions["ex_date"].eq(date_key)].copy()
        if not day_actions.empty:
            for row in day_actions.itertuples(index=False):
                stock_code = row.stock_code
                before_qty = float(current_qty.get(stock_code, 0.0))
                cash_per_share = float(row.cash_dividend_per_share or 0.0)
                share_ratio = float(row.share_increase_ratio or 0.0)
                dividend_cash = before_qty * cash_per_share
                added_qty = before_qty * share_ratio
                after_qty = before_qty + added_qty
                current_qty[stock_code] = after_qty
                cumulative_dividend += dividend_cash

                if share_ratio > 0 or added_qty != 0:
                    quantity_rows.append({
                        "ex_date": date_key,
                        "stock_code": stock_code,
                        "before_qty": before_qty,
                        "share_increase_ratio": share_ratio,
                        "added_qty": added_qty,
                        "after_qty": after_qty,
                    })
                if cash_per_share > 0 or dividend_cash != 0:
                    dividend_rows.append({
                        "ex_date": date_key,
                        "stock_code": stock_code,
                        "holding_qty_for_dividend": before_qty,
                        "cash_dividend_per_share": cash_per_share,
                        "dividend_cash": dividend_cash,
                        "cumulative_dividend_cash": cumulative_dividend,
                    })
        if date_key in date_keys_set:
            actual_by_date[date_key] = dict(current_qty)

    quantity_adjustments = pd.DataFrame(quantity_rows, columns=[
        "ex_date",
        "stock_code",
        "before_qty",
        "share_increase_ratio",
        "added_qty",
        "after_qty",
    ])
    dividend_records = pd.DataFrame(dividend_rows, columns=[
        "ex_date",
        "stock_code",
        "holding_qty_for_dividend",
        "cash_dividend_per_share",
        "dividend_cash",
        "cumulative_dividend_cash",
    ])
    if dividend_records.empty:
        daily_dividend_summary = pd.DataFrame(columns=["ex_date", "daily_dividend_cash", "cumulative_dividend_cash"])
    else:
        daily_dividend_summary = (
            dividend_records.groupby("ex_date", as_index=False)["dividend_cash"].sum()
            .rename(columns={"dividend_cash": "daily_dividend_cash"})
            .sort_values("ex_date")
        )
        daily_dividend_summary["cumulative_dividend_cash"] = daily_dividend_summary["daily_dividend_cash"].cumsum()

    return {
        "base_quantities_by_date": base_by_date,
        "actual_quantities_by_date": actual_by_date,
        "standardized_actions": actions,
        "quantity_adjustments": quantity_adjustments,
        "dividend_records": dividend_records,
        "daily_dividend_summary": daily_dividend_summary,
    }


def _resolve_col(df: pd.DataFrame, logical_col: str) -> str | None:
    for col in PRICE_COL_ALIASES.get(logical_col, (logical_col,)):
        if col in df.columns:
            return col
    return None


def _hhmmss(t) -> str:
    return str(int(float(t)))[-6:]


def _in_session(t, morning_start: str = "093000", afternoon_start: str = "130000") -> bool:
    hhmmss = _hhmmss(t)
    return (morning_start <= hhmmss <= "113000") or (afternoon_start <= hhmmss <= "150000")


def _to_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    values = pd.Index(index).map(lambda x: str(int(float(x)))[:14])
    return pd.to_datetime(values, format="%Y%m%d%H%M%S")


def _first_positive(raw: pd.DataFrame | None, col: str) -> float | None:
    if raw is None or raw.empty or col not in raw.columns:
        return None
    values = pd.to_numeric(raw[col], errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        return None
    return float(values.iloc[0])


def _extract_price_series(raw: pd.DataFrame | None, logical_col: str) -> pd.Series | None:
    if raw is None or raw.empty or "time" not in raw.columns:
        return None

    price_col = _resolve_col(raw, logical_col)
    if price_col is None:
        return None

    df = raw.copy()
    df["_tick_time"] = df["time"].map(lambda t: int(float(t)))
    df = df[df["_tick_time"].map(_in_session)]
    df["_price"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[df["_price"] > 0]
    if df.empty:
        return None

    df = df.drop_duplicates(subset="_tick_time", keep="last").sort_values("_tick_time")
    series = df.set_index("_tick_time")["_price"]
    return series if not series.empty else None


def _compute_stats(
    diff_amount: pd.Series,
    diff_ratio: pd.Series,
    basket_base_amount: float,
    real_index_preclose: float,
) -> dict:
    valid_ratio = diff_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "basket_base_amount": float(basket_base_amount),
        "real_index_preclose": float(real_index_preclose) if pd.notna(real_index_preclose) else np.nan,
        "diff_mean_amount": float(diff_amount.mean()),
        "diff_std_amount": float(diff_amount.std()),
        "diff_max_positive_amount": float(diff_amount.max()),
        "diff_max_negative_amount": float(diff_amount.min()),
        "diff_mean_pct": float(valid_ratio.mean() * 100),
        "diff_std_pct": float(valid_ratio.std() * 100),
        "diff_max_positive_pct": float(valid_ratio.max() * 100),
        "diff_max_negative_pct": float(valid_ratio.min() * 100),
        "diff_pct95_pct": float(valid_ratio.abs().quantile(0.95) * 100),
        "n_minutes": int(len(valid_ratio)),
    }


def _prepare_real_index_minute_series(
    real_index_frame: pd.DataFrame,
    basket_base_amount: float,
    *,
    real_index_price_caliber: str = "close",
    real_index_amount_mode: str = "basket_base_scaled",
    previous_index_close: float | None = None,
) -> tuple[pd.Series, float, float, pd.DataFrame]:
    if real_index_frame is None or real_index_frame.empty:
        raise RuntimeError("real_index_frame is empty.")

    real_index_price_caliber = str(real_index_price_caliber).strip()
    valid_price_calibers = {"preClose", "open", "close"}
    if real_index_price_caliber not in valid_price_calibers:
        raise ValueError(
            f"real_index_price_caliber must be one of {sorted(valid_price_calibers)}, "
            f"got {real_index_price_caliber!r}."
        )

    real_index_amount_mode = str(real_index_amount_mode).strip()
    valid_amount_modes = {"basket_base_scaled"}
    if real_index_amount_mode not in valid_amount_modes:
        raise ValueError(
            f"real_index_amount_mode must be one of {sorted(valid_amount_modes)}, "
            f"got {real_index_amount_mode!r}."
        )

    required_cols = {real_index_price_caliber, "preClose"}
    missing_cols = [col for col in required_cols if col not in real_index_frame.columns]
    if missing_cols:
        raise ValueError(f"real_index_frame missing required columns: {missing_cols}.")

    frame = real_index_frame.copy()
    if "time" in frame.columns:
        time_values = frame["time"]
    else:
        time_values = frame.index.to_series()

    if pd.api.types.is_numeric_dtype(time_values):
        try:
            dt = pd.to_datetime(time_values.astype("int64"), unit="ms")
        except Exception:
            dt = pd.to_datetime(time_values.astype(str).str[:14], format="%Y%m%d%H%M%S")
    else:
        dt = pd.to_datetime(time_values)

    minute_dt = pd.Index(dt).floor("min")
    frame["_tick_time"] = minute_dt.strftime("%Y%m%d%H%M%S").astype("int64")
    frame["_real_index_price"] = pd.to_numeric(frame[real_index_price_caliber], errors="coerce")
    frame["preClose"] = pd.to_numeric(frame["preClose"], errors="coerce")
    frame = frame.dropna(subset=["_tick_time", "_real_index_price", "preClose"])
    frame = frame[(frame["_real_index_price"] > 0) & (frame["preClose"] > 0)]
    frame = frame[frame["_tick_time"].map(_in_session)]
    frame = frame.drop_duplicates(subset="_tick_time", keep="last").sort_values("_tick_time")

    if frame.empty:
        raise RuntimeError("real_index_frame has no valid minute rows.")

    preclose = float(frame["preClose"].iloc[0])
    index_price = frame["_real_index_price"].to_numpy(dtype=float)

    if previous_index_close is None or not np.isfinite(previous_index_close) or previous_index_close <= 0:
        raise ValueError("previous_index_close must be positive for basket_base_scaled mode.")
    base_price = float(previous_index_close)
    amount_values = index_price / base_price * float(basket_base_amount)

    amount = pd.Series(
        amount_values,
        index=pd.Index(frame["_tick_time"].astype("int64")),
        dtype=float,
    )
    amount = amount[~amount.index.duplicated(keep="last")]
    return amount, preclose, base_price, frame


def _align_stock_to_minutes(
    raw: pd.DataFrame | None,
    minute_times: pd.Index,
    price_col: str,
    fallback_price: float | None,
) -> pd.Series | None:
    series = _extract_price_series(raw, price_col)
    if series is None or series.empty:
        if fallback_price is None:
            return None
        return pd.Series(float(fallback_price), index=minute_times, dtype=float)

    aligned = series.reindex(minute_times, method="ffill")
    if fallback_price is not None:
        aligned = aligned.fillna(float(fallback_price))
    return aligned


def build_minute_tracking_analysis(
    tick_dict: dict[str, pd.DataFrame | None],
    real_index_frame: pd.DataFrame,
    basket_quantities: dict[str, float],
    basket_base_amount: float,
    *,
    price_cols: tuple[str, ...] = ("lastPrice", "bid1", "ask1"),
    basket_price_col: str = "lastPrice",
    real_index_price_caliber: str = "close",
    real_index_amount_mode: str = "basket_base_scaled",
    previous_index_close: float | None = None,
    corporate_action_enabled: bool = False,
    quantity_adjustments: pd.DataFrame | None = None,
    dividend_records: pd.DataFrame | None = None,
    daily_dividend_summary: pd.DataFrame | None = None,
) -> MinuteTrackingResult:
    """
    Compare one fitting date's real XtQuant 1m index series with a stock basket.

    The basket quantities must be constructed from previous_trade_date closes.
    The output still contains exactly one point per official index minute.
    """
    if basket_price_col not in price_cols:
        price_cols = tuple(dict.fromkeys((*price_cols, basket_price_col)))

    basket_quantities = {
        code: float(qty)
        for code, qty in basket_quantities.items()
        if float(qty) > 0
    }
    if not basket_quantities:
        raise RuntimeError("basket_quantities is empty.")
    if basket_base_amount <= 0:
        raise RuntimeError("basket_base_amount must be positive.")

    real_index_amount, real_index_preclose, real_index_base_price, clean_real_index_frame = _prepare_real_index_minute_series(
        real_index_frame=real_index_frame,
        basket_base_amount=float(basket_base_amount),
        real_index_price_caliber=real_index_price_caliber,
        real_index_amount_mode=real_index_amount_mode,
        previous_index_close=previous_index_close,
    )
    minute_times = real_index_amount.index

    basket_amount: dict[str, pd.Series] = {}
    for price_col in price_cols:
        total = pd.Series(0.0, index=minute_times, dtype=float)
        missing_codes = []

        for code, qty in basket_quantities.items():
            raw = tick_dict.get(code)
            fallback = _first_positive(raw, "lastClose")
            aligned_price = _align_stock_to_minutes(raw, minute_times, price_col, fallback)
            if aligned_price is None:
                missing_codes.append(code)
                continue
            total = total.add(aligned_price * qty, fill_value=0.0)

        if missing_codes:
            warnings.warn(f"{len(missing_codes)} basket stocks have no usable {price_col} tick and no lastClose fallback.")
        basket_amount[price_col] = total

    diff_amount = basket_amount[basket_price_col] - real_index_amount
    diff_ratio = diff_amount / real_index_amount.replace(0, np.nan)
    stats = _compute_stats(diff_amount, diff_ratio, float(basket_base_amount), real_index_preclose)
    stats["real_index_base_price"] = real_index_base_price
    stats["real_index_price_caliber"] = real_index_price_caliber
    stats["real_index_amount_mode"] = real_index_amount_mode
    stats["corporate_action_enabled"] = bool(corporate_action_enabled)
    if daily_dividend_summary is not None and not daily_dividend_summary.empty and "daily_dividend_cash" in daily_dividend_summary.columns:
        stats["daily_dividend_cash"] = float(pd.to_numeric(daily_dividend_summary["daily_dividend_cash"], errors="coerce").fillna(0).sum())
        if "cumulative_dividend_cash" in daily_dividend_summary.columns:
            stats["cumulative_dividend_cash"] = float(
                pd.to_numeric(daily_dividend_summary["cumulative_dividend_cash"], errors="coerce").dropna().iloc[-1]
            )
    else:
        stats["daily_dividend_cash"] = 0.0
        stats["cumulative_dividend_cash"] = 0.0

    return MinuteTrackingResult(
        times=minute_times,
        basket_base_amount=float(basket_base_amount),
        real_index_preclose=real_index_preclose,
        real_index_base_price=real_index_base_price,
        real_index_price_caliber=real_index_price_caliber,
        real_index_amount_mode=real_index_amount_mode,
        real_index_amount=real_index_amount,
        basket_amount=basket_amount,
        diff_amount=diff_amount,
        diff_ratio=diff_ratio,
        stats=stats,
        real_index_frame=clean_real_index_frame,
        corporate_action_enabled=bool(corporate_action_enabled),
        quantity_adjustments=quantity_adjustments,
        dividend_records=dividend_records,
        daily_dividend_summary=daily_dividend_summary,
    )


def combine_minute_tracking_results(results_by_date: dict[str, MinuteTrackingResult]) -> MinuteTrackingResult:
    """
    Concatenate daily MinuteTrackingResult objects into one total result.

    The daily objects are kept unchanged by the caller. The total result has the
    same structure, with stats computed from the full concatenated time span.
    """
    if not results_by_date:
        raise RuntimeError("results_by_date is empty.")

    ordered_items = sorted(results_by_date.items(), key=lambda item: item[0])
    first_result = ordered_items[0][1]
    basket_base_amount = first_result.basket_base_amount
    basket_base_amount_by_date = {
        date_key: float(result.basket_base_amount)
        for date_key, result in ordered_items
    }

    for date_key, result in ordered_items:
        if result.real_index_price_caliber != first_result.real_index_price_caliber:
            raise ValueError(f"real_index_price_caliber differs on {date_key}.")
        if result.real_index_amount_mode != first_result.real_index_amount_mode:
            raise ValueError(f"real_index_amount_mode differs on {date_key}.")

    times = pd.Index(np.concatenate([np.asarray(result.times, dtype="int64") for _, result in ordered_items]))
    real_index_amount = pd.concat([result.real_index_amount for _, result in ordered_items]).sort_index()

    all_price_cols = sorted({col for _, result in ordered_items for col in result.basket_amount})
    basket_amount = {
        col: pd.concat([
            result.basket_amount[col]
            for _, result in ordered_items
            if col in result.basket_amount
        ]).sort_index()
        for col in all_price_cols
    }
    if "lastPrice" not in basket_amount:
        raise RuntimeError("Cannot build total_result without lastPrice basket amount.")

    diff_amount = basket_amount["lastPrice"] - real_index_amount
    diff_ratio = diff_amount / real_index_amount.replace(0, np.nan)
    stats = _compute_stats(diff_amount, diff_ratio, basket_base_amount, first_result.real_index_preclose)
    stats["date_count"] = len(ordered_items)
    stats["fitting_dates"] = [date_key for date_key, _ in ordered_items]
    stats["daily_stats"] = {date_key: result.stats for date_key, result in ordered_items}
    stats["real_index_preclose_by_date"] = {
        date_key: result.real_index_preclose for date_key, result in ordered_items
    }
    stats["basket_base_amount_by_date"] = basket_base_amount_by_date
    stats["basket_base_amount_changes"] = len(set(round(v, 6) for v in basket_base_amount_by_date.values()))
    stats["real_index_base_price"] = first_result.real_index_base_price
    stats["real_index_price_caliber"] = first_result.real_index_price_caliber
    stats["real_index_amount_mode"] = first_result.real_index_amount_mode
    stats["corporate_action_enabled"] = bool(first_result.corporate_action_enabled)

    quantity_adjustment_frames = []
    dividend_record_frames = []
    dividend_summary_frames = []
    for date_key, result in ordered_items:
        if result.quantity_adjustments is not None and not result.quantity_adjustments.empty:
            frame = result.quantity_adjustments.copy()
            frame["fitting_date"] = date_key
            quantity_adjustment_frames.append(frame)
        if result.dividend_records is not None and not result.dividend_records.empty:
            frame = result.dividend_records.copy()
            frame["fitting_date"] = date_key
            dividend_record_frames.append(frame)
        if result.daily_dividend_summary is not None and not result.daily_dividend_summary.empty:
            frame = result.daily_dividend_summary.copy()
            frame["fitting_date"] = date_key
            dividend_summary_frames.append(frame)

    quantity_adjustments = (
        pd.concat(quantity_adjustment_frames, ignore_index=True)
        if quantity_adjustment_frames else pd.DataFrame()
    )
    dividend_records = (
        pd.concat(dividend_record_frames, ignore_index=True)
        if dividend_record_frames else pd.DataFrame()
    )
    daily_dividend_summary = (
        pd.concat(dividend_summary_frames, ignore_index=True)
        if dividend_summary_frames else pd.DataFrame(columns=["ex_date", "daily_dividend_cash", "cumulative_dividend_cash", "fitting_date"])
    )
    stats["daily_dividend_cash"] = float(pd.to_numeric(daily_dividend_summary.get("daily_dividend_cash", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    if not daily_dividend_summary.empty and "cumulative_dividend_cash" in daily_dividend_summary.columns:
        stats["cumulative_dividend_cash"] = float(
            pd.to_numeric(daily_dividend_summary["cumulative_dividend_cash"], errors="coerce").dropna().iloc[-1]
        )
    else:
        stats["cumulative_dividend_cash"] = 0.0

    frames = []
    for date_key, result in ordered_items:
        frame = result.real_index_frame.copy()
        frame["fitting_date"] = date_key
        frames.append(frame)
    real_index_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return MinuteTrackingResult(
        times=times,
        basket_base_amount=basket_base_amount,
        real_index_preclose=first_result.real_index_preclose,
        real_index_base_price=first_result.real_index_base_price,
        real_index_price_caliber=first_result.real_index_price_caliber,
        real_index_amount_mode=first_result.real_index_amount_mode,
        real_index_amount=real_index_amount,
        basket_amount=basket_amount,
        diff_amount=diff_amount,
        diff_ratio=diff_ratio,
        stats=stats,
        real_index_frame=real_index_frame,
        corporate_action_enabled=bool(first_result.corporate_action_enabled),
        quantity_adjustments=quantity_adjustments,
        dividend_records=dividend_records,
        daily_dividend_summary=daily_dividend_summary,
    )


def _compressed_axis(times: pd.Index):
    dt = _to_datetime_index(times)
    x = np.arange(len(dt), dtype=float)
    labels = pd.Index(dt).strftime("%Y-%m-%d %H:%M")

    tick_positions = []
    tick_labels = []
    day_close_positions = []
    by_day: dict[str, list[int]] = {}
    for pos, value in enumerate(dt):
        by_day.setdefault(value.strftime("%Y%m%d"), []).append(pos)

    ordered_days = list(by_day.items())
    for day_no, (date_key, positions) in enumerate(ordered_days):
        day_dt = dt[positions]
        date_label = pd.Timestamp(day_dt[0]).strftime("%Y-%m-%d")
        for hhmm, suffix in (("09:30", "09:30"), ("11:30", "11:30"), ("15:00", "15:00")):
            matches = [positions[i] for i, item in enumerate(day_dt) if item.strftime("%H:%M") == hhmm]
            if matches:
                # Keep the compressed axis readable near day boundaries: every
                # day gets an open label, midday is sparse context, and only the
                # final close is labelled. Intermediate closes are still drawn
                # as vertical separators.
                if hhmm == "09:30":
                    tick_positions.append(matches[0])
                    tick_labels.append(f"{date_label}\n{suffix}")
                elif hhmm == "11:30":
                    tick_positions.append(matches[0])
                    tick_labels.append(suffix)
                elif day_no == len(ordered_days) - 1:
                    tick_positions.append(matches[0])
                    tick_labels.append(suffix)
                if hhmm == "15:00":
                    day_close_positions.append((matches[0], date_label))
        if not any(dt[pos].strftime("%H:%M") == "15:00" for pos in positions):
            day_close_positions.append((positions[-1], date_label))

    return x, labels, tick_positions, tick_labels, day_close_positions


def _amount_formatter(x, _pos=None):
    if abs(x) >= 1e8:
        return f"{x / 1e8:.2f}e8"
    if abs(x) >= 1e4:
        return f"{x / 1e4:.0f}w"
    return f"{x:,.0f}"


def plot_tracking_matplotlib(result: MinuteTrackingResult, index_name: str = ""):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False

    x, _labels, tick_positions, tick_labels, day_close_positions = _compressed_axis(result.times)
    base = result.basket_base_amount
    index_pct = (result.real_index_amount.reindex(result.times) / base - 1.0) * 100
    basket_pct = (result.basket_amount["lastPrice"].reindex(result.times) / base - 1.0) * 100

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1]},
    )

    if "bid1" in result.basket_amount and "ask1" in result.basket_amount:
        bid_pct = (result.basket_amount["bid1"].reindex(result.times) / base - 1.0) * 100
        ask_pct = (result.basket_amount["ask1"].reindex(result.times) / base - 1.0) * 100
        low = np.fmin(bid_pct.values, ask_pct.values)
        high = np.fmax(bid_pct.values, ask_pct.values)
        ax1.fill_between(x, low, high, color="#ff7f0e", alpha=0.14, label="Basket bid1-ask1 band")
        ax1.plot(x, bid_pct.values, linewidth=0.8, linestyle="--", color="#2ca02c", label="Basket bid1")
        ax1.plot(x, ask_pct.values, linewidth=0.8, linestyle="--", color="#d62728", label="Basket ask1")

    ax1.plot(
        x,
        index_pct.values,
        linewidth=1.3,
        color="#1f77b4",
        label=f"Real index 1m {result.real_index_price_caliber} ({result.real_index_amount_mode})",
    )
    ax1.plot(x, basket_pct.values, linewidth=1.1, color="#ff7f0e", label="Basket 1m lastPrice")
    ax1.set_ylabel("Return from previous close base (%)")
    ax1.set_title(f"Real Index vs Basket Minute Tracking {index_name}")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    for pos, _date_label in day_close_positions:
        ax1.axvline(pos, color="gray", linewidth=0.8, linestyle="--", alpha=0.55)
        ax2.axvline(pos, color="gray", linewidth=0.8, linestyle="--", alpha=0.55)

    def pct_to_amount(y):
        return base * (1.0 + np.asarray(y) / 100.0)

    def amount_to_pct(y):
        return (np.asarray(y) / base - 1.0) * 100.0

    ax1_right = ax1.secondary_yaxis("right", functions=(pct_to_amount, amount_to_pct))
    ax1_right.set_ylabel("Amount")
    ax1_right.yaxis.set_major_formatter(FuncFormatter(_amount_formatter))

    diff_pct = result.diff_ratio.reindex(result.times) * 100
    ax2.fill_between(x, 0, diff_pct.values, alpha=0.22, color="#d62728")
    ax2.plot(x, diff_pct.values, linewidth=0.8, color="#d62728", label="Basket - real index")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    sigma = result.stats["diff_std_pct"]
    ax2.axhline(y=3 * sigma, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax2.axhline(y=-3 * sigma, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax2.set_ylabel("Deviation (%)")
    ax2.set_xlabel("Compressed trading time")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    textstr = (
        f"Mean: {result.stats['diff_mean_pct']:.4f}%\n"
        f"Std:  {result.stats['diff_std_pct']:.4f}%\n"
        f"Max+: {result.stats['diff_max_positive_pct']:.4f}%\n"
        f"Max-: {result.stats['diff_max_negative_pct']:.4f}%\n"
        f"95%:  {result.stats['diff_pct95_pct']:.4f}%\n"
        f"Minutes: {result.stats['n_minutes']}\n"
        f"Days: {result.stats.get('date_count', 1)}\n"
        f"Base: {base:,.0f}\n"
        f"Idx px: {result.real_index_price_caliber}\n"
        f"Mode: {result.real_index_amount_mode}\n"
        f"CorpAct: {result.corporate_action_enabled}\n"
        f"Div: {result.stats.get('cumulative_dividend_cash', 0.0):,.0f}"
    )
    ax1.text(
        0.98,
        0.97,
        textstr,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
        family="monospace",
    )

    fig.subplots_adjust(left=0.07, right=0.92, top=0.92, bottom=0.14, hspace=0.08)
    return fig


def plot_tracking_plotly(
    result: MinuteTrackingResult,
    index_name: str = "",
    *,
    html_path: str | Path | None = None,
    auto_open: bool = False,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("plotly is required for interactive tracking chart.") from exc

    x, labels, tick_positions, tick_labels, day_close_positions = _compressed_axis(result.times)
    base = result.basket_base_amount
    index_amount = result.real_index_amount.reindex(result.times)
    basket_amount = result.basket_amount["lastPrice"].reindex(result.times)
    index_pct = (index_amount / base - 1.0) * 100
    basket_pct = (basket_amount / base - 1.0) * 100
    diff_pct = result.diff_ratio.reindex(result.times) * 100

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=("", "Deviation and amount gap"),
    )

    if "bid1" in result.basket_amount and "ask1" in result.basket_amount:
        bid_pct = (result.basket_amount["bid1"].reindex(result.times) / base - 1.0) * 100
        ask_pct = (result.basket_amount["ask1"].reindex(result.times) / base - 1.0) * 100
        low = np.fmin(bid_pct.values, ask_pct.values)
        high = np.fmax(bid_pct.values, ask_pct.values)

        fig.add_trace(go.Scatter(x=x, y=low, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=high, mode="lines", name="Basket bid1-ask1 band",
                                 fill="tonexty", fillcolor="rgba(255,127,14,0.16)", line=dict(width=0),
                                 customdata=np.column_stack([labels]),
                                 hovertemplate="Time=%{customdata[0]}<extra></extra>"),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=bid_pct, mode="lines", name="Basket bid1",
                                 customdata=np.column_stack([labels]),
                                 hovertemplate="Time=%{customdata[0]}<br>Return=%{y:.4f}%<extra></extra>",
                                 line=dict(color="#2ca02c", width=1, dash="dash")),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=x, y=ask_pct, mode="lines", name="Basket ask1",
                                 customdata=np.column_stack([labels]),
                                 hovertemplate="Time=%{customdata[0]}<br>Return=%{y:.4f}%<extra></extra>",
                                 line=dict(color="#d62728", width=1, dash="dash")),
                      row=1, col=1, secondary_y=False)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=index_pct,
            mode="lines",
            name=f"Real index 1m {result.real_index_price_caliber} ({result.real_index_amount_mode})",
            customdata=list(zip(labels, index_amount.values)),
            hovertemplate="Time=%{customdata[0]}<br>Return=%{y:.4f}%<br>Amount=%{customdata[1]:,.2f}<extra></extra>",
            line=dict(color="#1f77b4", width=1.4),
        ),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=basket_pct,
            mode="lines",
            name="Basket 1m lastPrice",
            customdata=list(zip(labels, basket_amount.values)),
            hovertemplate="Time=%{customdata[0]}<br>Return=%{y:.4f}%<br>Amount=%{customdata[1]:,.2f}<extra></extra>",
            line=dict(color="#ff7f0e", width=1.2),
        ),
        row=1, col=1, secondary_y=False,
    )

    fig.add_trace(go.Scatter(x=x, y=index_amount, mode="lines", showlegend=False,
                             hoverinfo="skip", line=dict(width=0), opacity=0),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=basket_amount, mode="lines", showlegend=False,
                             hoverinfo="skip", line=dict(width=0), opacity=0),
                  row=1, col=1, secondary_y=True)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=diff_pct,
            mode="lines",
            name="Deviation %",
            customdata=list(zip(labels, result.diff_amount.reindex(result.times).values)),
            hovertemplate="Time=%{customdata[0]}<br>Deviation=%{y:.4f}%<br>Amount gap=%{customdata[1]:,.2f}<extra></extra>",
            line=dict(color="#d62728", width=1),
        ),
        row=2, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=result.diff_amount.reindex(result.times),
            name="Amount gap",
            customdata=np.column_stack([labels]),
            marker_color="rgba(214,39,40,0.28)",
            hovertemplate="Time=%{customdata[0]}<br>Amount gap=%{y:,.2f}<extra></extra>",
        ),
        row=2, col=1, secondary_y=True,
    )

    for pos, date_label in day_close_positions:
        fig.add_vline(
            x=pos,
            line_width=1,
            line_dash="dash",
            line_color="rgba(120,120,120,0.7)",
        )

    fig.add_hline(y=0, line_width=1, line_color="black", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Return from previous close base (%)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Amount", row=1, col=1, secondary_y=True, tickformat=",.0f")
    fig.update_yaxes(title_text="Deviation (%)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Amount gap", row=2, col=1, secondary_y=True, tickformat=",.0f")
    fig.update_xaxes(
        title_text="Compressed trading time",
        row=2,
        col=1,
        rangeslider_visible=True,
        tickmode="array",
        tickvals=tick_positions,
        ticktext=tick_labels,
    )
    fig.update_layout(
        title=dict(text=f"Real Index vs Basket Minute Tracking {index_name}", x=0.5, y=0.98),
        height=840,
        width=1350,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.07, xanchor="left", x=0),
        margin=dict(l=70, r=85, t=120, b=80),
    )

    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path), include_plotlyjs="cdn", auto_open=auto_open)

    return fig


def _date_label(fitting_dates: str | list[str] | tuple[str, ...]) -> str:
    if isinstance(fitting_dates, str):
        return fitting_dates
    values = [str(item) for item in fitting_dates]
    if not values:
        return "unknown"
    if len(values) == 1:
        return values[0]
    return f"{values[0]}_{values[-1]}_{len(values)}d"


def _curve_frame(result: MinuteTrackingResult) -> pd.DataFrame:
    time_values = pd.Index(result.times).astype("int64").astype(str)
    frame = pd.DataFrame({
        "time": result.times,
        "fitting_date": time_values.str[:8],
        "real_index_price_caliber": result.real_index_price_caliber,
        "real_index_amount_mode": result.real_index_amount_mode,
        "real_index_base_price": result.real_index_base_price,
        "real_index_amount": result.real_index_amount.reindex(result.times).to_numpy(),
        "basket_lastPrice_amount": result.basket_amount["lastPrice"].reindex(result.times).to_numpy(),
        "basket_bid1_amount": result.basket_amount.get("bid1", pd.Series(index=result.times, dtype=float)).reindex(result.times).to_numpy(),
        "basket_ask1_amount": result.basket_amount.get("ask1", pd.Series(index=result.times, dtype=float)).reindex(result.times).to_numpy(),
        "diff_amount": result.diff_amount.reindex(result.times).to_numpy(),
        "diff_ratio": result.diff_ratio.reindex(result.times).to_numpy(),
        "corporate_action_enabled": result.corporate_action_enabled,
    })
    if result.daily_dividend_summary is not None and not result.daily_dividend_summary.empty:
        div = result.daily_dividend_summary.copy()
        if "ex_date" in div.columns:
            div["fitting_date"] = div["ex_date"].astype(str).str.replace("-", "", regex=False)
        div = div.drop_duplicates(subset=["fitting_date"], keep="last")
        frame = frame.merge(
            div[["fitting_date", "daily_dividend_cash", "cumulative_dividend_cash"]],
            on="fitting_date",
            how="left",
        )
        frame[["daily_dividend_cash", "cumulative_dividend_cash"]] = frame[["daily_dividend_cash", "cumulative_dividend_cash"]].fillna(0.0)
    else:
        frame["daily_dividend_cash"] = 0.0
        frame["cumulative_dividend_cash"] = 0.0
    return frame


def _stats_frame(result: MinuteTrackingResult, daily_results: dict[str, MinuteTrackingResult] | None = None) -> pd.DataFrame:
    rows = [{"scope": "total", "fitting_date": "ALL", **result.stats}]
    if daily_results:
        for date_key, daily_result in sorted(daily_results.items()):
            rows.append({"scope": "daily", "fitting_date": date_key, **daily_result.stats})
    frame = pd.DataFrame(rows)
    for col in ("daily_stats", "real_index_preclose_by_date", "basket_base_amount_by_date", "fitting_dates"):
        if col in frame.columns:
            frame = frame.drop(columns=[col])
    return frame


def save_tracking_outputs(
    result: MinuteTrackingResult,
    output_dir: str | Path,
    fitting_dates: str | list[str] | tuple[str, ...],
    index_name: str = "",
    *,
    daily_results: dict[str, MinuteTrackingResult] | None = None,
) -> dict:
    
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = _date_label(fitting_dates)
    curve_path = output_dir / f"minute_tracking_v10_{label}.csv"
    _curve_frame(result).to_csv(curve_path, index=False, encoding="utf-8-sig")

    stats_path = output_dir / f"minute_tracking_stats_v10_{label}.csv"
    _stats_frame(result, daily_results).to_csv(stats_path, index=False, encoding="utf-8-sig")

    quantity_adjustments_path = output_dir / f"minute_tracking_quantity_adjustments_v10_{label}.csv"
    dividend_records_path = output_dir / f"minute_tracking_dividends_v10_{label}.csv"
    dividend_summary_path = output_dir / f"minute_tracking_dividend_summary_v10_{label}.csv"
    if result.quantity_adjustments is not None:
        result.quantity_adjustments.to_csv(quantity_adjustments_path, index=False, encoding="utf-8-sig")
    if result.dividend_records is not None:
        result.dividend_records.to_csv(dividend_records_path, index=False, encoding="utf-8-sig")
    if result.daily_dividend_summary is not None:
        result.daily_dividend_summary.to_csv(dividend_summary_path, index=False, encoding="utf-8-sig")

    matplotlib_fig = plot_tracking_matplotlib(result, index_name)

    plt.close(matplotlib_fig)

    html_path = output_dir / f"minute_tracking_v10_{label}.html"
    plotly_fig = plot_tracking_plotly(result, index_name, html_path=html_path, auto_open=False)

    return {
        "curve_path": curve_path,
        "stats_path": stats_path,
        "quantity_adjustments_path": quantity_adjustments_path,
        "dividend_records_path": dividend_records_path,
        "dividend_summary_path": dividend_summary_path,
        "html_path": html_path,
        "matplotlib_fig": matplotlib_fig,
        "plotly_fig": plotly_fig,
    }

