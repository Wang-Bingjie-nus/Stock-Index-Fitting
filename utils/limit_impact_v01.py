from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .minute_tick_cache_v03 import (
    get_minute_field,
    normalize_stock_code,
    normalize_stock_codes,
    normalize_trade_date,
    validate_minute_tick_cache,
)


XT_LIMIT_META_ID = 9506
XT_DAILY_PERIOD_MS = 86_400_000
OPENING_MINUTE = "093100"

TASK11_DISPLAY_COLUMNS = [
    "basket",
    "held_stock_count",
    "basket_build_amount",
    "basket_amount_0931",
    "common_base_gap_amount",
    "common_base_gap_pct",
    "return_gap_pct",
    "cash_adjusted_gap_pct",
]


def normalize_construction_date_list(construction_dates) -> list[str]:
    """Normalize a non-empty, duplicate-free batch of construction dates."""

    if isinstance(construction_dates, (str, bytes)):
        raise TypeError(
            "construction_date_ls must be a sequence of dates, not a single string."
        )
    try:
        raw_dates = list(construction_dates)
    except TypeError as exc:
        raise TypeError("construction_date_ls must be an iterable of dates.") from exc
    if not raw_dates:
        raise ValueError("construction_date_ls cannot be empty.")

    normalized = [normalize_trade_date(value) for value in raw_dates]
    duplicated = sorted({date for date in normalized if normalized.count(date) > 1})
    if duplicated:
        raise ValueError(f"construction_date_ls contains duplicate dates: {duplicated}")
    return normalized


def combine_task11_summaries(
    summaries: Mapping[str, pd.DataFrame],
    construction_dates,
) -> pd.DataFrame:
    """Combine per-date Task11 results into one ordered 2N-by-9 table."""

    dates = normalize_construction_date_list(construction_dates)
    normalized_summaries = {
        normalize_trade_date(date): frame for date, frame in summaries.items()
    }
    missing_dates = [date for date in dates if date not in normalized_summaries]
    extra_dates = sorted(set(normalized_summaries) - set(dates))
    if missing_dates or extra_dates:
        raise ValueError(
            "Task11 summary dates do not match construction_date_ls: "
            f"missing={missing_dates}, extra={extra_dates}"
        )

    expected_baskets = ["basket1", "basket3"]
    ordered_frames = []
    for date in dates:
        frame = normalized_summaries[date].copy()
        missing_columns = sorted(set(TASK11_DISPLAY_COLUMNS) - set(frame.columns))
        if missing_columns:
            raise ValueError(
                f"Task11 summary for {date} is missing columns: {missing_columns}"
            )
        if "construction_date" in frame.columns:
            reported_dates = {
                normalize_trade_date(value)
                for value in frame["construction_date"].dropna().unique()
            }
            if reported_dates != {date}:
                raise ValueError(
                    f"Task11 summary for {date} reports dates {sorted(reported_dates)}."
                )
            frame["construction_date"] = date
        else:
            frame.insert(0, "construction_date", date)

        baskets = frame["basket"].astype(str).tolist()
        if sorted(baskets) != expected_baskets or len(frame) != 2:
            raise ValueError(
                f"Task11 summary for {date} must contain basket1/3 exactly once; "
                f"got {baskets}."
            )
        basket_order = {name: order for order, name in enumerate(expected_baskets)}
        frame["_basket_order"] = frame["basket"].map(basket_order)
        frame = frame.sort_values("_basket_order").drop(columns="_basket_order")
        ordered_frames.append(frame[["construction_date", *TASK11_DISPLAY_COLUMNS]])

    combined = pd.concat(ordered_frames, ignore_index=True)
    if len(combined) != 2 * len(dates):
        raise AssertionError("Combined Task11 row count is inconsistent.")
    return combined


def _calendar_item_to_trade_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8 and digits[:4].isdigit():
        candidate = digits[:8]
        try:
            return normalize_trade_date(candidate)
        except ValueError:
            pass
    try:
        numeric = int(float(text))
    except (TypeError, ValueError, OverflowError):
        return None
    unit = "ms" if abs(numeric) >= 10**11 else "s"
    try:
        stamp = pd.to_datetime(numeric, unit=unit, utc=True).tz_convert("Asia/Shanghai")
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
        return None
    return stamp.strftime("%Y%m%d")


def get_next_trading_date(
    xtdata_client: Any,
    construction_date: str,
    *,
    market: str = "SH",
    search_calendar_days: int = 31,
) -> str:
    """Return the first XtQuant trading date strictly after construction_date."""

    construction_date = normalize_trade_date(construction_date)
    end_date = (
        pd.Timestamp(construction_date) + pd.Timedelta(days=int(search_calendar_days))
    ).strftime("%Y%m%d")
    calendar = xtdata_client.get_trading_calendar(
        market,
        start_time=construction_date,
        end_time=end_date,
    )
    dates = sorted(
        {
            date
            for date in (
                _calendar_item_to_trade_date(item) for item in (calendar or [])
            )
            if date is not None and date > construction_date
        }
    )
    if not dates:
        raise RuntimeError(
            f"XtQuant returned no {market} trading date after {construction_date}."
        )
    return dates[0]


def get_recent_trading_dates(
    xtdata_client: Any,
    end_date: str,
    count: int,
    *,
    market: str = "SH",
    lookback_calendar_days: int = 366,
) -> list[str]:
    """Return the final ``count`` market dates ending at or before end_date."""

    end_date = normalize_trade_date(end_date)
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive.")
    start_date = (
        pd.Timestamp(end_date) - pd.Timedelta(days=int(lookback_calendar_days))
    ).strftime("%Y%m%d")
    calendar = xtdata_client.get_trading_calendar(
        market,
        start_time=start_date,
        end_time=end_date,
    )
    dates = sorted(
        {
            date
            for date in (
                _calendar_item_to_trade_date(item) for item in (calendar or [])
            )
            if date is not None and date <= end_date
        }
    )
    if len(dates) < count:
        raise RuntimeError(
            f"XtQuant returned only {len(dates)} {market} trading dates through "
            f"{end_date}; need {count}."
        )
    return dates[-count:]


def expand_construction_date_list(
    xtdata_client: Any,
    construction_dates,
    *,
    market: str = "SH",
) -> list[str]:
    """Expand two date boundaries to all market dates in the closed interval.

    A list whose length is not two remains an explicit construction-date list.
    When exactly two values are supplied, they are interpreted as the start and
    end of an observation interval. Non-trading boundary dates are allowed and
    are naturally omitted from the returned trading-date list.
    """

    dates = normalize_construction_date_list(construction_dates)
    if len(dates) != 2:
        return dates

    start_date, end_date = dates
    if start_date > end_date:
        raise ValueError(
            "construction_date_ls range start must not be after range end: "
            f"{start_date} > {end_date}"
        )

    calendar = xtdata_client.get_trading_calendar(
        market,
        start_time=start_date,
        end_time=end_date,
    )
    expanded = sorted(
        {
            date
            for date in (
                _calendar_item_to_trade_date(item) for item in (calendar or [])
            )
            if date is not None and start_date <= date <= end_date
        }
    )
    if not expanded:
        raise RuntimeError(
            f"XtQuant returned no {market} trading dates in "
            f"[{start_date}, {end_date}]."
        )
    return expanded


def _iter_chunks(values: list[str], chunk_size: int):
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def fetch_xt_historical_limit_prices(
    xtdata_client: Any,
    stock_codes,
    trade_date: str,
    *,
    chunk_size: int = 500,
    meta_id: int = XT_LIMIT_META_ID,
    period_ms: int = XT_DAILY_PERIOD_MS,
) -> pd.DataFrame:
    """Read historical daily up/down-limit prices from XtQuant metadata 9506.

    ``get_instrument_detail`` only represents the current trading day.  The
    tuple period ``(9506, 86400000)`` is used here so historical D+1 studies do
    not accidentally use today's contract limits.
    """

    trade_date = normalize_trade_date(trade_date)
    codes = normalize_stock_codes(stock_codes)
    if not codes:
        raise ValueError("stock_codes is empty.")

    rows = []
    for batch in _iter_chunks(codes, chunk_size):
        result = xtdata_client.get_market_data_ex(
            [],
            batch,
            period=(int(meta_id), int(period_ms)),
            start_time=trade_date,
            end_time=trade_date,
            count=-1,
            fill_data=False,
        )
        result = result or {}
        for code in batch:
            frame = result.get(code)
            up_price = np.nan
            down_price = np.nan
            if frame is not None and not frame.empty:
                up_col = _resolve_column(
                    frame,
                    ("涨停价", "e", "UpStopPrice", "upStopPrice", "upperLimit"),
                )
                down_col = _resolve_column(
                    frame,
                    ("跌停价", "f", "DownStopPrice", "downStopPrice", "lowerLimit"),
                )
                if up_col is None or down_col is None:
                    raise ValueError(
                        f"Unexpected XtQuant 9506 columns for {code}: "
                        f"{frame.columns.tolist()}"
                    )
                up_values = pd.to_numeric(frame[up_col], errors="coerce").dropna()
                down_values = pd.to_numeric(frame[down_col], errors="coerce").dropna()
                if not up_values.empty:
                    up_price = float(up_values.iloc[-1])
                if not down_values.empty:
                    down_price = float(down_values.iloc[-1])
            rows.append(
                {
                    "stock_code": code,
                    "status_date": trade_date,
                    "up_limit_price": up_price,
                    "down_limit_price": down_price,
                }
            )
    return pd.DataFrame(rows)


def fetch_xt_daily_suspend_status(
    xtdata_client: Any,
    stock_codes,
    trade_date: str,
    *,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Read historical D+1 ``suspendFlag`` from XtQuant unadjusted daily bars."""

    trade_date = normalize_trade_date(trade_date)
    codes = normalize_stock_codes(stock_codes)
    if not codes:
        raise ValueError("stock_codes is empty.")

    rows = []
    for batch in _iter_chunks(codes, chunk_size):
        result = xtdata_client.get_market_data_ex(
            ["suspendFlag", "close"],
            batch,
            period="1d",
            start_time=trade_date,
            end_time=trade_date,
            count=-1,
            dividend_type="none",
            fill_data=False,
        )
        result = result or {}
        for code in batch:
            frame = result.get(code)
            suspend_flag = np.nan
            daily_close = np.nan
            if frame is not None and not frame.empty:
                if "suspendFlag" not in frame.columns:
                    raise ValueError(
                        f"XtQuant 1d data has no suspendFlag for {code}: "
                        f"{frame.columns.tolist()}"
                    )
                suspend_values = pd.to_numeric(
                    frame["suspendFlag"], errors="coerce"
                ).dropna()
                close_values = pd.to_numeric(
                    frame.get("close", pd.Series(dtype=float)), errors="coerce"
                ).dropna()
                if not suspend_values.empty:
                    suspend_flag = float(suspend_values.iloc[-1])
                if not close_values.empty:
                    daily_close = float(close_values.iloc[-1])
            rows.append(
                {
                    "stock_code": code,
                    "status_date": trade_date,
                    "suspend_flag": suspend_flag,
                    "daily_close": daily_close,
                }
            )
    return pd.DataFrame(rows)


def fetch_xt_historical_trading_status(
    xtdata_client: Any,
    stock_codes,
    trade_date: str,
    *,
    chunk_size: int = 500,
    download_missing: bool = True,
    download_chunk_size: int = 50,
) -> pd.DataFrame:
    """Return historical limit prices and suspension flags for the full universe.

    XtQuant only returns historical metadata already present in the local quote
    store.  On the first miss this function downloads only the missing codes for
    metadata 9506 and/or the daily bar, then queries once more.  Remaining nulls
    are deliberately retained so the downstream classifier can fail loudly.
    """

    trade_date = normalize_trade_date(trade_date)
    codes = normalize_stock_codes(stock_codes)
    limits = fetch_xt_historical_limit_prices(
        xtdata_client,
        codes,
        trade_date,
        chunk_size=chunk_size,
    )
    suspended = fetch_xt_daily_suspend_status(
        xtdata_client,
        codes,
        trade_date,
        chunk_size=chunk_size,
    )
    combined = (
        limits.merge(
            suspended,
            on=["stock_code", "status_date"],
            how="outer",
            validate="one_to_one",
        )
        .sort_values("stock_code")
        .reset_index(drop=True)
    )

    missing_limit_codes = combined.loc[
        combined[["up_limit_price", "down_limit_price"]].isna().any(axis=1),
        "stock_code",
    ].tolist()
    missing_daily_codes = combined.loc[
        combined["suspend_flag"].isna(), "stock_code"
    ].tolist()
    if download_missing and (missing_limit_codes or missing_daily_codes):
        downloader = getattr(xtdata_client, "download_history_data2", None)
        if downloader is None:
            raise RuntimeError(
                "XtQuant client has no download_history_data2 method, but local "
                "historical status data is incomplete."
            )
        for batch in _iter_chunks(missing_limit_codes, download_chunk_size):
            downloader(
                batch,
                (XT_LIMIT_META_ID, XT_DAILY_PERIOD_MS),
                trade_date,
                trade_date,
            )
        for batch in _iter_chunks(missing_daily_codes, download_chunk_size):
            downloader(batch, "1d", trade_date, trade_date)

        return fetch_xt_historical_trading_status(
            xtdata_client,
            codes,
            trade_date,
            chunk_size=chunk_size,
            download_missing=False,
            download_chunk_size=download_chunk_size,
        )
    return combined


def extract_opening_minute_snapshot(
    minute_cache: dict,
    stock_codes,
    trade_date: str,
    *,
    opening_minute: str = OPENING_MINUTE,
) -> pd.DataFrame:
    """Extract the D+1 09:31 minute close and last-close fallback.

    ``last_price_0931`` is used for the limit-state test.  The valuation column
    falls back to ``lastClose`` only for stocks without a transaction, matching
    the existing tracking module's valuation behavior for suspended names.
    """

    trade_date = normalize_trade_date(trade_date)
    codes = normalize_stock_codes(stock_codes)
    validate_minute_tick_cache(
        minute_cache,
        trade_date=trade_date,
        stock_codes=codes,
    )
    minute_key = int(f"{trade_date}{str(opening_minute)}")
    last_price = get_minute_field(
        minute_cache,
        "lastPrice",
        stock_codes=codes,
        copy=True,
    )
    last_close = get_minute_field(
        minute_cache,
        "lastClose",
        stock_codes=codes,
        copy=True,
    )
    if minute_key not in last_price.index:
        raise KeyError(f"Opening minute {minute_key} is absent from minute cache.")

    opening = pd.DataFrame(
        {
            "stock_code": codes,
            "status_date": trade_date,
            "last_price_0931": pd.to_numeric(
                last_price.loc[minute_key].reindex(codes), errors="coerce"
            ).to_numpy(),
            "last_close": pd.to_numeric(
                last_close.loc[minute_key].reindex(codes), errors="coerce"
            ).to_numpy(),
        }
    )
    opening["last_price_0931"] = opening["last_price_0931"].where(
        opening["last_price_0931"] > 0
    )
    opening["last_close"] = opening["last_close"].where(opening["last_close"] > 0)
    opening["valuation_price_0931"] = opening["last_price_0931"].fillna(
        opening["last_close"]
    )
    return opening


def classify_unavailable_stocks(
    xt_status: pd.DataFrame,
    opening_snapshot: pd.DataFrame,
    *,
    manual_unavailable_codes: list[str] | None = None,
    price_tolerance: float = 1e-6,
    include_suspended: bool = True,
) -> pd.DataFrame:
    """Classify D+1 09:31 limit-up, limit-down, suspended and manual exclusions."""

    required_status = {
        "stock_code",
        "status_date",
        "up_limit_price",
        "down_limit_price",
        "suspend_flag",
    }
    required_opening = {
        "stock_code",
        "status_date",
        "last_price_0931",
        "last_close",
        "valuation_price_0931",
    }
    missing_status = sorted(required_status - set(xt_status.columns))
    missing_opening = sorted(required_opening - set(opening_snapshot.columns))
    if missing_status or missing_opening:
        raise ValueError(
            f"Missing status/opening columns: status={missing_status}, "
            f"opening={missing_opening}"
        )

    status = xt_status.copy()
    opening = opening_snapshot.copy()
    status["stock_code"] = status["stock_code"].map(normalize_stock_code)
    opening["stock_code"] = opening["stock_code"].map(normalize_stock_code)
    if (
        status["stock_code"].duplicated().any()
        or opening["stock_code"].duplicated().any()
    ):
        raise ValueError("status/opening inputs contain duplicate stock codes.")

    report = status.merge(
        opening,
        on=["stock_code", "status_date"],
        how="outer",
        validate="one_to_one",
    )
    manual = {normalize_stock_code(code) for code in (manual_unavailable_codes or [])}
    for column in [
        "up_limit_price",
        "down_limit_price",
        "suspend_flag",
        "last_price_0931",
        "last_close",
        "valuation_price_0931",
    ]:
        report[column] = pd.to_numeric(report[column], errors="coerce")

    # XtQuant may omit the entire historical daily row for a suspended stock,
    # rather than returning a row whose suspendFlag is 1.  Treat that case as
    # suspended only when the D+1 minute cache independently confirms there was
    # no 09:31 trade and still provides a valid lastClose fallback.
    explicit_suspended = report["suspend_flag"].notna() & report["suspend_flag"].ne(0)
    inferred_suspended = (
        report["suspend_flag"].isna()
        & report["last_price_0931"].isna()
        & report["last_close"].notna()
    )
    unresolved_suspend_flag = report["suspend_flag"].isna() & ~inferred_suspended
    if unresolved_suspend_flag.any():
        bad = report.loc[unresolved_suspend_flag, "stock_code"].tolist()
        raise RuntimeError(
            "XtQuant daily row/suspendFlag is missing without independent "
            f"suspension evidence: {bad[:20]}"
        )
    report["is_suspended"] = explicit_suspended | inferred_suspended
    report["suspension_source"] = np.select(
        [explicit_suspended, inferred_suspended],
        ["XT_1D_SUSPEND_FLAG", "XT_NO_DAILY_BAR_AND_NO_0931_TRADE"],
        default="NOT_SUSPENDED",
    )

    missing_limits = (
        report[["up_limit_price", "down_limit_price"]].isna().any(axis=1)
        & ~report["is_suspended"]
    )
    if missing_limits.any():
        bad = report.loc[missing_limits, "stock_code"].tolist()
        raise RuntimeError(f"Missing XtQuant historical limit prices: {bad[:20]}")

    missing_opening_price = report["last_price_0931"].isna() & ~report["is_suspended"]
    if missing_opening_price.any():
        bad = report.loc[missing_opening_price, "stock_code"].tolist()
        raise RuntimeError(
            f"Non-suspended stocks have no usable 09:31 close: {bad[:20]}"
        )
    if report["valuation_price_0931"].isna().any():
        bad = report.loc[report["valuation_price_0931"].isna(), "stock_code"].tolist()
        raise RuntimeError(f"Stocks have no usable 09:31 valuation price: {bad[:20]}")

    price = report["last_price_0931"].to_numpy(dtype=float)
    up = report["up_limit_price"].to_numpy(dtype=float)
    down = report["down_limit_price"].to_numpy(dtype=float)
    report["is_limit_up_0931"] = np.isclose(
        price,
        up,
        rtol=0.0,
        atol=float(price_tolerance),
        equal_nan=False,
    )
    report["is_limit_down_0931"] = np.isclose(
        price,
        down,
        rtol=0.0,
        atol=float(price_tolerance),
        equal_nan=False,
    )
    report["is_manual_unavailable"] = report["stock_code"].isin(manual)
    report["is_unavailable"] = (
        report["is_limit_up_0931"]
        | report["is_limit_down_0931"]
        | report["is_manual_unavailable"]
        | (report["is_suspended"] if include_suspended else False)
    )

    def reason_for_row(row) -> str:
        reasons = []
        if bool(row.is_limit_up_0931):
            reasons.append("LIMIT_UP_0931")
        if bool(row.is_limit_down_0931):
            reasons.append("LIMIT_DOWN_0931")
        if include_suspended and bool(row.is_suspended):
            reasons.append("SUSPENDED")
        if bool(row.is_manual_unavailable):
            reasons.append("MANUAL")
        return "|".join(reasons) if reasons else "AVAILABLE"

    report["unavailable_reason"] = [
        reason_for_row(row) for row in report.itertuples(index=False)
    ]
    return report.sort_values("stock_code").reset_index(drop=True)


def unavailable_codes_from_report(status_report: pd.DataFrame) -> list[str]:
    if not {"stock_code", "is_unavailable"}.issubset(status_report.columns):
        raise ValueError("status_report requires stock_code and is_unavailable.")
    return sorted(
        status_report.loc[status_report["is_unavailable"], "stock_code"]
        .map(normalize_stock_code)
        .unique()
        .tolist()
    )


def build_basket2_from_basket1(
    basket1: pd.DataFrame,
    status_report: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zero unavailable basket1 positions without changing any remaining quantity."""

    required = {"stock_code", "target_qty", "close_price"}
    missing = sorted(required - set(basket1.columns))
    if missing:
        raise ValueError(f"basket1 missing columns: {missing}")
    if not {"stock_code", "is_unavailable", "unavailable_reason"}.issubset(
        status_report.columns
    ):
        raise ValueError("status_report has incomplete exclusion columns.")

    basket = basket1.copy()
    basket["stock_code"] = basket["stock_code"].map(normalize_stock_code)
    status = status_report[
        ["stock_code", "is_unavailable", "unavailable_reason"]
    ].copy()
    status["stock_code"] = status["stock_code"].map(normalize_stock_code)
    basket = basket.merge(status, on="stock_code", how="left", validate="one_to_one")
    if basket["is_unavailable"].isna().any():
        bad = basket.loc[basket["is_unavailable"].isna(), "stock_code"].tolist()
        raise ValueError(f"basket1 codes are absent from status report: {bad[:20]}")

    basket["basket1_qty"] = pd.to_numeric(basket["target_qty"], errors="raise")
    basket["removed_qty"] = basket["basket1_qty"].where(basket["is_unavailable"], 0)
    basket["target_qty"] = basket["basket1_qty"].where(~basket["is_unavailable"], 0)
    basket["removed_build_amount"] = basket["removed_qty"] * basket["close_price"]
    basket["target_market_value"] = basket["target_qty"] * basket["close_price"]
    basket["is_held"] = basket["target_qty"] > 0
    basket["portfolio_label"] = "basket2"

    removed = basket.loc[basket["removed_qty"] > 0].copy()
    return basket, removed.reset_index(drop=True)


def evaluate_baskets_at_0931(
    baskets: Mapping[str, pd.DataFrame],
    opening_snapshot: pd.DataFrame,
    *,
    construction_index_close: float,
    opening_index_close: float,
    common_base_amount: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate stock-only, normalized-return and cash-adjusted gaps.

    ``common_base_amount`` should be basket1's actual construction-day invested
    amount.  Cash adjustment uses ``common_base_amount - basket_build_amount``;
    this can be negative when a fitted basket uses the permitted over-budget
    interval, which transparently represents the extra funding.
    """

    construction_index_close = float(construction_index_close)
    opening_index_close = float(opening_index_close)
    common_base_amount = float(common_base_amount)
    if min(construction_index_close, opening_index_close, common_base_amount) <= 0:
        raise ValueError("index prices and common_base_amount must be positive.")

    opening = opening_snapshot.copy()
    if not {"stock_code", "valuation_price_0931"}.issubset(opening.columns):
        raise ValueError(
            "opening_snapshot requires stock_code and valuation_price_0931."
        )
    opening["stock_code"] = opening["stock_code"].map(normalize_stock_code)
    opening["valuation_price_0931"] = pd.to_numeric(
        opening["valuation_price_0931"], errors="coerce"
    )
    if opening["stock_code"].duplicated().any():
        raise ValueError("opening_snapshot contains duplicate stock codes.")

    index_ratio = opening_index_close / construction_index_close
    index_return = index_ratio - 1.0
    common_index_amount_0931 = common_base_amount * index_ratio
    rows = []
    details: dict[str, pd.DataFrame] = {}

    for basket_name, basket_frame in baskets.items():
        basket = basket_frame.copy()
        missing = sorted(
            {"stock_code", "target_qty", "close_price"} - set(basket.columns)
        )
        if missing:
            raise ValueError(f"{basket_name} missing columns: {missing}")
        basket["stock_code"] = basket["stock_code"].map(normalize_stock_code)
        basket["target_qty"] = pd.to_numeric(basket["target_qty"], errors="raise")
        basket["close_price"] = pd.to_numeric(basket["close_price"], errors="raise")
        held = basket.loc[basket["target_qty"] > 0].copy()
        held = held.merge(
            opening[["stock_code", "valuation_price_0931"]],
            on="stock_code",
            how="left",
            validate="one_to_one",
        )
        if held.empty:
            raise RuntimeError(f"{basket_name} has no positive holdings.")
        if (
            held["valuation_price_0931"].isna().any()
            or (held["valuation_price_0931"] <= 0).any()
        ):
            bad = held.loc[
                held["valuation_price_0931"].isna()
                | held["valuation_price_0931"].le(0),
                "stock_code",
            ].tolist()
            raise RuntimeError(
                f"{basket_name} has invalid 09:31 valuation prices: {bad[:20]}"
            )

        held["build_amount"] = held["target_qty"] * held["close_price"]
        held["amount_0931"] = held["target_qty"] * held["valuation_price_0931"]
        basket_build_amount = float(held["build_amount"].sum())
        basket_amount_0931 = float(held["amount_0931"].sum())
        basket_return = basket_amount_0931 / basket_build_amount - 1.0
        common_gap_amount = basket_amount_0931 - common_index_amount_0931
        cash_balance = common_base_amount - basket_build_amount
        cash_adjusted_amount_0931 = basket_amount_0931 + cash_balance
        cash_gap_amount = cash_adjusted_amount_0931 - common_index_amount_0931

        rows.append(
            {
                "basket": str(basket_name),
                "held_stock_count": int(len(held)),
                "basket_build_amount": basket_build_amount,
                "basket_amount_0931": basket_amount_0931,
                "common_base_amount": common_base_amount,
                "common_index_amount_0931": common_index_amount_0931,
                "common_base_gap_amount": common_gap_amount,
                "common_base_gap_pct": common_gap_amount
                / common_index_amount_0931
                * 100.0,
                "basket_return_pct": basket_return * 100.0,
                "index_return_pct": index_return * 100.0,
                "return_gap_pct": (basket_return - index_return) * 100.0,
                "cash_balance_from_common_base": cash_balance,
                "cash_adjusted_amount_0931": cash_adjusted_amount_0931,
                "cash_adjusted_gap_amount": cash_gap_amount,
                "cash_adjusted_gap_pct": cash_gap_amount
                / common_index_amount_0931
                * 100.0,
                "construction_index_close": construction_index_close,
                "opening_index_close": opening_index_close,
            }
        )
        held.insert(0, "basket", str(basket_name))
        details[str(basket_name)] = held.reset_index(drop=True)

    summary = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(baskets)}
    summary["_order"] = summary["basket"].map(order)
    summary = (
        summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    )
    return summary, details


__all__ = [
    "OPENING_MINUTE",
    "TASK11_DISPLAY_COLUMNS",
    "XT_DAILY_PERIOD_MS",
    "XT_LIMIT_META_ID",
    "build_basket2_from_basket1",
    "classify_unavailable_stocks",
    "combine_task11_summaries",
    "evaluate_baskets_at_0931",
    "expand_construction_date_list",
    "extract_opening_minute_snapshot",
    "fetch_xt_daily_suspend_status",
    "fetch_xt_historical_limit_prices",
    "fetch_xt_historical_trading_status",
    "get_next_trading_date",
    "get_recent_trading_dates",
    "normalize_construction_date_list",
    "unavailable_codes_from_report",
]
