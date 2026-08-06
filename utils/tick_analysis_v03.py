from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .minute_tick_cache_v03 import (
    get_minute_field,
    normalize_stock_code,
    validate_minute_tick_cache,
)
from .tick_analysis_v10 import (
    MinuteTrackingResult,
    _compute_stats,
    _curve_frame,
    _date_label,
    _prepare_real_index_minute_series,
    _stats_frame,
    build_corporate_action_quantity_schedule,
    combine_minute_tracking_results,
    merge_corporate_action_sources,
    plot_tracking_matplotlib,
    plot_tracking_plotly,
    standardize_corporate_actions,
)


PRICE_CACHE_FIELDS = {
    "lastPrice": "lastPrice",
    "bid1": "bidPrice1",
    "ask1": "askPrice1",
}


def _quantity_series(basket_quantities: dict[str, float]) -> pd.Series:
    values = {
        normalize_stock_code(code): float(qty)
        for code, qty in basket_quantities.items()
        if float(qty) > 0
    }
    if not values:
        raise RuntimeError("basket_quantities is empty.")
    return pd.Series(values, dtype=float).groupby(level=0).sum().sort_index()


def _aligned_price_matrix(
    minute_cache: dict,
    logical_price_col: str,
    minute_times: pd.Index,
    stock_codes: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if logical_price_col not in PRICE_CACHE_FIELDS:
        raise ValueError(
            f"Unsupported price column {logical_price_col!r}; "
            f"expected one of {sorted(PRICE_CACHE_FIELDS)}."
        )

    cache_field = PRICE_CACHE_FIELDS[logical_price_col]
    prices = get_minute_field(
        minute_cache,
        cache_field,
        stock_codes=stock_codes,
        copy=True,
    ).reindex(minute_times)
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.where(prices.gt(0)).ffill()

    last_close = get_minute_field(
        minute_cache,
        "lastClose",
        stock_codes=stock_codes,
        copy=True,
    ).reindex(minute_times)
    last_close = last_close.apply(pd.to_numeric, errors="coerce")
    last_close = last_close.where(last_close.gt(0)).ffill().bfill()
    prices = prices.fillna(last_close)

    missing_codes = prices.columns[prices.isna().all(axis=0)].tolist()
    return prices, missing_codes


def build_minute_tracking_analysis(
    minute_cache: dict,
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
    """Compare an index minute series with a basket using a v03 minute cache."""

    validate_minute_tick_cache(minute_cache)
    if basket_price_col not in price_cols:
        price_cols = tuple(dict.fromkeys((*price_cols, basket_price_col)))

    quantities = _quantity_series(basket_quantities)
    if basket_base_amount <= 0:
        raise RuntimeError("basket_base_amount must be positive.")

    real_index_amount, real_index_preclose, real_index_base_price, clean_real_index_frame = (
        _prepare_real_index_minute_series(
            real_index_frame=real_index_frame,
            basket_base_amount=float(basket_base_amount),
            real_index_price_caliber=real_index_price_caliber,
            real_index_amount_mode=real_index_amount_mode,
            previous_index_close=previous_index_close,
        )
    )
    minute_times = pd.Index(real_index_amount.index, dtype="int64", name="time")
    stock_codes = quantities.index.tolist()

    basket_amount: dict[str, pd.Series] = {}
    for price_col in price_cols:
        aligned_prices, missing_codes = _aligned_price_matrix(
            minute_cache,
            price_col,
            minute_times,
            stock_codes,
        )
        if missing_codes:
            warnings.warn(
                f"{len(missing_codes)} basket stocks have no usable {price_col} "
                "minute price and no lastClose fallback."
            )
        total = aligned_prices.fillna(0.0).mul(quantities, axis=1).sum(axis=1)
        total.index = minute_times
        basket_amount[price_col] = total.astype(float)

    diff_amount = basket_amount[basket_price_col] - real_index_amount
    diff_ratio = diff_amount / real_index_amount.replace(0, np.nan)
    stats = _compute_stats(
        diff_amount,
        diff_ratio,
        float(basket_base_amount),
        real_index_preclose,
    )
    stats["real_index_base_price"] = real_index_base_price
    stats["real_index_price_caliber"] = real_index_price_caliber
    stats["real_index_amount_mode"] = real_index_amount_mode
    stats["corporate_action_enabled"] = bool(corporate_action_enabled)
    stats["tick_cache_kind"] = "basket_tick_v03"
    if (
        daily_dividend_summary is not None
        and not daily_dividend_summary.empty
        and "daily_dividend_cash" in daily_dividend_summary.columns
    ):
        stats["daily_dividend_cash"] = float(
            pd.to_numeric(
                daily_dividend_summary["daily_dividend_cash"],
                errors="coerce",
            ).fillna(0).sum()
        )
        if "cumulative_dividend_cash" in daily_dividend_summary.columns:
            stats["cumulative_dividend_cash"] = float(
                pd.to_numeric(
                    daily_dividend_summary["cumulative_dividend_cash"],
                    errors="coerce",
                ).dropna().iloc[-1]
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

    curve_path = output_dir / f"minute_tracking_v03_{label}.csv"
    _curve_frame(result).to_csv(curve_path, index=False, encoding="utf-8-sig")

    stats_path = output_dir / f"minute_tracking_stats_v03_{label}.csv"
    _stats_frame(result, daily_results).to_csv(stats_path, index=False, encoding="utf-8-sig")

    quantity_adjustments_path = output_dir / f"minute_tracking_quantity_adjustments_v03_{label}.csv"
    dividend_records_path = output_dir / f"minute_tracking_dividends_v03_{label}.csv"
    dividend_summary_path = output_dir / f"minute_tracking_dividend_summary_v03_{label}.csv"
    if result.quantity_adjustments is not None:
        result.quantity_adjustments.to_csv(quantity_adjustments_path, index=False, encoding="utf-8-sig")
    if result.dividend_records is not None:
        result.dividend_records.to_csv(dividend_records_path, index=False, encoding="utf-8-sig")
    if result.daily_dividend_summary is not None:
        result.daily_dividend_summary.to_csv(dividend_summary_path, index=False, encoding="utf-8-sig")

    matplotlib_fig = plot_tracking_matplotlib(result, index_name)
    plt.close(matplotlib_fig)

    html_path = output_dir / f"minute_tracking_v03_{label}.html"
    plotly_fig = plot_tracking_plotly(
        result,
        index_name,
        html_path=html_path,
        auto_open=False,
    )
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


__all__ = [
    "MinuteTrackingResult",
    "build_corporate_action_quantity_schedule",
    "merge_corporate_action_sources",
    "standardize_corporate_actions",
    "build_minute_tracking_analysis",
    "combine_minute_tracking_results",
    "save_tracking_outputs",
    "plot_tracking_matplotlib",
    "plot_tracking_plotly",
]
