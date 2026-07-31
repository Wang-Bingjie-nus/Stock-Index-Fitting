from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .minute_tick_cache_v03 import (
    CONTINUOUS_TRADING_MINUTE_ROWS,
    get_minute_field,
    normalize_stock_codes,
    validate_minute_tick_cache,
)
from .risk_model_v02 import ShrunkRiskModel, build_ew_oas_covariance


# def build_intraday_log_return_matrix(
#     minute_cache_by_date: dict[str, dict],
#     stock_codes: list[str],
#     *,
#     price_col: str = "lastPrice",
# ) -> pd.DataFrame:
#     """Build within-day log returns from v03 minute-wide tick caches."""

#     stock_codes = normalize_stock_codes(stock_codes)
#     daily_returns = []

#     for date_key, minute_cache in sorted(minute_cache_by_date.items()):
#         validate_minute_tick_cache(minute_cache, trade_date=str(date_key), stock_codes=stock_codes)
#         prices = get_minute_field(
#             minute_cache,
#             price_col,
#             stock_codes=stock_codes,
#             copy=True,
#         )
#         prices = prices.apply(pd.to_numeric, errors="coerce")
#         prices = prices.where(prices.gt(0)).ffill().bfill()

#         log_returns = np.log(prices).diff().iloc[1:].ffill()
#         if log_returns.empty:
#             continue
#         minute_labels = ((log_returns.index.to_numpy(dtype="int64") % 1_000_000) // 100).astype(int)
#         log_returns.index = [
#             f"{str(date_key)}_{minute:04d}"
#             for minute in minute_labels
#         ]
#         daily_returns.append(log_returns)

#     if not daily_returns:
#         return pd.DataFrame(columns=stock_codes)

#     returns = pd.concat(daily_returns, axis=0).reindex(columns=stock_codes)
#     missing = returns.isna().sum()
#     if missing.any():
#         details = missing.loc[missing.gt(0)].sort_values(ascending=False).head(10).to_dict()
#         raise ValueError(f"Risk returns still contain missing values after ffill/bfill: {details}")
#     return returns

def build_intraday_log_return_matrix(
    minute_cache_by_date: dict[str, dict],
    stock_codes: list[str],
    *,
    price_col: str = "lastPrice",
) -> pd.DataFrame:
    stock_codes = normalize_stock_codes(stock_codes)
    daily_returns = []

    for date_key, minute_cache in sorted(minute_cache_by_date.items()):
        date_key = str(date_key)

        validate_minute_tick_cache(
            minute_cache,
            trade_date=date_key,
            stock_codes=stock_codes,
        )

        # 获取需要计算收益率的分钟价格
        prices = get_minute_field(
            minute_cache,
            price_col,
            stock_codes=stock_codes,
            copy=True,
        )
        prices = prices.apply(pd.to_numeric, errors="coerce")
        if len(prices) != CONTINUOUS_TRADING_MINUTE_ROWS:
            raise ValueError(
                f"Risk model requires {CONTINUOUS_TRADING_MINUTE_ROWS} continuous-trading "
                f"minute prices for {date_key}, got {len(prices)}."
            )

        # 非正价格不视为有效市场价格
        prices = prices.where(prices.gt(0))

        # 识别全天没有任何有效价格的股票
        all_missing_codes = (
            prices.columns[prices.isna().all(axis=0)]
            .astype(str)
            .tolist()
        )

        fallback_codes = []

        if all_missing_codes:
            # 只读取异常股票的 lastClose
            last_close = get_minute_field(
                minute_cache,
                "lastClose",
                stock_codes=all_missing_codes,
                copy=True,
            )
            last_close = last_close.apply(
                pd.to_numeric,
                errors="coerce",
            )
            last_close = last_close.where(last_close.gt(0))
            last_close = last_close.ffill().bfill()

            # 只有存在有效 lastClose 的股票才能进行回退
            fallback_codes = [
                code
                for code in all_missing_codes
                if (
                    code in last_close.columns
                    and last_close[code].notna().any()
                )
            ]

            for code in fallback_codes:
                prices[code] = last_close[code]

            if fallback_codes:
                print(
                    f"[Risk fallback {date_key}] "
                    f"使用 lastClose 填充全天无有效 {price_col} "
                    f"的股票，共 {len(fallback_codes)} 只："
                    f"{fallback_codes}"
                )

            unresolved_codes = sorted(
                set(all_missing_codes) - set(fallback_codes)
            )
            if unresolved_codes:
                print(
                    f"[Risk unresolved {date_key}] "
                    f"以下股票全天无有效 {price_col}，"
                    f"同时也没有有效 lastClose："
                    f"{unresolved_codes}"
                )

        # 部分分钟缺失使用日内前后价格填充。
        # 全天缺失的股票已经在上面优先使用 lastClose。
        prices = prices.ffill().bfill()

        # 240 个连续交易分钟价格在每个交易日内产生 239 条纯日内收益，
        # 不引入昨收价，因此不混入隔夜和集合竞价收益。
        log_returns = np.log(prices).diff().iloc[1:].ffill()

        if log_returns.empty:
            continue

        minute_labels = (
            (
                log_returns.index.to_numpy(dtype="int64")
                % 1_000_000
            )
            // 100
        ).astype(int)

        log_returns.index = [
            f"{date_key}_{minute:04d}"
            for minute in minute_labels
        ]

        daily_returns.append(log_returns)

    if not daily_returns:
        return pd.DataFrame(columns=stock_codes)

    returns = pd.concat(
        daily_returns,
        axis=0,
    ).reindex(columns=stock_codes)

    # 最终完整性检查
    missing = returns.isna().sum()

    if missing.any():
        details = (
            missing.loc[missing.gt(0)]
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        )

        raise ValueError(
            "Risk returns still contain missing values after "
            f"lastClose fallback and ffill/bfill: {details}"
        )

    return returns


def build_shrunk_risk_model_from_minute_caches(
    minute_cache_by_date: dict[str, dict],
    stock_codes: list[str],
    *,
    price_col: str = "lastPrice",
    half_life_days: float = 3.0,
) -> ShrunkRiskModel:
    returns = build_intraday_log_return_matrix(
        minute_cache_by_date,
        stock_codes,
        price_col=price_col,
    )
    covariance, correlation, summary = build_ew_oas_covariance(
        returns,
        half_life_days=half_life_days,
    )
    summary["price_col"] = price_col
    summary["tick_cache_kind"] = "basket_tick_v03"
    return ShrunkRiskModel(
        covariance=covariance,
        correlation=correlation,
        returns=returns,
        summary=summary,
    )


def build_shrunk_risk_model_from_daily_loader(
    risk_dates: list[str],
    stock_codes: list[str],
    daily_minute_cache_loader: Callable[[str], dict | None],
    *,
    price_col: str = "lastPrice",
    half_life_days: float = 3.0,
    min_valid_dates: int = 1,
) -> ShrunkRiskModel:
    """Build a risk model while skipping unavailable dates and streaming caches."""

    stock_codes = normalize_stock_codes(stock_codes)
    requested_risk_dates = [str(risk_date) for risk_date in risk_dates]
    min_valid_dates = int(min_valid_dates)
    if min_valid_dates <= 0:
        raise ValueError("min_valid_dates must be positive.")

    daily_return_frames = []
    used_risk_dates = []
    skipped_risk_dates = []
    for risk_date in requested_risk_dates:
        minute_cache = daily_minute_cache_loader(risk_date)
        if minute_cache is None:
            skipped_risk_dates.append(risk_date)
            continue
        daily_returns = build_intraday_log_return_matrix(
            {str(risk_date): minute_cache},
            stock_codes,
            price_col=price_col,
        )
        if daily_returns.empty:
            raise ValueError(f"Risk returns are empty for {risk_date}.")
        daily_return_frames.append(daily_returns)
        used_risk_dates.append(risk_date)
        del minute_cache

    if len(used_risk_dates) < min_valid_dates:
        raise ValueError(
            f"Risk model requires at least {min_valid_dates} valid risk dates; "
            f"requested={requested_risk_dates}, used={used_risk_dates}, "
            f"skipped={skipped_risk_dates}."
        )

    returns = pd.concat(daily_return_frames, axis=0).reindex(columns=stock_codes)
    covariance, correlation, summary = build_ew_oas_covariance(
        returns,
        half_life_days=half_life_days,
    )
    summary["price_col"] = price_col
    summary["tick_cache_kind"] = "basket_tick_v03"
    summary["tick_loading"] = "daily_minute_cache_streaming"
    summary["requested_risk_dates"] = requested_risk_dates
    summary["used_risk_dates"] = used_risk_dates
    summary["skipped_risk_dates"] = skipped_risk_dates
    summary["min_valid_dates"] = min_valid_dates
    return ShrunkRiskModel(
        covariance=covariance,
        correlation=correlation,
        returns=returns,
        summary=summary,
    )
