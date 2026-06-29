"""静态暴露汇总：行业暴露、市值暴露、个股 Top 偏离、Active Share。

纯函数，只做汇总，不负责 gogoal SQL 查询。
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd


def _strip_market_suffix(value: str) -> str:
    return str(value).strip().upper().split(".")[0].zfill(6)


def calculate_exposure_deviation(
    df_deviation_report: pd.DataFrame,
    df_industry: pd.DataFrame,
    df_mcap: pd.DataFrame,
    import_time: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """计算静态暴露汇总。

    Returns:
        industry_df, mcap_df, stock_top_df, portfolio_summary_patch（含 active_share）
    """
    # ========== 准备偏差表 ==========
    df = df_deviation_report.copy()
    df["code_6"] = df["stock_code"].map(_strip_market_suffix)

    # ========== 行业暴露 ==========
    df_ind = df_industry.copy()
    df_ind["code_6"] = df_ind["stock_code"].astype(str).str.zfill(6)
    # 一对多取字典序第一条
    df_ind = df_ind.sort_values("industry_code").groupby("code_6", as_index=False).first()

    df_merged = df.merge(df_ind[["code_6", "industry_code", "industry_name"]], on="code_6", how="left")

    missing_ind = df_merged["industry_code"].isna()
    if missing_ind.any():
        warnings.warn(f"{missing_ind.sum()} 只股票缺申万二级行业，归入 UNCLASSIFIED")
    df_merged["industry_code"] = df_merged["industry_code"].fillna("UNCLASSIFIED")
    df_merged["industry_name"] = df_merged["industry_name"].fillna("未分类")

    industry_agg = df_merged.groupby(["industry_code", "industry_name"], as_index=False).agg(
        theoretical_weight_sum=("theoretical_weight", "sum"),
        target_funding_weight_sum=("target_funding_weight", "sum"),
        active_weight_sum=("funding_weight_deviation", "sum"),
    )
    industry_agg["import_time"] = import_time

    # 排序：UNCLASSIFIED 放最后
    industry_agg["_sort"] = industry_agg["industry_code"].map(lambda x: "zzz" if x == "UNCLASSIFIED" else x)
    industry_agg = industry_agg.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

    # ========== 市值暴露 ==========
    df_mcap_clean = df_mcap.copy()
    df_mcap_clean["code_6"] = df_mcap_clean["stock_code"].astype(str).str.zfill(6)
    df_mcap_clean = df_mcap_clean[["code_6", "mcap"]].copy()
    df_mcap_clean["mcap"] = pd.to_numeric(df_mcap_clean["mcap"], errors="coerce")

    df_for_mcap = df.merge(df_mcap_clean, on="code_6", how="left")

    missing_mcap = df_for_mcap["mcap"].isna() | (df_for_mcap["mcap"] <= 0)
    if missing_mcap.any():
        warnings.warn(f"{missing_mcap.sum()} 只股票缺 mcap 或 mcap≤0，不参与市值分桶")

    df_mcap_valid = df_for_mcap[~missing_mcap].copy()

    mcap_df = None
    if len(df_mcap_valid) < 3:
        warnings.warn(f"有效 mcap 样本数 {len(df_mcap_valid)} < 3，跳过 market_cap_exposure.csv")
    else:
        df_mcap_valid = df_mcap_valid.sort_values("mcap", ascending=False).reset_index(drop=True)
        n = len(df_mcap_valid)
        n_large = math.ceil(n / 3)
        n_small = n // 3
        n_mid = n - n_large - n_small

        def _assign_bucket(idx: int) -> str:
            if idx < n_large:
                return "LARGE"
            elif idx < n_large + n_mid:
                return "MID"
            else:
                return "SMALL"

        df_mcap_valid["size_bucket"] = [_assign_bucket(i) for i in range(n)]
        mcap_agg = df_mcap_valid.groupby("size_bucket", as_index=False).agg(
            mcap_min=("mcap", "min"),
            mcap_max=("mcap", "max"),
            stock_count=("stock_code", "count"),
            theoretical_weight_sum=("theoretical_weight", "sum"),
            target_funding_weight_sum=("target_funding_weight", "sum"),
            active_weight_sum=("funding_weight_deviation", "sum"),
        )
        mcap_agg["import_time"] = import_time
        # 固定顺序
        bucket_order = {"LARGE": 0, "MID": 1, "SMALL": 2}
        mcap_agg["_sort"] = mcap_agg["size_bucket"].map(bucket_order)
        mcap_agg = mcap_agg.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
        mcap_df = mcap_agg

    # ========== 个股暴露 Top10 ==========
    # 超配
    over = df[df["funding_weight_deviation"] > 0].copy()
    over = over.sort_values(["funding_weight_deviation", "stock_code"], ascending=[False, True])
    over = over.head(10).reset_index(drop=True)
    over["rank"] = range(1, len(over) + 1)
    over["direction"] = "OVER"

    # 低配
    under = df[df["funding_weight_deviation"] < 0].copy()
    under = under.sort_values(["funding_weight_deviation", "stock_code"], ascending=[True, True])
    under = under.head(10).reset_index(drop=True)
    under["rank"] = range(1, len(under) + 1)
    under["direction"] = "UNDER"

    top_cols = [
        "rank", "direction", "stock_code",
        "theoretical_weight", "target_funding_weight",
        "funding_weight_deviation", "absolute_amount_error",
    ]
    stock_top = pd.concat([over[top_cols], under[top_cols]], ignore_index=True)
    stock_top["import_time"] = import_time

    # ========== Active Share ==========
    active_share = 0.5 * df["funding_weight_deviation"].abs().sum()
    portfolio_summary_patch = {"active_share": float(active_share)}

    return industry_agg, mcap_df, stock_top, portfolio_summary_patch
