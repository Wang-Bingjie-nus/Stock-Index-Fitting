"""
Tick 级跟踪误差分析：从个股 tick 合成指数 tick，与篮子 tick 比较。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def build_index_tick(
    stock_codes: list[str],
    weights: dict[str, float],
    build_date_dash: str,
    base_dir: str = r"Z:\高频行情迅投\ticks",
    nonnull_ratio_threshold: float = 0.99,
) -> pd.Series:
    """从个股 tick 加权合成指数 tick。

    参数:
        stock_codes: 带后缀的股票代码列表，如 ['600028.SH', '000001.SZ', ...]
        weights: {stock_code: weight} 权重字典（理论权重，小数形式，如 0.00785）
        build_date_dash: 构建日，格式 YYYY-MM-DD
        base_dir: NAS tick 数据根目录
        nonnull_ratio_threshold: 非空比例阈值（默认 99%），用于自动过滤开盘初期不稳定数据

    返回:
        pd.Series，index 为对齐后的时间戳（int, YYYYMMDDHHMMSS），
        values 为加权指数值（0~1 之间的相对值）
    
    说明:
        - 自动检测首个时刻，使得至此为止 >= nonnull_ratio_threshold 的成分股都有有效价格
        - 这样避免了人为设定开盘延迟时间，同时确保数据质量
    """
    from utils.reader import read_stocks_ticks

    tick_dict = read_stocks_ticks(build_date_dash, stock_codes, base_dir)
    if not tick_dict:
        raise RuntimeError("未能从 NAS 读取任何 tick 数据")

    # Step 1: 收集每只股票的 lastPrice 时间序列
    price_series = {}
    all_times = set()

    def _is_continuous_time(t):
        """只保留真正的连续竞价时段，从 09:30:00 开始（避免开盘混乱）"""
        hhmmss = str(int(float(t)))[-6:]
        # 09:30 到 11:30，13:01 到 15:00
        return ("093000" <= hhmmss <= "113000") or ("130100" <= hhmmss <= "150000")

    for code in stock_codes:
        raw = tick_dict.get(code)
        if raw is None or raw.empty:
            continue
        # 过滤：只保留连续竞价时段 + 有效价格
        raw = raw[raw["time"].map(_is_continuous_time)]
        raw = raw[raw["lastPrice"] > 0]
        if raw.empty:
            continue
        raw_dedup = raw.drop_duplicates(subset="time", keep="last")
        ts = raw_dedup.set_index("time")["lastPrice"]
        if ts.empty:
            continue
        all_times.update(ts.index.tolist())
        price_series[code] = ts

    if not price_series:
        raise RuntimeError("所有股票的 tick 数据均为空或无效")

    missing = set(stock_codes) - set(price_series.keys())
    if missing:
        warnings.warn(f"{len(missing)} 只股票无 tick 数据，已排除")
    
    # 自动检测开盘稳定点：找到首个时刻使得非空比例 >= threshold
    all_times_sorted = sorted(all_times)
    # min_nonnull_count = len(price_series) * nonnull_ratio_threshold
    # start_idx = 0
    
    # for i, t in enumerate(all_times_sorted):
    #     nonnull_count = sum(1 for code in price_series if t in price_series[code].index)
    #     if nonnull_count >= min_nonnull_count:
    #         start_idx = i
    #         break
    
    # if start_idx > 0:
    #     t_start = all_times_sorted[start_idx]
    #     t_start_str = str(t_start)
    #     print(f"[开盘稳定检测] 非空比例 {nonnull_ratio_threshold*100:.0f}% 达成于时刻 {t_start_str}，"
    #           f"已过滤 {start_idx} 个 tick 时点")
    #     all_times_sorted = all_times_sorted[start_idx:]

    # Step 2: 对齐到统一时间轴
    df_prices = pd.DataFrame(index=all_times_sorted)

    for code, ts in price_series.items():
        df_prices[code] = ts.reindex(all_times_sorted).ffill().bfill()

    # Step 3: 填补剩余 NaN（部分股票在最早几个 tick 前无数据）
    # df_prices = df_prices.ffill().bfill()

    # Step 4: 加权求和
    index_tick = pd.Series(0.0, index=all_times_sorted)
    total_weight = 0.0
    for code in price_series:
        w = weights.get(code, 0.0)
        if w > 0:
            index_tick += df_prices[code].fillna(0) * w
            total_weight += w

    # 归一化（因为可能有些股票缺数据，权重和不一定是 1）
    if total_weight > 0:
        index_tick = index_tick / total_weight

    return index_tick


def compute_tracking_error(
    index_tick: pd.Series,
    basket_tick: pd.Series,
) -> tuple[pd.Series, dict]:
    """计算篮子跟踪误差。

    参数:
        index_tick: 指数 tick 序列（**已用 lastClose 归一化**）
        basket_tick: 篮子 tick 序列（**已用 lastClose 归一化**）

    返回:
        (tracking_error_series, summary_stats_dict)

    注意: 调用方必须先将两个序列用相同的基准归一化（如除以 lastClose 加权和），
          本函数不再做任何额外归一化，直接对齐后计算 (bkt - idx) / idx。
    """
    # 对齐到共同时间轴
    common_idx = index_tick.index.intersection(basket_tick.index)
    common_idx_nodup = common_idx.drop_duplicates() if hasattr(common_idx, 'drop_duplicates') else pd.Index(common_idx).drop_duplicates()
    idx_aligned = index_tick.reindex(common_idx_nodup)
    bkt_aligned = basket_tick.reindex(common_idx_nodup)

    # 不再做任何归一化：数据应在传入前已用统一基准（如 lastClose）处理过
    # 直接计算百分比 TE: (basket - index) / index
    te = (bkt_aligned - idx_aligned) / idx_aligned.replace(0, np.nan)

    stats = {
        "te_mean_pct": float(te.mean() * 100),
        "te_std_pct": float(te.std() * 100),
        "te_max_positive_pct": float(te.max() * 100),
        "te_max_negative_pct": float(te.min() * 100),
        "te_pct95_pct": float(te.abs().quantile(0.95) * 100),
        "n_ticks": len(te),
    }

    return te, stats


def plot_tracking(
    index_tick: pd.Series,
    basket_tick: pd.Series,
    te_series: pd.Series,
    stats: dict,
    index_name: str = "",
):
    """绘制跟踪误差图表（上下两面板）。

    面板:
    - 上: 指数 vs 篮子 走势叠加
    - 下: 跟踪误差 %（含 ±3σ 区间）
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams['font.family'] = 'SimHei'

    # 时间轴转换: YYYYMMDDHHMMSS int → datetime
    def _to_dt(ts_idx):
        return pd.to_datetime(ts_idx.astype(str).str[:14], format="%Y%m%d%H%M%S")

    # 【强行过滤】只画 09:31:00 之后的数据，跳过开盘混乱
    def _filter_from_0931(s):
        def _is_0931(t):
            hhmmss = str(int(float(t)))[-6:]
            return ("093100" <= hhmmss <= "113000") or ("130100" <= hhmmss <= "150000")
        return s[s.index.map(_is_0931)]
    
    index_tick = _filter_from_0931(index_tick)
    basket_tick = _filter_from_0931(basket_tick)
    te_series = _filter_from_0931(te_series)

    common = index_tick.index.intersection(basket_tick.index)
    t = _to_dt(common)
    # 数据已在 Task16 中用 lastClose 归一化，直接取值
    idx = index_tick.reindex(common).values
    bkt = basket_tick.reindex(common).values
    te = te_series.reindex(common).values * 100  # %

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    # 上: 走势叠加
    ax1.plot(t, idx, linewidth=1.2, label="Index (CSI weight)", color="#1f77b4")
    ax1.plot(t, bkt, linewidth=1.0, label="Basket (target)", color="#ff7f0e", alpha=0.85)
    ax1.set_ylabel("Normalized value (base: lastClose=1.0)")
    ax1.set_title(f"Index vs Basket — Intraday Tracking {index_name}")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # 下: 跟踪误差
    ax2.fill_between(t, 0, te, alpha=0.25, color="#d62728")
    ax2.plot(t, te, linewidth=0.6, color="#d62728")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    # ±3σ 区间
    sigma = stats["te_std_pct"]
    ax2.axhline(y=3 * sigma, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax2.axhline(y=-3 * sigma, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax2.set_ylabel("Tracking Error (%)")
    ax2.set_xlabel("Time")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.grid(True, alpha=0.3)

    # 统计卡片（右上角文本框）
    textstr = (
        f"Mean TE: {stats['te_mean_pct']:.4f}%\n"
        f"TE Std:  {stats['te_std_pct']:.4f}%\n"
        f"Max +:   {stats['te_max_positive_pct']:.4f}%\n"
        f"Max -:   {stats['te_max_negative_pct']:.4f}%\n"
        f"95%ile:  {stats['te_pct95_pct']:.4f}%\n"
        f"Normalized by lastClose"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8)
    ax1.text(
        0.98, 0.97, textstr, transform=ax1.transAxes,
        fontsize=9, verticalalignment="top", horizontalalignment="right",
        bbox=props, family="monospace",
    )

    plt.tight_layout()
    return fig