from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    real_index_amount: pd.Series
    basket_amount: dict[str, pd.Series]
    diff_amount: pd.Series
    diff_ratio: pd.Series
    stats: dict
    real_index_frame: pd.DataFrame


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


def _prepare_real_index_minute_series(
    real_index_frame: pd.DataFrame,
    basket_base_amount: float,
) -> tuple[pd.Series, float, pd.DataFrame]:
    if real_index_frame is None or real_index_frame.empty:
        raise RuntimeError("real_index_frame is empty.")
    if "close" not in real_index_frame.columns:
        raise ValueError("real_index_frame must contain close.")
    if "preClose" not in real_index_frame.columns:
        raise ValueError("real_index_frame must contain preClose.")

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

    # Keep minute granularity only. Seconds are set to 00 by construction.
    minute_dt = pd.Index(dt).floor("min")
    frame["_tick_time"] = minute_dt.strftime("%Y%m%d%H%M%S").astype("int64")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["preClose"] = pd.to_numeric(frame["preClose"], errors="coerce")
    frame = frame.dropna(subset=["_tick_time", "close", "preClose"])
    frame = frame[(frame["close"] > 0) & (frame["preClose"] > 0)]
    frame = frame[frame["_tick_time"].map(_in_session)]
    frame = frame.drop_duplicates(subset="_tick_time", keep="last").sort_values("_tick_time")

    if frame.empty:
        raise RuntimeError("real_index_frame has no valid minute rows.")

    preclose = float(frame["preClose"].iloc[0])
    amount = pd.Series(
        frame["close"].to_numpy(dtype=float) / preclose * basket_base_amount,
        index=pd.Index(frame["_tick_time"].astype("int64")),
        dtype=float,
    )
    amount = amount[~amount.index.duplicated(keep="last")]
    return amount, preclose, frame


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

    # Align high-frequency stock ticks directly to official index minute timestamps.
    # There is no second-level expansion; only one point per minute is produced.
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
) -> MinuteTrackingResult:
    """
    Compare a real XtQuant 1m index series with a stock basket aligned to 1m.

    The basket quantities must be constructed from previous_trade_date closes.
    The basket_base_amount should be the previous_trade_date basket value.
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

    real_index_amount, real_index_preclose, clean_real_index_frame = _prepare_real_index_minute_series(
        real_index_frame=real_index_frame,
        basket_base_amount=float(basket_base_amount),
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

    stats = {
        "basket_base_amount": float(basket_base_amount),
        "real_index_preclose": real_index_preclose,
        "diff_mean_amount": float(diff_amount.mean()),
        "diff_std_amount": float(diff_amount.std()),
        "diff_max_positive_amount": float(diff_amount.max()),
        "diff_max_negative_amount": float(diff_amount.min()),
        "diff_mean_pct": float(diff_ratio.mean() * 100),
        "diff_std_pct": float(diff_ratio.std() * 100),
        "diff_max_positive_pct": float(diff_ratio.max() * 100),
        "diff_max_negative_pct": float(diff_ratio.min() * 100),
        "diff_pct95_pct": float(diff_ratio.abs().quantile(0.95) * 100),
        "n_minutes": int(len(diff_ratio.dropna())),
    }

    return MinuteTrackingResult(
        times=minute_times,
        basket_base_amount=float(basket_base_amount),
        real_index_preclose=real_index_preclose,
        real_index_amount=real_index_amount,
        basket_amount=basket_amount,
        diff_amount=diff_amount,
        diff_ratio=diff_ratio,
        stats=stats,
        real_index_frame=clean_real_index_frame,
    )


def plot_tracking_matplotlib(result: MinuteTrackingResult, index_name: str = ""):

    plt.rcParams["font.family"] = "SimHei"
    plt.rcParams['axes.unicode_minus'] = False

    t = _to_datetime_index(result.times)
    base = result.basket_base_amount
    index_pct = (result.real_index_amount / base - 1.0) * 100
    basket_pct = (result.basket_amount["lastPrice"] / base - 1.0) * 100

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(15, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1]},
    )

    if "bid1" in result.basket_amount and "ask1" in result.basket_amount:
        bid_pct = (result.basket_amount["bid1"] / base - 1.0) * 100
        ask_pct = (result.basket_amount["ask1"] / base - 1.0) * 100
        low = np.fmin(bid_pct.values, ask_pct.values)
        high = np.fmax(bid_pct.values, ask_pct.values)
        ax1.fill_between(t, low, high, color="#ff7f0e", alpha=0.14, label="Basket bid1-ask1 band")
        ax1.plot(t, bid_pct.values, linewidth=0.8, linestyle="--", color="#2ca02c", label="Basket bid1")
        ax1.plot(t, ask_pct.values, linewidth=0.8, linestyle="--", color="#d62728", label="Basket ask1")

    ax1.plot(t, index_pct.values, linewidth=1.3, color="#1f77b4", label="Real index 1m close/preClose")
    ax1.plot(t, basket_pct.values, linewidth=1.1, color="#ff7f0e", label="Basket 1m lastPrice")
    ax1.set_ylabel("Return from previous close base (%)")
    ax1.set_title(f"Real Index vs Basket Minute Tracking {index_name}")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    def pct_to_amount(y):
        return base * (1.0 + np.asarray(y) / 100.0)

    def amount_to_pct(y):
        return (np.asarray(y) / base - 1.0) * 100.0

    def amount_fmt(x, _pos):
        if abs(x) >= 1e8:
            return f"{x / 1e8:.2f}e8"
        if abs(x) >= 1e4:
            return f"{x / 1e4:.0f}w"
        return f"{x:,.0f}"

    ax1_right = ax1.secondary_yaxis("right", functions=(pct_to_amount, amount_to_pct))
    ax1_right.set_ylabel("Amount")
    ax1_right.yaxis.set_major_formatter(FuncFormatter(amount_fmt))

    diff_pct = result.diff_ratio * 100
    ax2.fill_between(t, 0, diff_pct.values, alpha=0.22, color="#d62728")
    ax2.plot(t, diff_pct.values, linewidth=0.8, color="#d62728", label="Basket - real index")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    sigma = result.stats["diff_std_pct"]
    ax2.axhline(y=3 * sigma, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax2.axhline(y=-3 * sigma, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax2.set_ylabel("Deviation (%)")
    ax2.set_xlabel("Time")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    textstr = (
        f"Mean: {result.stats['diff_mean_pct']:.4f}%\n"
        f"Std:  {result.stats['diff_std_pct']:.4f}%\n"
        f"Max+: {result.stats['diff_max_positive_pct']:.4f}%\n"
        f"Max-: {result.stats['diff_max_negative_pct']:.4f}%\n"
        f"95%:  {result.stats['diff_pct95_pct']:.4f}%\n"
        f"Minutes: {result.stats['n_minutes']}\n"
        f"Base: {base:,.0f}"
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

    plt.tight_layout()
    return fig


def plot_tracking_plotly(
    result: MinuteTrackingResult,
    index_name: str = "",
    *,
    html_path: str | Path | None = None,
    auto_open: bool = False,
):

    t = _to_datetime_index(result.times)
    base = result.basket_base_amount
    index_amount = result.real_index_amount
    basket_amount = result.basket_amount["lastPrice"]
    index_pct = (index_amount / base - 1.0) * 100
    basket_pct = (basket_amount / base - 1.0) * 100
    diff_pct = result.diff_ratio * 100

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=(f"Real Index vs Basket Minute Tracking {index_name}", "Deviation and amount gap"),
    )

    if "bid1" in result.basket_amount and "ask1" in result.basket_amount:
        bid_pct = (result.basket_amount["bid1"] / base - 1.0) * 100
        ask_pct = (result.basket_amount["ask1"] / base - 1.0) * 100
        low = np.fmin(bid_pct.values, ask_pct.values)
        high = np.fmax(bid_pct.values, ask_pct.values)

        fig.add_trace(go.Scatter(x=t, y=low, mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=t, y=high, mode="lines", name="Basket bid1-ask1 band",
                                 fill="tonexty", fillcolor="rgba(255,127,14,0.16)",
                                 line=dict(width=0)),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=t, y=bid_pct, mode="lines", name="Basket bid1",
                                 line=dict(color="#2ca02c", width=1, dash="dash")),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=t, y=ask_pct, mode="lines", name="Basket ask1",
                                 line=dict(color="#d62728", width=1, dash="dash")),
                      row=1, col=1, secondary_y=False)

    fig.add_trace(
        go.Scatter(
            x=t,
            y=index_pct,
            mode="lines",
            name="Real index 1m close/preClose",
            customdata=np.column_stack([index_amount.values]),
            hovertemplate="Time=%{x}<br>Return=%{y:.4f}%<br>Amount=%{customdata[0]:,.2f}<extra></extra>",
            line=dict(color="#1f77b4", width=1.4),
        ),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=basket_pct,
            mode="lines",
            name="Basket 1m lastPrice",
            customdata=np.column_stack([basket_amount.values]),
            hovertemplate="Time=%{x}<br>Return=%{y:.4f}%<br>Amount=%{customdata[0]:,.2f}<extra></extra>",
            line=dict(color="#ff7f0e", width=1.2),
        ),
        row=1, col=1, secondary_y=False,
    )

    fig.add_trace(go.Scatter(x=t, y=index_amount, mode="lines", showlegend=False,
                             hoverinfo="skip", line=dict(width=0), opacity=0),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=t, y=basket_amount, mode="lines", showlegend=False,
                             hoverinfo="skip", line=dict(width=0), opacity=0),
                  row=1, col=1, secondary_y=True)

    fig.add_trace(
        go.Scatter(
            x=t,
            y=diff_pct,
            mode="lines",
            name="Deviation %",
            customdata=np.column_stack([result.diff_amount.values]),
            hovertemplate="Time=%{x}<br>Deviation=%{y:.4f}%<br>Amount gap=%{customdata[0]:,.2f}<extra></extra>",
            line=dict(color="#d62728", width=1),
        ),
        row=2, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=t,
            y=result.diff_amount,
            name="Amount gap",
            marker_color="rgba(214,39,40,0.28)",
            hovertemplate="Time=%{x}<br>Amount gap=%{y:,.2f}<extra></extra>",
        ),
        row=2, col=1, secondary_y=True,
    )

    fig.add_hline(y=0, line_width=1, line_color="black", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Return from previous close base (%)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Amount", row=1, col=1, secondary_y=True, tickformat=",.0f")
    fig.update_yaxes(title_text="Deviation (%)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Amount gap", row=2, col=1, secondary_y=True, tickformat=",.0f")
    fig.update_xaxes(title_text="Time", row=2, col=1, rangeslider_visible=True)
    fig.update_layout(
        height=820,
        width=1300,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=70, r=80, t=90, b=60),
    )

    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path), include_plotlyjs="cdn", auto_open=auto_open)

    return fig


def save_tracking_outputs(
    result: MinuteTrackingResult,
    output_dir: str | Path,
    build_date: str,
    index_name: str = "",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_path = output_dir / f"minute_tracking_v4_{build_date}.csv"
    pd.DataFrame({
        "time": result.times,
        "real_index_amount": result.real_index_amount.reindex(result.times).to_numpy(),
        "basket_lastPrice_amount": result.basket_amount["lastPrice"].reindex(result.times).to_numpy(),
        "basket_bid1_amount": result.basket_amount.get("bid1", pd.Series(index=result.times, dtype=float)).reindex(result.times).to_numpy(),
        "basket_ask1_amount": result.basket_amount.get("ask1", pd.Series(index=result.times, dtype=float)).reindex(result.times).to_numpy(),
        "diff_amount": result.diff_amount.reindex(result.times).to_numpy(),
        "diff_ratio": result.diff_ratio.reindex(result.times).to_numpy(),
    }).to_csv(curve_path, index=False, encoding="utf-8-sig")

    matplotlib_fig = plot_tracking_matplotlib(result, index_name)

    plt.close(matplotlib_fig)

    html_path = output_dir / f"minute_tracking_v4_{build_date}.html"
    plotly_fig = plot_tracking_plotly(result, index_name, html_path=html_path, auto_open=False)

    return {
        "curve_path": curve_path,
        "html_path": html_path,
        "matplotlib_fig": matplotlib_fig,
        "plotly_fig": plotly_fig,
    }
