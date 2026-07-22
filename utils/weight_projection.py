from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .csi_reader import read_csi_file
from .downloader_v02 import download_csi_constituent_v02
from .tick_analysis_v10 import standardize_corporate_actions


CSI_CODE_FIELD = "成份券代码Constituent Code"
CSI_NAME_FIELD = "成份券名称Constituent Name"
CSI_WEIGHT_FIELD = "权重(%)weight"


@dataclass
class ProjectionOutput:
    target_weights: pd.DataFrame
    trace: dict[str, pd.DataFrame]
    daily_summary: pd.DataFrame
    source_weights: pd.DataFrame
    standardized_actions: pd.DataFrame
    output_dir: Path
    target_path: Path
    trace_dir: Path | None
    summary_path: Path


@dataclass
class WeightSourceSelection:
    weights: pd.DataFrame
    source_path: Path
    source_date: str
    selection_mode: str
    latest_source_date: str


def normalize_stock_code(value) -> str:
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


def strip_market_suffix(value) -> str:
    return str(value).strip().upper().split(".")[0].zfill(6)


def normalize_date_key(value) -> str:
    raw = str(value).strip()
    if re.fullmatch(r"\d{8}", raw):
        return raw
    return pd.Timestamp(value).strftime("%Y%m%d")


def normalize_date_dash(value) -> str:
    raw = str(value).strip()
    if re.fullmatch(r"\d{8}", raw):
        return pd.to_datetime(raw, format="%Y%m%d").strftime("%Y-%m-%d")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def sql_literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_in(values) -> str:
    vals = list(values)
    if not vals:
        raise ValueError("SQL IN list is empty.")
    return ", ".join(sql_literal(v) for v in vals)


def _extract_single_csi_data_date(frame: pd.DataFrame, label: str) -> str:
    date_col = frame.columns[0]
    parsed_dates = []
    for value in frame[date_col].dropna().unique():
        raw = str(value).strip()
        if re.fullmatch(r"\d{8}", raw):
            parsed = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
        else:
            parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            parsed_dates.append(pd.Timestamp(parsed).strftime("%Y%m%d"))
    unique_dates = sorted(set(parsed_dates))
    if len(unique_dates) != 1:
        raise RuntimeError(f"{label} first-column date is not unique: {unique_dates}")
    return unique_dates[0]


def standardize_weight_frame(raw_weights: pd.DataFrame, *, source_path: str | Path | None = None) -> pd.DataFrame:
    if raw_weights is None or raw_weights.empty:
        raise RuntimeError("weight source is empty.")

    frame = raw_weights.copy()
    if {"stock_code", "raw_weight_pct"} <= set(frame.columns):
        out = frame.copy()
        out["stock_code"] = out["stock_code"].map(normalize_stock_code)
        if "stock_name" not in out.columns:
            out["stock_name"] = ""
        if "closeweight_data_date" in out.columns:
            out["closeweight_data_date"] = out["closeweight_data_date"].map(normalize_date_key)
        else:
            raise RuntimeError("standardized weight CSV must contain closeweight_data_date.")
        out["raw_weight_pct"] = pd.to_numeric(out["raw_weight_pct"], errors="coerce")
    elif {"stock_code", "weight_pct"} <= set(frame.columns):
        date_col = "trade_date" if "trade_date" in frame.columns else "date" if "date" in frame.columns else None
        if date_col is None:
            raise RuntimeError("historical weight CSV must contain trade_date or date.")
        out = pd.DataFrame({
            "stock_code": frame["stock_code"].map(normalize_stock_code),
            "stock_name": frame.get("stock_name", pd.Series("", index=frame.index)).astype(str).str.strip(),
            "raw_weight_pct": pd.to_numeric(frame["weight_pct"], errors="coerce"),
            "closeweight_data_date": frame[date_col].map(normalize_date_key),
        })
    else:
        missing = [col for col in (CSI_CODE_FIELD, CSI_NAME_FIELD, CSI_WEIGHT_FIELD) if col not in frame.columns]
        if missing:
            raise RuntimeError(f"CSI weight file missing columns: {missing}. Actual columns={frame.columns.tolist()}")
        data_date = _extract_single_csi_data_date(frame, "CSI closeweight file")
        out = pd.DataFrame({
            "stock_code": frame[CSI_CODE_FIELD].map(normalize_stock_code),
            "stock_name": frame[CSI_NAME_FIELD].astype(str).str.strip(),
            "raw_weight_pct": pd.to_numeric(frame[CSI_WEIGHT_FIELD], errors="coerce"),
            "closeweight_data_date": data_date,
        })

    out = out[["stock_code", "stock_name", "raw_weight_pct", "closeweight_data_date"]].copy()
    out["source_path"] = str(source_path) if source_path else ""
    out = out.dropna(subset=["stock_code", "raw_weight_pct"])
    if out.empty:
        raise RuntimeError("standardized weight source has no valid rows.")
    if out["stock_code"].duplicated().any():
        dup = out.loc[out["stock_code"].duplicated(), "stock_code"].tolist()
        raise RuntimeError(f"weight source contains duplicate stock codes: {dup[:10]}")
    if (out["raw_weight_pct"] < 0).any():
        raise RuntimeError("weight source contains negative weights.")
    source_dates = sorted(out["closeweight_data_date"].dropna().unique())
    if len(source_dates) != 1:
        raise RuntimeError(f"weight source date is not unique: {source_dates}")
    return out.sort_values("stock_code").reset_index(drop=True)


def load_weight_source_file(file_path: str | Path) -> tuple[pd.DataFrame, Path]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raw = read_csi_file(str(path))
    return standardize_weight_frame(raw, source_path=path), path


def load_or_download_weight_source(
    *,
    index_code: str,
    file_path: str | Path | None,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_path:
        path = Path(file_path)
    else:
        downloaded = download_csi_constituent_v02(index_code, str(output_dir), "closeweight")
        if not downloaded:
            raise RuntimeError("CSI closeweight download failed.")
        path = Path(downloaded)

    return load_weight_source_file(path)


def select_weight_source_for_target(
    *,
    target_date: str,
    latest_weights: pd.DataFrame,
    latest_source_path: str | Path,
    historical_weight_files: dict[str, str | Path],
) -> WeightSourceSelection:
    """Choose the nearest available source weight date on or before target_date.

    The freshly downloaded CSI file remains authoritative whenever its embedded
    weight date is on or before the target.  For an older target, only the local
    historical files are considered.
    """
    target_date = normalize_date_key(target_date)
    latest_source_path = Path(latest_source_path)
    latest_weights = standardize_weight_frame(latest_weights, source_path=latest_source_path)
    latest_source_date = normalize_date_key(latest_weights["closeweight_data_date"].iloc[0])

    if latest_source_date <= target_date:
        return WeightSourceSelection(
            weights=latest_weights,
            source_path=latest_source_path,
            source_date=latest_source_date,
            selection_mode="latest_download",
            latest_source_date=latest_source_date,
        )

    configured_files = {
        normalize_date_key(source_date): Path(path)
        for source_date, path in historical_weight_files.items()
    }
    eligible_dates = sorted(source_date for source_date in configured_files if source_date <= target_date)
    if not eligible_dates:
        configured_dates = sorted(configured_files)
        raise RuntimeError(
            f"No historical weight source is available on or before target_date {target_date}. "
            f"Configured historical dates: {configured_dates}"
        )

    selected_date = eligible_dates[-1]
    selected_weights, selected_path = load_weight_source_file(configured_files[selected_date])
    embedded_date = normalize_date_key(selected_weights["closeweight_data_date"].iloc[0])
    if embedded_date != selected_date:
        raise RuntimeError(
            f"Historical weight file date mismatch: configured={selected_date}, "
            f"embedded={embedded_date}, path={selected_path}"
        )

    return WeightSourceSelection(
        weights=selected_weights,
        source_path=selected_path,
        source_date=selected_date,
        selection_mode="historical_local",
        latest_source_date=latest_source_date,
    )


def fetch_stock_closes_range(gogoal_query, stock_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    stripped_codes = [strip_market_suffix(code) for code in stock_codes]
    sql = f"""
        SELECT trade_date, stock_code, stock_name, tclose, is_valid
        FROM qt_stk_daily
        WHERE trade_date >= {sql_literal(normalize_date_dash(start_date))}
          AND trade_date <= {sql_literal(normalize_date_dash(end_date))}
          AND stock_code IN ({sql_in(stripped_codes)})
          AND is_valid = 1
    """
    raw = gogoal_query(sql, output_format="dataframe")
    if raw.empty:
        raise RuntimeError("Go-Goal daily stock close query returned empty.")
    raw = raw.copy()
    raw["stock_code"] = raw["stock_code"].map(normalize_stock_code)
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
    raw["close_price"] = pd.to_numeric(raw["tclose"], errors="coerce")
    return raw[["trade_date", "stock_code", "stock_name", "close_price"]].dropna(subset=["trade_date", "stock_code"])


def fetch_corporate_actions(gogoal_query, stock_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    stripped_codes = [strip_market_suffix(code) for code in stock_codes]
    sql = f"""
        SELECT
            stock_code,
            stock_name,
            declare_date,
            xr_xd_date AS ex_date,
            beftax_maxcashdiv,
            beftax_mincashdiv,
            aftax_cashdiv,
            stockdiv_ratio,
            trans_ratio,
            bonus_ratio,
            is_newest,
            is_valid
        FROM bas_stk_hisdistribution
        WHERE stock_code IN ({sql_in(stripped_codes)})
          AND xr_xd_date > {sql_literal(normalize_date_dash(start_date))}
          AND xr_xd_date <= {sql_literal(normalize_date_dash(end_date))}
          AND is_valid = 1
    """
    raw = gogoal_query(sql, output_format="dataframe")
    if raw.empty:
        return pd.DataFrame(columns=["stock_code", "ex_date"])
    raw = raw.copy()
    raw["stock_code"] = raw["stock_code"].map(normalize_stock_code)
    raw["ex_date"] = pd.to_datetime(raw["ex_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return raw


def project_weights_by_close_and_actions(
    source_weights: pd.DataFrame,
    daily_closes: pd.DataFrame,
    corporate_actions: pd.DataFrame | None,
    *,
    target_date: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    weights = standardize_weight_frame(source_weights)
    base_date = normalize_date_key(weights["closeweight_data_date"].iloc[0])
    target_date = normalize_date_key(target_date)
    if target_date < base_date:
        raise ValueError(f"target_date {target_date} is before source weight date {base_date}.")

    stock_codes = weights["stock_code"].tolist()
    closes = daily_closes.copy()
    closes["trade_date"] = closes["trade_date"].map(normalize_date_key)
    closes["stock_code"] = closes["stock_code"].map(normalize_stock_code)
    closes["close_price"] = pd.to_numeric(closes["close_price"], errors="coerce")
    closes = closes.dropna(subset=["trade_date", "stock_code", "close_price"])
    closes = closes[closes["stock_code"].isin(stock_codes)]

    trading_dates = sorted(d for d in closes["trade_date"].unique() if base_date <= d <= target_date)
    if base_date not in trading_dates:
        raise RuntimeError(f"daily closes do not contain source weight date {base_date}.")
    if target_date not in trading_dates:
        raise RuntimeError(f"daily closes do not contain target_date {target_date}; it may not be a trading day.")

    close_matrix = (
        closes.pivot_table(index="trade_date", columns="stock_code", values="close_price", aggfunc="last")
        .reindex(index=trading_dates, columns=stock_codes)
        .ffill()
    )
    if close_matrix.loc[base_date].isna().any():
        missing = close_matrix.columns[close_matrix.loc[base_date].isna()].tolist()
        raise RuntimeError(f"base date close is missing for {len(missing)} stocks: {missing[:10]}")

    actions = standardize_corporate_actions(corporate_actions)
    actions = actions[
        actions["stock_code"].isin(stock_codes)
        & actions["ex_date"].gt(base_date)
        & actions["ex_date"].le(target_date)
    ].copy()

    base_close = close_matrix.loc[base_date].astype(float)
    base_weight = weights.set_index("stock_code")["raw_weight_pct"].astype(float) / 100.0
    stock_name = weights.set_index("stock_code")["stock_name"]
    share_factor = pd.Series(1.0, index=stock_codes, dtype=float)
    cumulative_cash_per_base_share = pd.Series(0.0, index=stock_codes, dtype=float)

    trace: dict[str, pd.DataFrame] = {}
    summary_rows = []
    target_weights = None

    for i, trade_date in enumerate(trading_dates):
        if trade_date > base_date and not actions.empty:
            day_actions = actions.loc[actions["ex_date"].eq(trade_date)]
            for row in day_actions.itertuples(index=False):
                code = row.stock_code
                before_factor = share_factor.at[code]
                cash_per_share = float(row.cash_dividend_per_share or 0.0)
                share_ratio = float(row.share_increase_ratio or 0.0)
                cumulative_cash_per_base_share.at[code] += before_factor * cash_per_share
                share_factor.at[code] = before_factor * (1.0 + share_ratio)

        close_price = close_matrix.loc[trade_date].astype(float)
        relative_value = base_weight * share_factor * close_price / base_close
        projected_weight = relative_value / relative_value.sum()
        frame = pd.DataFrame({
            "trade_date": trade_date,
            "stock_code": stock_codes,
            "stock_name": stock_name.reindex(stock_codes).values,
            "source_weight_date": base_date,
            "source_weight_pct": (base_weight.reindex(stock_codes) * 100.0).values,
            "projected_weight_pct": (projected_weight.reindex(stock_codes) * 100.0).values,
            "base_close": base_close.reindex(stock_codes).values,
            "close_price": close_price.reindex(stock_codes).values,
            "share_factor": share_factor.reindex(stock_codes).values,
            "relative_value": relative_value.reindex(stock_codes).values,
            "cumulative_cash_dividend_per_base_share": cumulative_cash_per_base_share.reindex(stock_codes).values,
        }).sort_values("projected_weight_pct", ascending=False).reset_index(drop=True)

        daily_actions = actions.loc[actions["ex_date"].eq(trade_date)] if not actions.empty else pd.DataFrame()
        summary_rows.append({
            "trade_date": trade_date,
            "stock_count": int(len(frame)),
            "weight_sum_pct": float(frame["projected_weight_pct"].sum()),
            "action_count": int(len(daily_actions)),
            "share_action_count": int((pd.to_numeric(daily_actions.get("share_increase_ratio", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()),
            "cash_dividend_action_count": int((pd.to_numeric(daily_actions.get("cash_dividend_per_share", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).sum()),
        })

        if trade_date == target_date:
            target_weights = frame
        # Include the reference-date frame so downloaded and projected weights
        # are always available through the same 000300-YYYYMMDD.csv schema.
        trace[trade_date] = frame

    if target_weights is None:
        raise RuntimeError(f"target_date {target_date} was not projected.")
    daily_summary = pd.DataFrame(summary_rows)
    return target_weights, trace, daily_summary, actions


def save_projection_outputs(
    *,
    target_weights: pd.DataFrame,
    trace: dict[str, pd.DataFrame],
    daily_summary: pd.DataFrame,
    source_weights: pd.DataFrame,
    standardized_actions: pd.DataFrame,
    output_dir: str | Path,
    target_date: str,
    index_code: str,
) -> ProjectionOutput:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_date = normalize_date_key(target_date)

    target_path = output_dir / f"{index_code}-{target_date}.csv"
    summary_path = output_dir / f"{index_code}-{target_date}-daily_summary.csv"
    source_path = output_dir / f"{index_code}-{target_date}-source_weights.csv"
    actions_path = output_dir / f"{index_code}-{target_date}-standardized_actions.csv"

    # Save the reference date and every projected trading day in one schema.
    # Each file uses current_date's close price and represents the estimated
    # weights for the next index day.
    for current_date, frame in trace.items():
        frame.to_csv(output_dir / f"{index_code}-{current_date}.csv", index=False, encoding="utf-8-sig")

    # Keep explicit target and audit files for quick lookup and reproducibility.
    target_weights.to_csv(target_path, index=False, encoding="utf-8-sig")
    # daily_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    # source_weights.to_csv(source_path, index=False, encoding="utf-8-sig")
    # standardized_actions.to_csv(actions_path, index=False, encoding="utf-8-sig")

    return ProjectionOutput(
        target_weights=target_weights,
        trace=trace,
        daily_summary=daily_summary,
        source_weights=source_weights,
        standardized_actions=standardized_actions,
        output_dir=output_dir,
        target_path=target_path,
        trace_dir=output_dir,
        summary_path=summary_path,
    )


def make_projection_output_dir(project_root: str | Path, index_code: str | None = None, target_date: str | None = None) -> Path:
    return Path(project_root) / "data" / "weights_projection"
