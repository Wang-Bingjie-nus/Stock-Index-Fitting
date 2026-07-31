from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing as mp
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .file_utils import get_tick_file_path
from .minute_tick_cache_v03 import (
    CACHE_KIND,
    CACHE_SCHEMA_VERSION,
    CACHE_VALUE_FIELDS,
    CONTINUOUS_TRADING_MINUTE_ROWS,
    MINUTE_ALIGNMENT,
    RAW_CACHE_COLUMNS,
    build_unavailable_minute_tick_cache,
    build_minute_tick_cache,
    build_trading_minute_index,
    load_minute_tick_cache,
    load_unavailable_minute_tick_cache,
    normalize_stock_codes,
    normalize_trade_date,
    save_minute_tick_cache,
    save_unavailable_minute_tick_cache,
    unavailable_minute_tick_cache_path,
    validate_minute_tick_cache,
)


CSI300_INDEX_CODE = "000300"
CSI300_WEIGHT_PATTERN = re.compile(r"^沪深300_样本权重_(\d{8})\.csv$", re.IGNORECASE)
UNAVAILABLE_SUMMARY_COLUMNS = (
    "trade_date",
    "reason",
    "error_type",
    "message",
    "errors",
    "source_files",
    "details",
    "marker_file",
    "created_at_utc",
)


@dataclass(frozen=True)
class TradingDateResolution:
    configured_start_date: str
    configured_end_date: str
    adjusted_start_date: str
    adjusted_end_date: str
    trade_dates: tuple[str, ...]


@dataclass(frozen=True)
class MarketCacheTask:
    trade_date: str
    market: str
    source_path: str
    stock_codes: tuple[str, ...]
    partition_cache_path: str
    force_rebuild: bool = False


def _normalize_index_code(index_code: str) -> str:
    digits = re.sub(r"\D", "", str(index_code))[:6]
    if digits != CSI300_INDEX_CODE:
        raise ValueError(
            f"cache_generator currently supports CSI 300 ({CSI300_INDEX_CODE}) only, "
            f"got {index_code!r}."
        )
    return digits


def _calendar_item_to_trade_date(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8 and digits[:4] in {str(year) for year in range(1990, 2101)}:
        candidate = digits[:8]
        try:
            return normalize_trade_date(candidate)
        except ValueError:
            return None

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


def _normalize_calendar_dates(values: Any) -> list[str]:
    dates = {
        parsed
        for parsed in (_calendar_item_to_trade_date(value) for value in (values or []))
        if parsed is not None
    }
    return sorted(dates)


def get_previous_closest_trading_date(
    current_date: str,
    *,
    xtdata_client: Any | None = None,
) -> str:
    """Return the closest SH trading day strictly before ``current_date``."""

    current_date = normalize_trade_date(current_date)
    if xtdata_client is None:
        from xtquant import xtdata as xtdata_client

    target = pd.Timestamp(current_date)
    start_date = (target - pd.Timedelta(days=366)).strftime("%Y%m%d")
    calendar = xtdata_client.get_trading_calendar(
        "SH",
        start_time=start_date,
        end_time=current_date,
    )
    available_dates = [
        trade_date
        for trade_date in _normalize_calendar_dates(calendar)
        if trade_date < current_date
    ]
    if not available_dates:
        raise RuntimeError(f"Cannot find an SH trading day before {current_date}.")
    return available_dates[-1]


def resolve_trading_dates(
    start_date: str,
    end_date: str,
    *,
    xtdata_client: Any | None = None,
) -> TradingDateResolution:
    """Resolve an inclusive SH trading-date range using the XtQuant calendar."""

    configured_start = normalize_trade_date(start_date)
    configured_end = normalize_trade_date(end_date)
    if configured_start > configured_end:
        raise ValueError("start_date must not be later than end_date.")
    if xtdata_client is None:
        from xtquant import xtdata as xtdata_client

    adjusted_start = configured_start
    start_calendar = xtdata_client.get_trading_calendar(
        "SH",
        start_time=configured_start,
        end_time=configured_start,
    )
    if configured_start not in _normalize_calendar_dates(start_calendar):
        adjusted_start = get_previous_closest_trading_date(
            configured_start,
            xtdata_client=xtdata_client,
        )

    adjusted_end = configured_end
    end_calendar = xtdata_client.get_trading_calendar(
        "SH",
        start_time=configured_end,
        end_time=configured_end,
    )
    if configured_end not in _normalize_calendar_dates(end_calendar):
        adjusted_end = get_previous_closest_trading_date(
            configured_end,
            xtdata_client=xtdata_client,
        )

    if adjusted_start > adjusted_end:
        raise RuntimeError(
            f"Adjusted start date {adjusted_start} is later than adjusted end date {adjusted_end}."
        )

    calendar = xtdata_client.get_trading_calendar(
        "SH",
        start_time=adjusted_start,
        end_time=adjusted_end,
    )
    trade_dates = tuple(
        date
        for date in _normalize_calendar_dates(calendar)
        if adjusted_start <= date <= adjusted_end
    )
    if not trade_dates:
        raise RuntimeError(
            f"XtQuant returned no SH trading dates in {adjusted_start}..{adjusted_end}."
        )

    return TradingDateResolution(
        configured_start_date=configured_start,
        configured_end_date=configured_end,
        adjusted_start_date=adjusted_start,
        adjusted_end_date=adjusted_end,
        trade_dates=trade_dates,
    )


def discover_csi300_weight_files(weights_dir: str | Path) -> pd.DataFrame:
    """List available CSI 300 monthly sample-weight files."""

    weights_dir = Path(weights_dir)
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights directory does not exist: {weights_dir}")

    records = []
    for path in weights_dir.iterdir():
        if not path.is_file():
            continue
        match = CSI300_WEIGHT_PATTERN.fullmatch(path.name)
        if match:
            records.append({"weight_date": match.group(1), "weight_path": path})
    if not records:
        raise FileNotFoundError(
            f"No files matching '沪深300_样本权重_YYYYMMDD.csv' in {weights_dir}."
        )
    return pd.DataFrame(records).sort_values("weight_date").reset_index(drop=True)


def resolve_csi300_weight_file(
    trade_date: str,
    weights_dir: str | Path,
) -> Path:
    """Use the latest sample-weight file from the previous calendar month."""

    trade_date = normalize_trade_date(trade_date)
    first_day = pd.Timestamp(trade_date).replace(day=1)
    previous_month_day = first_day - pd.Timedelta(days=1)
    previous_month = previous_month_day.strftime("%Y%m")

    files = discover_csi300_weight_files(weights_dir)
    matches = files.loc[files["weight_date"].str[:6].eq(previous_month)]
    if matches.empty:
        available = files["weight_date"].tolist()
        raise FileNotFoundError(
            f"No CSI 300 sample-weight file for previous month {previous_month}; "
            f"trade_date={trade_date}, available={available}."
        )
    return Path(matches.iloc[-1]["weight_path"])


def load_csi300_stock_codes(
    trade_date: str,
    weights_dir: str | Path,
    *,
    require_component_count: int = 300,
) -> tuple[list[str], Path]:
    """Load the exact CSI 300 universe used for one trade date."""

    trade_date = normalize_trade_date(trade_date)
    weight_path = resolve_csi300_weight_file(trade_date, weights_dir)
    frame = pd.read_csv(weight_path, dtype={"stock_code": str, "index_code": str})
    if "stock_code" not in frame.columns:
        raise ValueError(f"Weight file has no stock_code column: {weight_path}")

    if "index_code" in frame.columns:
        index_codes = {
            re.sub(r"\D", "", str(value))[:6]
            for value in frame["index_code"].dropna().unique()
        }
        if index_codes != {CSI300_INDEX_CODE}:
            raise ValueError(
                f"Unexpected index_code values in {weight_path}: {sorted(index_codes)}"
            )

    stock_codes = normalize_stock_codes(frame["stock_code"].dropna().tolist())
    if require_component_count and len(stock_codes) != int(require_component_count):
        raise ValueError(
            f"Expected {require_component_count} unique CSI 300 constituents in {weight_path}, "
            f"got {len(stock_codes)}."
        )
    return stock_codes, weight_path


def _cache_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def final_cache_path(
    cache_dir: str | Path,
    index_code: str,
    trade_date: str,
    stock_codes: list[str] | tuple[str, ...],
) -> Path:
    """Return the exact filename expected by correlation_v03.ipynb."""

    index_code = _normalize_index_code(index_code)
    trade_date = normalize_trade_date(trade_date)
    tick_codes = normalize_stock_codes(stock_codes)
    payload = {
        "kind": "basket_tick_v03",
        "index_code": index_code,
        "fitting_date": trade_date,
        "tick_codes": tick_codes,
    }
    return Path(cache_dir) / f"basket_minute_wide_{trade_date}_{_cache_hash(payload)}.pkl"


def _publish_unavailable_marker(
    cache_dir: str | Path,
    trade_date: str,
    *,
    reason: str,
    error_type: str | None = None,
    details: dict[str, Any] | None = None,
    stock_codes=None,
) -> Path:
    marker_path = unavailable_minute_tick_cache_path(cache_dir, trade_date)
    marker = build_unavailable_minute_tick_cache(
        trade_date,
        reason=reason,
        error_type=error_type,
        details=details,
        stock_codes=stock_codes,
        generated_by="utils.cache_generator",
    )
    return save_unavailable_minute_tick_cache(marker, marker_path)


def _remove_unavailable_marker(cache_dir: str | Path, trade_date: str) -> None:
    marker_path = unavailable_minute_tick_cache_path(cache_dir, trade_date)
    if marker_path.exists():
        try:
            marker_path.unlink()
        except OSError as exc:
            print(
                f"  [{normalize_trade_date(trade_date)}] warning: complete cache is valid, "
                f"but stale unavailable marker could not be deleted: {exc}"
            )


def unavailable_cache_summary_path(
    cache_dir: str | Path,
    trade_dates: list[str] | tuple[str, ...],
) -> Path:
    """Return the CSV path for one requested cache-generation batch."""

    normalized_dates = [normalize_trade_date(value) for value in trade_dates]
    if not normalized_dates:
        label = "empty"
    else:
        label = f"{min(normalized_dates)}_{max(normalized_dates)}_{len(normalized_dates)}d"
    return Path(cache_dir) / f"unavailable_cache_summary_{label}.csv"


def write_unavailable_cache_summary(
    cache_dir: str | Path,
    trade_dates: list[str] | tuple[str, ...],
) -> Path:
    """Write unavailable reasons for the requested dates, including marker-hit dates."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_dates = [normalize_trade_date(value) for value in trade_dates]
    rows: list[dict[str, Any]] = []

    for trade_date in normalized_dates:
        marker_path = unavailable_minute_tick_cache_path(cache_dir, trade_date)
        if not marker_path.is_file():
            continue
        try:
            marker = load_unavailable_minute_tick_cache(
                marker_path,
                trade_date=trade_date,
            )
            metadata = marker["metadata"]
            details = dict(metadata.get("details") or {})
            rows.append(
                {
                    "trade_date": trade_date,
                    "reason": str(metadata.get("reason") or ""),
                    "error_type": str(metadata.get("error_type") or ""),
                    "message": str(details.get("message") or ""),
                    "errors": json.dumps(
                        details.get("errors", []),
                        ensure_ascii=False,
                        default=str,
                    ),
                    "source_files": json.dumps(
                        details.get("source_files", []),
                        ensure_ascii=False,
                        default=str,
                    ),
                    "details": json.dumps(
                        details,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                    "marker_file": str(marker_path),
                    "created_at_utc": str(metadata.get("created_at_utc") or ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "trade_date": trade_date,
                    "reason": "unavailable_marker_read_failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "errors": "[]",
                    "source_files": "[]",
                    "details": json.dumps(
                        {"message": str(exc)},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "marker_file": str(marker_path),
                    "created_at_utc": "",
                }
            )

    report_path = unavailable_cache_summary_path(cache_dir, normalized_dates)
    pd.DataFrame(rows, columns=UNAVAILABLE_SUMMARY_COLUMNS).to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"[UNAVAILABLE SUMMARY] {len(rows)}/{len(normalized_dates)} date(s): "
        f"{report_path}"
    )
    return report_path


def _market_from_source_path(source_path: str | Path) -> str:
    match = re.search(r"_tick_(sh_zb|sh_kcb|sz_zb|sz_cyb|bj)\.pkl$", Path(source_path).name)
    if not match:
        raise ValueError(f"Cannot determine tick market from source file: {source_path}")
    return match.group(1)


def _partition_cache_path(
    partition_dir: str | Path,
    trade_date: str,
    market: str,
    source_path: str | Path,
    stock_codes: list[str],
) -> Path:
    source_path = Path(source_path)
    try:
        stat = source_path.stat()
        source_signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        source_signature = {"size": None, "mtime_ns": None}
    payload = {
        "kind": "basket_tick_partition_v03",
        "schema_version": CACHE_SCHEMA_VERSION,
        "trade_date": normalize_trade_date(trade_date),
        "market": market,
        "source_path": str(source_path),
        "source_signature": source_signature,
        "stock_codes": normalize_stock_codes(stock_codes),
    }
    return Path(partition_dir) / (
        f"minute_partition_{normalize_trade_date(trade_date)}_{market}_{_cache_hash(payload)}.pkl"
    )


def build_market_tasks(
    trade_date: str,
    stock_codes: list[str] | tuple[str, ...],
    *,
    source_tick_root: str | Path,
    partition_dir: str | Path,
    force_rebuild: bool = False,
) -> list[MarketCacheTask]:
    """Group a requested stock universe by its physical market pickle file."""

    trade_date = normalize_trade_date(trade_date)
    tick_codes = normalize_stock_codes(stock_codes)
    if not tick_codes:
        raise ValueError("stock_codes is empty.")

    grouped: dict[str, list[str]] = {}
    for code in tick_codes:
        source_path = get_tick_file_path(trade_date, code, str(source_tick_root))
        grouped.setdefault(str(Path(source_path)), []).append(code)

    tasks = []
    for source_path, codes in sorted(grouped.items()):
        market = _market_from_source_path(source_path)
        partition_path = _partition_cache_path(
            partition_dir,
            trade_date,
            market,
            source_path,
            codes,
        )
        tasks.append(
            MarketCacheTask(
                trade_date=trade_date,
                market=market,
                source_path=source_path,
                stock_codes=tuple(normalize_stock_codes(codes)),
                partition_cache_path=str(partition_path),
                force_rebuild=bool(force_rebuild),
            )
        )
    return tasks


def preview_generation_plan(
    trade_dates: list[str] | tuple[str, ...],
    *,
    weights_dir: str | Path,
    source_tick_root: str | Path,
    cache_dir: str | Path,
) -> pd.DataFrame:
    """Build a read-only per-date/per-source preview for the notebook."""

    rows = []
    partition_dir = Path(cache_dir) / "partitions"
    for trade_date in trade_dates:
        trade_date = normalize_trade_date(trade_date)
        try:
            stock_codes, weight_path = load_csi300_stock_codes(trade_date, weights_dir)
            tasks = build_market_tasks(
                trade_date,
                stock_codes,
                source_tick_root=source_tick_root,
                partition_dir=partition_dir,
            )
            target = final_cache_path(cache_dir, CSI300_INDEX_CODE, trade_date, stock_codes)
            for task in tasks:
                source = Path(task.source_path)
                rows.append(
                    {
                        "trade_date": trade_date,
                        "weight_file": weight_path.name,
                        "component_count": len(stock_codes),
                        "market": task.market,
                        "market_stock_count": len(task.stock_codes),
                        "source_path": task.source_path,
                        "source_exists": source.exists(),
                        "source_gb": round(source.stat().st_size / 1024**3, 3) if source.exists() else np.nan,
                        "final_cache_path": str(target),
                        "final_cache_exists": target.exists(),
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "trade_date": trade_date,
                    "status": "plan_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


def select_complete_trade_dates(
    plan: pd.DataFrame,
    trade_dates: list[str] | tuple[str, ...],
) -> tuple[list[str], dict[str, list[str]]]:
    """Keep only dates whose preview has no missing source or planning error."""

    requested_dates = [normalize_trade_date(value) for value in trade_dates]
    excluded_reasons: dict[str, list[str]] = {}

    if not isinstance(plan, pd.DataFrame) or "trade_date" not in plan.columns:
        return [], {
            trade_date: ["No preview rows were produced for this date."]
            for trade_date in requested_dates
        }

    normalized_plan = plan.copy()
    normalized_plan["trade_date"] = normalized_plan["trade_date"].map(
        lambda value: normalize_trade_date(value) if pd.notna(value) else ""
    )

    valid_dates = []
    for trade_date in requested_dates:
        date_rows = normalized_plan.loc[normalized_plan["trade_date"].eq(trade_date)]
        reasons = []

        if date_rows.empty:
            reasons.append("No preview rows were produced for this date.")
        else:
            if "status" in date_rows.columns:
                plan_errors = date_rows.loc[date_rows["status"].eq("plan_error")]
                for error in plan_errors.get("error", pd.Series(dtype=object)).dropna():
                    reasons.append(f"Preview planning error: {error}")

            if "source_exists" not in date_rows.columns:
                if not reasons:
                    reasons.append("Preview rows do not contain source_exists information.")
            else:
                non_error_rows = date_rows
                if "status" in date_rows.columns:
                    non_error_rows = date_rows.loc[~date_rows["status"].eq("plan_error")]
                source_exists = non_error_rows["source_exists"].astype("boolean").fillna(False)
                missing_rows = non_error_rows.loc[~source_exists]
                for source_path in missing_rows.get(
                    "source_path",
                    pd.Series(dtype=object),
                ).fillna("<unknown source path>"):
                    reasons.append(f"Missing source pickle: {source_path}")

        if reasons:
            excluded_reasons[trade_date] = reasons
        else:
            valid_dates.append(trade_date)

    return valid_dates, excluded_reasons


def _build_market_partition(task: MarketCacheTask) -> tuple[str, float]:
    """Worker entry point: read, reduce, transform and save one source pickle."""

    started = time.perf_counter()
    source_path = Path(task.source_path)
    cache_path = Path(task.partition_cache_path)

    if cache_path.exists() and not task.force_rebuild:
        cache = load_minute_tick_cache(
            cache_path,
            trade_date=task.trade_date,
            stock_codes=task.stock_codes,
        )
        missing = cache["metadata"].get("missing_stocks", [])
        if not missing:
            return "cache_hit", time.perf_counter() - started

    if not source_path.is_file():
        raise FileNotFoundError(f"Source tick file does not exist: {source_path}")

    with source_path.open("rb") as handle:
        file_data = pickle.load(handle)
    if not isinstance(file_data, dict):
        raise TypeError(f"Source pickle must contain a dict, got {type(file_data).__name__}.")

    selected: dict[str, pd.DataFrame | None] = {}
    missing_stocks = []
    for code in task.stock_codes:
        frame = file_data.get(code)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            missing_stocks.append(code)
            selected[code] = None
        else:
            selected[code] = frame
    del file_data
    gc.collect()

    if missing_stocks:
        raise ValueError(
            f"{len(missing_stocks)} requested stocks are absent or empty in {source_path.name}: "
            f"{missing_stocks[:20]}"
        )

    cache = build_minute_tick_cache(selected, task.trade_date, task.stock_codes)
    del selected
    cache["metadata"].update(
        {
            "partition_market": task.market,
            "source_file": str(source_path),
            "generated_by": "utils.cache_generator",
        }
    )
    save_minute_tick_cache(cache, cache_path)
    return "built", time.perf_counter() - started


def _combine_partition_caches(
    tasks: list[MarketCacheTask],
    expected_stock_codes: list[str],
    trade_date: str,
) -> dict[str, Any]:
    trade_date = normalize_trade_date(trade_date)
    expected_stock_codes = normalize_stock_codes(expected_stock_codes)
    frames = []
    seen_codes: set[str] = set()

    for task in tasks:
        cache = load_minute_tick_cache(
            task.partition_cache_path,
            trade_date=trade_date,
            stock_codes=task.stock_codes,
        )
        missing = cache["metadata"].get("missing_stocks", [])
        if missing:
            raise ValueError(f"Partition {task.market} contains missing stocks: {missing}")
        overlap = seen_codes.intersection(task.stock_codes)
        if overlap:
            raise ValueError(f"Stock codes appear in multiple partitions: {sorted(overlap)}")
        seen_codes.update(task.stock_codes)
        frames.append(cache["data"])

    if seen_codes != set(expected_stock_codes):
        missing = sorted(set(expected_stock_codes) - seen_codes)
        unexpected = sorted(seen_codes - set(expected_stock_codes))
        raise ValueError(
            f"Partition stock universe mismatch; missing={missing[:20]}, unexpected={unexpected[:20]}."
        )

    minute_index = build_trading_minute_index(trade_date)
    wide = pd.concat(frames, axis=1)
    if wide.columns.duplicated().any():
        duplicates = wide.columns[wide.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate final cache columns: {duplicates[:20]}")
    expected_columns = pd.MultiIndex.from_product(
        [CACHE_VALUE_FIELDS, expected_stock_codes],
        names=["field", "stock_code"],
    )
    wide = wide.reindex(index=minute_index, columns=expected_columns)
    wide.index = wide.index.astype("int64")

    cache = {
        "metadata": {
            "kind": CACHE_KIND,
            "schema_version": CACHE_SCHEMA_VERSION,
            "trade_date": trade_date,
            "stock_codes": expected_stock_codes,
            "source_fields": list(RAW_CACHE_COLUMNS),
            "cached_value_fields": list(CACHE_VALUE_FIELDS),
            "time_index_dtype": "int64",
            "minute_alignment": MINUTE_ALIGNMENT,
            "minute_rows": int(len(wide)),
            "missing_stocks": [],
            "partition_count": len(tasks),
            "partition_markets": [task.market for task in tasks],
            "generated_by": "utils.cache_generator",
        },
        "data": wide,
    }
    return validate_minute_tick_cache(
        cache,
        trade_date=trade_date,
        stock_codes=expected_stock_codes,
    )


def generate_trade_date_cache(
    trade_date: str,
    stock_codes: list[str] | tuple[str, ...],
    *,
    index_code: str = CSI300_INDEX_CODE,
    source_tick_root: str | Path,
    cache_dir: str | Path,
    max_workers: int = 4,
    force_rebuild: bool = False,
) -> Path | None:
    """Generate one cache, or publish a date-level unavailable marker on failure."""
    start_time = time.perf_counter()
    index_code = _normalize_index_code(index_code)
    trade_date = normalize_trade_date(trade_date)
    stock_codes = normalize_stock_codes(stock_codes)
    if not stock_codes:
        raise ValueError("stock_codes is empty.")
    max_workers = int(max_workers)
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")

    cache_dir = Path(cache_dir)
    partition_dir = cache_dir / "partitions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    partition_dir.mkdir(parents=True, exist_ok=True)
    target = final_cache_path(cache_dir, index_code, trade_date, stock_codes)

    if target.exists() and not force_rebuild:
        try:
            cache = load_minute_tick_cache(
                target,
                trade_date=trade_date,
                stock_codes=stock_codes,
            )
            missing = cache["metadata"].get("missing_stocks", [])
            if not missing:
                _remove_unavailable_marker(cache_dir, trade_date)
                print(f"  [final:all] cache_hit: {target.name}")
                return target
        except Exception as exc:
            print(f"  [final:all] invalid cache; rebuilding: {type(exc).__name__}: {exc}")

    marker_path = unavailable_minute_tick_cache_path(cache_dir, trade_date)
    if marker_path.exists() and not force_rebuild:
        try:
            marker = load_unavailable_minute_tick_cache(
                marker_path,
                trade_date=trade_date,
            )
            print(
                f"  [unavailable:{trade_date}] marker_hit: {marker_path.name}; "
                f"reason={marker['metadata']['reason']}"
            )
            return None
        except Exception as exc:
            print(
                f"  [unavailable:{trade_date}] invalid marker; retrying source data: "
                f"{type(exc).__name__}: {exc}"
            )

    errors: list[str] = []
    tasks: list[MarketCacheTask] = []
    try:
        tasks = build_market_tasks(
            trade_date,
            stock_codes,
            source_tick_root=source_tick_root,
            partition_dir=partition_dir,
            force_rebuild=force_rebuild,
        )
        missing_source_paths = sorted({
            task.source_path
            for task in tasks
            if not Path(task.source_path).is_file()
        })
        if missing_source_paths:
            marker_path = _publish_unavailable_marker(
                cache_dir,
                trade_date,
                reason="source_file_missing",
                error_type="FileNotFoundError",
                details={"source_files": missing_source_paths},
                stock_codes=stock_codes,
            )
            print(
                f"  [unavailable:{trade_date}] source files missing; "
                f"marker={marker_path.name}; files={missing_source_paths}"
            )
            return None

        worker_count = min(max_workers, len(tasks))
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            future_to_task = {
                executor.submit(_build_market_partition, task): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    status, elapsed = future.result()
                    print(f"  [partition:{task.market}] {status} ({elapsed:.1f}s)")
                except Exception as exc:
                    message = f"partition {task.market}: {type(exc).__name__}: {exc}"
                    errors.append(message)
                    print(
                        f"  [partition:{task.market}] error\n    "
                        f"{type(exc).__name__}: {exc}"
                    )

        if errors:
            raise RuntimeError(
                f"Cannot publish complete cache for {trade_date}; "
                + "; ".join(errors)
            )

        final_cache = _combine_partition_caches(tasks, stock_codes, trade_date)
        save_minute_tick_cache(final_cache, target)
        validate_minute_tick_cache(
            load_minute_tick_cache(target),
            trade_date=trade_date,
            stock_codes=stock_codes,
        )
        _remove_unavailable_marker(cache_dir, trade_date)
        print(
            f"  [final:all] built ({time.perf_counter() - start_time:.1f}s): "
            f"{target.name}"
        )
        return target
    except Exception as exc:
        reason = "partition_generation_failed" if errors else "complete_cache_generation_failed"
        details = {
            "message": str(exc),
            "errors": list(errors),
            "source_files": [task.source_path for task in tasks],
        }
        marker_path = _publish_unavailable_marker(
            cache_dir,
            trade_date,
            reason=reason,
            error_type=type(exc).__name__,
            details=details,
            stock_codes=stock_codes,
        )
        print(
            f"  [unavailable:{trade_date}] complete cache not published; "
            f"marker={marker_path.name}; {type(exc).__name__}: {exc}"
        )
        return None


def generate_csi300_caches(
    trade_dates: list[str] | tuple[str, ...],
    *,
    weights_dir: str | Path,
    source_tick_root: str | Path,
    cache_dir: str | Path,
    max_workers: int = 4,
    force_rebuild: bool = False,
) -> list[Path]:
    """Generate CSI 300 caches sequentially by date and in parallel by source file."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    normalized_dates = [normalize_trade_date(value) for value in trade_dates]
    for date_number, trade_date in enumerate(normalized_dates, start=1):
        trade_date = normalize_trade_date(trade_date)
        print(
            f"[DATE {date_number}/{len(normalized_dates)}] {trade_date}: "
            f"preparing CSI 300 cache with max_workers={max_workers}"
        )
        stock_codes: list[str] = []
        try:
            stock_codes, weight_path = load_csi300_stock_codes(trade_date, weights_dir)
            print(f"  weights={weight_path.name}, components={len(stock_codes)}")
            generated_path = generate_trade_date_cache(
                trade_date,
                stock_codes,
                index_code=CSI300_INDEX_CODE,
                source_tick_root=source_tick_root,
                cache_dir=cache_dir,
                max_workers=max_workers,
                force_rebuild=force_rebuild,
            )
            if generated_path is not None:
                generated_paths.append(generated_path)
        except Exception as exc:
            marker_path = unavailable_minute_tick_cache_path(cache_dir, trade_date)
            if marker_path.exists() and not force_rebuild:
                try:
                    marker = load_unavailable_minute_tick_cache(
                        marker_path,
                        trade_date=trade_date,
                    )
                    print(
                        f"  [unavailable:{trade_date}] marker_hit: {marker_path.name}; "
                        f"reason={marker['metadata']['reason']}"
                    )
                    continue
                except Exception:
                    pass
            marker_path = _publish_unavailable_marker(
                cache_dir,
                trade_date,
                reason="complete_cache_generation_failed",
                error_type=type(exc).__name__,
                details={"message": str(exc)},
                stock_codes=stock_codes,
            )
            print(
                f"  [unavailable:{trade_date}] generation failed; "
                f"marker={marker_path.name}; {type(exc).__name__}: {exc}"
            )

    write_unavailable_cache_summary(cache_dir, normalized_dates)
    return generated_paths


def find_compatible_partition_cache(
    task: MarketCacheTask,
    partition_dir: str | Path,
) -> Path:
    """Find the newest valid partition matching a market task exactly."""

    partition_dir = Path(partition_dir)
    expected_path = Path(task.partition_cache_path)
    fallback_paths = sorted(
        partition_dir.glob(f"minute_partition_{task.trade_date}_{task.market}_*.pkl"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )

    candidate_paths = []
    seen_paths: set[Path] = set()
    for path in [expected_path, *fallback_paths]:
        path = Path(path)
        if path not in seen_paths and path.is_file():
            seen_paths.add(path)
            candidate_paths.append(path)

    if not candidate_paths:
        raise FileNotFoundError(
            f"No partition cache found for date={task.trade_date}, market={task.market}; "
            f"expected={expected_path}."
        )

    validation_errors = []
    for candidate_path in candidate_paths:
        try:
            cache = load_minute_tick_cache(
                candidate_path,
                trade_date=task.trade_date,
                stock_codes=task.stock_codes,
            )
            missing_stocks = cache["metadata"].get("missing_stocks", [])
            if missing_stocks:
                raise ValueError(f"Partition contains missing stocks: {missing_stocks[:20]}")
            partition_market = cache["metadata"].get("partition_market")
            if partition_market is not None and partition_market != task.market:
                raise ValueError(
                    f"Partition market mismatch: {partition_market!r} != {task.market!r}."
                )
            return candidate_path
        except Exception as exc:
            validation_errors.append(
                f"{candidate_path.name}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        f"No valid partition cache for date={task.trade_date}, market={task.market}; "
        f"errors={validation_errors}."
    )


def _validate_complete_cache(
    cache_path: str | Path,
    trade_date: str,
    stock_codes: list[str],
) -> dict[str, Any]:
    cache = load_minute_tick_cache(
        cache_path,
        trade_date=trade_date,
        stock_codes=stock_codes,
    )
    missing_stocks = cache["metadata"].get("missing_stocks", [])
    if missing_stocks:
        raise ValueError(f"Complete cache contains missing stocks: {missing_stocks[:20]}")
    if len(cache["metadata"]["stock_codes"]) != len(stock_codes):
        raise ValueError("Complete cache stock count does not match the requested universe.")
    if int(cache["metadata"]["minute_rows"]) != CONTINUOUS_TRADING_MINUTE_ROWS:
        raise ValueError(
            "Complete cache must contain "
            f"{CONTINUOUS_TRADING_MINUTE_ROWS} minute rows, got "
            f"{cache['metadata']['minute_rows']}."
        )
    return cache


def _delete_partition_paths(paths: list[Path], trade_date: str) -> None:
    deletion_errors = []
    for path in paths:
        try:
            path.unlink()
            print(f"  [{trade_date}] deleted partition: {path.name}")
        except Exception as exc:
            deletion_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    if deletion_errors:
        raise RuntimeError(
            f"Complete cache is valid, but some partition files could not be deleted: "
            f"{deletion_errors}"
        )


def merge_partition_caches_for_date(
    trade_date: str,
    *,
    weights_dir: str | Path,
    source_tick_root: str | Path,
    cache_dir: str | Path,
    delete_partition_caches: bool = True,
    overwrite_complete_cache: bool = False,
) -> Path | None:
    """Merge one date's market partitions and optionally delete them after validation."""

    trade_date = normalize_trade_date(trade_date)
    cache_dir = Path(cache_dir)
    partition_dir = cache_dir / "partitions"
    marker_path = unavailable_minute_tick_cache_path(cache_dir, trade_date)
    try:
        stock_codes, weight_path = load_csi300_stock_codes(trade_date, weights_dir)
    except Exception as exc:
        if marker_path.exists() and not overwrite_complete_cache:
            try:
                marker = load_unavailable_minute_tick_cache(
                    marker_path,
                    trade_date=trade_date,
                )
                print(
                    f"  [{trade_date}] unavailable marker hit: {marker_path.name}; "
                    f"reason={marker['metadata']['reason']}"
                )
                return None
            except Exception:
                pass
        marker_path = _publish_unavailable_marker(
            cache_dir,
            trade_date,
            reason="complete_cache_merge_failed",
            error_type=type(exc).__name__,
            details={"message": str(exc)},
        )
        print(f"  [{trade_date}] merge unavailable: {marker_path.name}; {exc}")
        return None
    complete_path = final_cache_path(
        cache_dir,
        CSI300_INDEX_CODE,
        trade_date,
        stock_codes,
    )
    try:
        expected_tasks = build_market_tasks(
            trade_date,
            stock_codes,
            source_tick_root=source_tick_root,
            partition_dir=partition_dir,
            force_rebuild=False,
        )
    except Exception as exc:
        marker_path = _publish_unavailable_marker(
            cache_dir,
            trade_date,
            reason="complete_cache_merge_failed",
            error_type=type(exc).__name__,
            details={"message": str(exc)},
            stock_codes=stock_codes,
        )
        print(f"  [{trade_date}] merge unavailable: {marker_path.name}; {exc}")
        return None

    complete_cache_is_valid = False
    if complete_path.is_file() and not overwrite_complete_cache:
        try:
            _validate_complete_cache(complete_path, trade_date, stock_codes)
            complete_cache_is_valid = True
        except Exception as exc:
            print(
                f"  [{trade_date}] existing complete cache is invalid; "
                f"rebuilding: {type(exc).__name__}: {exc}"
            )

    if complete_cache_is_valid:
        _remove_unavailable_marker(cache_dir, trade_date)
        print(f"  [{trade_date}] complete cache hit: {complete_path.name}")
        if delete_partition_caches:
            available_partition_paths = []
            for task in expected_tasks:
                try:
                    available_partition_paths.append(
                        find_compatible_partition_cache(task, partition_dir)
                    )
                except (FileNotFoundError, RuntimeError):
                    continue
            _delete_partition_paths(available_partition_paths, trade_date)
        return complete_path

    if marker_path.exists() and not overwrite_complete_cache:
        try:
            marker = load_unavailable_minute_tick_cache(
                marker_path,
                trade_date=trade_date,
            )
            print(
                f"  [{trade_date}] unavailable marker hit: {marker_path.name}; "
                f"reason={marker['metadata']['reason']}"
            )
            return None
        except Exception as exc:
            print(
                f"  [{trade_date}] invalid unavailable marker; retrying merge: "
                f"{type(exc).__name__}: {exc}"
            )

    try:
        resolved_tasks = []
        used_partition_paths = []
        for task in expected_tasks:
            partition_path = find_compatible_partition_cache(task, partition_dir)
            resolved_tasks.append(replace(task, partition_cache_path=str(partition_path)))
            used_partition_paths.append(partition_path)
            print(
                f"  [{trade_date}] partition {task.market}: "
                f"stocks={len(task.stock_codes)}, file={partition_path.name}"
            )

        complete_cache = _combine_partition_caches(
            resolved_tasks,
            stock_codes,
            trade_date,
        )
        save_minute_tick_cache(complete_cache, complete_path)
        _validate_complete_cache(complete_path, trade_date, stock_codes)
    except Exception as exc:
        marker_path = _publish_unavailable_marker(
            cache_dir,
            trade_date,
            reason="complete_cache_merge_failed",
            error_type=type(exc).__name__,
            details={"message": str(exc)},
            stock_codes=stock_codes,
        )
        print(f"  [{trade_date}] merge unavailable: {marker_path.name}; {exc}")
        return None

    _remove_unavailable_marker(cache_dir, trade_date)
    print(
        f"  [{trade_date}] complete cache built: {complete_path.name} "
        f"({complete_path.stat().st_size / 1024**2:.2f} MB), "
        f"weights={weight_path.name}"
    )
    if delete_partition_caches:
        _delete_partition_paths(used_partition_paths, trade_date)
    return complete_path


def merge_partition_caches_for_range(
    start_date: str,
    end_date: str,
    *,
    xtdata_client: Any,
    weights_dir: str | Path,
    source_tick_root: str | Path,
    cache_dir: str | Path,
    delete_partition_caches: bool = True,
    overwrite_complete_cache: bool = False,
    excluded_trade_dates: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Merge one algorithm-compatible complete cache per SH trading day."""

    resolution = resolve_trading_dates(
        start_date,
        end_date,
        xtdata_client=xtdata_client,
    )
    excluded_date_set = {
        normalize_trade_date(value)
        for value in (excluded_trade_dates or [])
    }
    trade_dates = [
        trade_date
        for trade_date in resolution.trade_dates
        if trade_date not in excluded_date_set
    ]
    skipped_dates = [
        trade_date
        for trade_date in resolution.trade_dates
        if trade_date in excluded_date_set
    ]
    if skipped_dates:
        print(f"Merge skipped incomplete dates: {skipped_dates}")
    print(
        f"Merge trading dates ({len(trade_dates)}): "
        f"{resolution.adjusted_start_date}..{resolution.adjusted_end_date}"
    )

    complete_paths = []
    for number, trade_date in enumerate(trade_dates, start=1):
        print(f"[MERGE {number}/{len(trade_dates)}] {trade_date}")
        complete_path = merge_partition_caches_for_date(
            trade_date,
            weights_dir=weights_dir,
            source_tick_root=source_tick_root,
            cache_dir=cache_dir,
            delete_partition_caches=delete_partition_caches,
            overwrite_complete_cache=overwrite_complete_cache,
        )
        if complete_path is not None:
            complete_paths.append(complete_path)
    write_unavailable_cache_summary(cache_dir, trade_dates)
    return complete_paths


__all__ = [
    "CSI300_INDEX_CODE",
    "MarketCacheTask",
    "TradingDateResolution",
    "build_market_tasks",
    "discover_csi300_weight_files",
    "final_cache_path",
    "find_compatible_partition_cache",
    "generate_csi300_caches",
    "generate_trade_date_cache",
    "get_previous_closest_trading_date",
    "load_csi300_stock_codes",
    "merge_partition_caches_for_date",
    "merge_partition_caches_for_range",
    "preview_generation_plan",
    "resolve_csi300_weight_file",
    "resolve_trading_dates",
    "select_complete_trade_dates",
    "unavailable_cache_summary_path",
    "write_unavailable_cache_summary",
]
