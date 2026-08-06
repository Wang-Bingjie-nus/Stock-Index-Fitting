from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from .file_utils import get_tick_file_path
from .minute_tick_cache_v03 import (
    build_minute_tick_cache,
    build_unavailable_minute_tick_cache,
    load_minute_tick_cache,
    load_unavailable_minute_tick_cache,
    normalize_stock_code,
    normalize_stock_codes,
    normalize_trade_date,
    save_minute_tick_cache,
    save_unavailable_minute_tick_cache,
    unavailable_minute_tick_cache_path,
)
from .reader import read_stocks_ticks


def normalize_trade_date_dash(value) -> str:
    return pd.Timestamp(normalize_trade_date(value)).strftime("%Y-%m-%d")


def _cache_hash(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def minute_cache_path(
    cache_dir: str | Path,
    index_code: str,
    trade_date: str,
    stock_codes,
) -> Path:
    """Return the cache path used by ``correlation_v03.ipynb``."""

    trade_date = normalize_trade_date(trade_date)
    tick_codes = normalize_stock_codes(stock_codes)
    if not tick_codes:
        raise ValueError("stock_codes is empty.")
    payload = {
        "kind": "basket_tick_v03",
        "index_code": str(index_code),
        "fitting_date": trade_date,
        "tick_codes": tick_codes,
    }
    return Path(cache_dir) / (
        f"basket_minute_wide_{trade_date}_{_cache_hash(payload)}.pkl"
    )


def find_missing_tick_source_files(
    trade_date: str,
    stock_codes,
    *,
    source_tick_root: str | Path | None = None,
) -> list[str]:
    """Check each distinct board-level raw tick file once."""

    trade_date_dash = normalize_trade_date_dash(trade_date)
    codes = normalize_stock_codes(stock_codes)
    if source_tick_root is None:
        source_files = {
            get_tick_file_path(trade_date_dash, stock_code)
            for stock_code in codes
        }
    else:
        source_files = {
            get_tick_file_path(trade_date_dash, stock_code, str(source_tick_root))
            for stock_code in codes
        }
    return sorted(path for path in source_files if not os.path.exists(path))


def _write_unavailable_marker(
    cache_dir: Path,
    trade_date: str,
    stock_codes,
    *,
    reason: str,
    error_type: str | None = None,
    details: dict | None = None,
) -> Path:
    marker_path = unavailable_minute_tick_cache_path(cache_dir, trade_date)
    marker = build_unavailable_minute_tick_cache(
        trade_date,
        reason=reason,
        error_type=error_type,
        details=details,
        stock_codes=stock_codes,
        generated_by="utils.minute_cache_loader_v03",
    )
    save_unavailable_minute_tick_cache(marker, marker_path)
    return marker_path


def _remove_unavailable_marker(cache_dir: Path, trade_date: str) -> None:
    marker_path = unavailable_minute_tick_cache_path(cache_dir, trade_date)
    if marker_path.exists():
        try:
            marker_path.unlink()
        except OSError as exc:
            print(
                f"[CACHE WARNING] Could not delete stale marker "
                f"{marker_path.name}: {exc}"
            )


def load_stock_minute_cache_cached(
    trade_date: str,
    stock_codes,
    *,
    index_code: str,
    cache_dir: str | Path,
    source_tick_root: str | Path | None = None,
    label: str = "stock minute cache",
    source_missing_policy: str = "raise",
) -> dict | None:
    """Load or build the complete v03 minute cache for one stock universe.

    This is the reusable form of the cache cell in ``correlation_v03.ipynb``.
    A complete cache contains all requested stocks and the 240 continuous-trading
    minute closes.  When ``source_missing_policy='skip'``, an unavailable marker
    is written and ``None`` is returned instead of publishing a partial cache.
    """

    trade_date = normalize_trade_date(trade_date)
    trade_date_dash = normalize_trade_date_dash(trade_date)
    tick_codes = normalize_stock_codes(stock_codes)
    if not tick_codes:
        raise RuntimeError(f"{label}: stock_codes is empty.")

    source_missing_policy = str(source_missing_policy).strip().lower()
    if source_missing_policy not in {"raise", "skip"}:
        raise ValueError("source_missing_policy must be 'raise' or 'skip'.")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = minute_cache_path(cache_dir, index_code, trade_date, tick_codes)
    marker_file = unavailable_minute_tick_cache_path(cache_dir, trade_date)

    if cache_file.exists():
        try:
            minute_cache = load_minute_tick_cache(
                cache_file,
                trade_date=trade_date,
                stock_codes=tick_codes,
            )
            missing_stocks = minute_cache["metadata"].get("missing_stocks", [])
            if missing_stocks:
                raise ValueError(
                    f"Complete cache contains missing stocks: {missing_stocks[:20]}"
                )
            _remove_unavailable_marker(cache_dir, trade_date)
            cache_mb = cache_file.stat().st_size / 1024**2
            print(
                f"[CACHE HIT] {label} {trade_date}: "
                f"{cache_file.name} ({cache_mb:.2f} MB)"
            )
            return minute_cache
        except Exception as exc:
            print(f"[INVALID CACHE] {label} {trade_date}: {type(exc).__name__}: {exc}")

    if marker_file.exists():
        try:
            marker = load_unavailable_minute_tick_cache(
                marker_file,
                trade_date=trade_date,
            )
            metadata = marker["metadata"]
            print(
                f"[UNAVAILABLE CACHE HIT] {label} {trade_date}: "
                f"reason={metadata['reason']}; details={metadata.get('details', {})}"
            )
            if source_missing_policy == "skip":
                return None
        except Exception as exc:
            print(
                f"[INVALID UNAVAILABLE MARKER] {trade_date}: "
                f"{type(exc).__name__}: {exc}"
            )

    missing_source_files = find_missing_tick_source_files(
        trade_date,
        tick_codes,
        source_tick_root=source_tick_root,
    )
    if missing_source_files:
        message = (
            f"{label} {trade_date} requires {len(missing_source_files)} "
            f"missing source file(s): {missing_source_files}"
        )
        marker_path = _write_unavailable_marker(
            cache_dir,
            trade_date,
            tick_codes,
            reason="source_file_missing",
            error_type="FileNotFoundError",
            details={"source_files": missing_source_files, "message": message},
        )
        if source_missing_policy == "skip":
            print(f"[SKIP DATE] {message}; marker={marker_path.name}")
            return None
        raise FileNotFoundError(message)

    print(f"[CACHE MISS] reading raw ticks for {label} {trade_date}...")
    try:
        if source_tick_root is None:
            raw_tick_dict = read_stocks_ticks(trade_date_dash, tick_codes)
        else:
            raw_tick_dict = read_stocks_ticks(
                trade_date_dash,
                tick_codes,
                base_dir=str(source_tick_root),
            )
        minute_cache = build_minute_tick_cache(
            raw_tick_dict,
            trade_date,
            tick_codes,
        )
        del raw_tick_dict

        missing_stocks = minute_cache["metadata"].get("missing_stocks", [])
        if missing_stocks:
            message = (
                f"{label} {trade_date} contains {len(missing_stocks)} "
                f"missing or empty requested stocks: {missing_stocks[:20]}"
            )
            marker_path = _write_unavailable_marker(
                cache_dir,
                trade_date,
                tick_codes,
                reason="requested_stocks_missing_or_empty",
                error_type="ValueError",
                details={"missing_stocks": missing_stocks, "message": message},
            )
            if source_missing_policy == "skip":
                print(f"[SKIP DATE] {message}; marker={marker_path.name}")
                return None
            raise ValueError(message)

        save_minute_tick_cache(minute_cache, cache_file)
        _remove_unavailable_marker(cache_dir, trade_date)
        cache_mb = cache_file.stat().st_size / 1024**2
        print(f"{label} cached: {cache_file.name} ({cache_mb:.2f} MB)")
        return minute_cache
    except Exception as exc:
        if marker_file.exists():
            marker_path = marker_file
        else:
            marker_path = _write_unavailable_marker(
                cache_dir,
                trade_date,
                tick_codes,
                reason="complete_cache_generation_failed",
                error_type=type(exc).__name__,
                details={"message": str(exc)},
            )
        if source_missing_policy == "skip":
            print(
                f"[SKIP DATE] {label} {trade_date}: {type(exc).__name__}: "
                f"{exc}; marker={marker_path.name}"
            )
            return None
        raise


__all__ = [
    "find_missing_tick_source_files",
    "load_stock_minute_cache_cached",
    "minute_cache_path",
    "normalize_trade_date_dash",
]
