from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


CACHE_KIND = "basket_tick_v03"
CACHE_SCHEMA_VERSION = 3

# The source is narrowed to these eight fields before any DataFrame copy.
# ``time`` becomes the unique int64 index of the cached minute-wide table.
RAW_CACHE_COLUMNS = (
    "time",
    "lastPrice",
    "lastClose",
    "bidPrice1",
    "askPrice1",
    "volume",
    "amount",
    "stockStatus",
)
CACHE_VALUE_FIELDS = RAW_CACHE_COLUMNS[1:]
PRICE_FIELDS = ("lastPrice", "lastClose", "bidPrice1", "askPrice1")
MINUTE_ALIGNMENT = "last_observation_at_or_before_minute_second_00"


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


def normalize_trade_date(value) -> str:
    raw = re.sub(r"\D", "", str(value))
    if len(raw) != 8:
        raise ValueError(f"trade_date must be YYYYMMDD, got {value!r}.")
    return raw


def normalize_stock_codes(stock_codes) -> list[str]:
    return sorted({normalize_stock_code(code) for code in stock_codes if pd.notna(code)})


def build_trading_minute_index(trade_date: str) -> pd.Index:
    """Return the 242 minute-second-zero observations in the two A-share sessions."""

    trade_date = normalize_trade_date(trade_date)
    day = pd.Timestamp(trade_date)
    morning = pd.date_range(
        day.replace(hour=9, minute=30),
        day.replace(hour=11, minute=30),
        freq="min",
    )
    afternoon = pd.date_range(
        day.replace(hour=13, minute=0),
        day.replace(hour=15, minute=0),
        freq="min",
    )
    values = np.concatenate([
        morning.strftime("%Y%m%d%H%M%S").astype("int64"),
        afternoon.strftime("%Y%m%d%H%M%S").astype("int64"),
    ])
    return pd.Index(values, dtype="int64", name="time")


def _time_to_int64(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        text = pd.to_datetime(values, errors="coerce").dt.strftime("%Y%m%d%H%M%S")
    else:
        # XtQuant exports commonly use strings such as
        # ``20260717093000.000``. Taking the first 14-digit block also handles
        # integer and millisecond-suffixed representations without float loss.
        text = values.astype(str).str.extract(r"(\d{14})", expand=False)
    return pd.to_numeric(text, errors="coerce").astype("Int64")


def _empty_stock_minute_frame(minute_index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=minute_index, columns=CACHE_VALUE_FIELDS, dtype=float)


def _build_stock_minute_frame(
    raw: pd.DataFrame | None,
    minute_index: pd.Index,
    trade_date: str,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_stock_minute_frame(minute_index)

    available_columns = [col for col in RAW_CACHE_COLUMNS if col in raw.columns]
    # Deliberately select before copying: the 28 unused XtQuant fields never
    # enter the transformation or cache-building memory footprint.
    slim = raw.loc[:, available_columns].copy()
    if "time" not in slim.columns:
        slim.insert(0, "time", pd.Index(raw.index).to_numpy(copy=True))

    slim["time"] = _time_to_int64(slim["time"])
    slim = slim.dropna(subset=["time"])
    if slim.empty:
        return _empty_stock_minute_frame(minute_index)
    slim["time"] = slim["time"].astype("int64")

    for field in CACHE_VALUE_FIELDS:
        if field not in slim.columns:
            slim[field] = np.nan
        slim[field] = pd.to_numeric(slim[field], errors="coerce")

    last_close_values = slim.loc[slim["lastClose"].gt(0), "lastClose"]
    last_close_fallback = float(last_close_values.iloc[0]) if not last_close_values.empty else np.nan

    date_number = int(trade_date)
    hhmmss = slim["time"] % 1_000_000
    in_session = hhmmss.between(93_000, 113_000) | hhmmss.between(130_000, 150_000)
    same_date = (slim["time"] // 1_000_000).eq(date_number)
    slim = slim.loc[in_session & same_date, ["time", *CACHE_VALUE_FIELDS]]
    slim = slim.sort_values("time").drop_duplicates(subset="time", keep="last")
    if slim.empty:
        result = _empty_stock_minute_frame(minute_index)
        if np.isfinite(last_close_fallback):
            result["lastClose"] = last_close_fallback
        return result

    # Match the old consumers: zero/negative quotes are not observations and
    # therefore do not erase the latest positive quote.
    for field in PRICE_FIELDS:
        slim[field] = slim[field].where(slim[field].gt(0)).ffill()
    if np.isfinite(last_close_fallback):
        slim["lastClose"] = slim["lastClose"].fillna(last_close_fallback)

    slim = slim.set_index("time", drop=True)
    slim.index = slim.index.astype("int64")
    slim.index.name = "time"
    result = slim.reindex(minute_index, method="ffill")
    if np.isfinite(last_close_fallback):
        result["lastClose"] = result["lastClose"].fillna(last_close_fallback)
    return result.reindex(columns=CACHE_VALUE_FIELDS).astype(float)


def build_minute_tick_cache(
    raw_tick_dict: dict[str, pd.DataFrame | None],
    trade_date: str,
    stock_codes,
) -> dict:
    """Convert raw 3-second per-stock frames into one minute-wide cache object."""

    trade_date = normalize_trade_date(trade_date)
    stock_codes = normalize_stock_codes(stock_codes)
    if not stock_codes:
        raise ValueError("stock_codes is empty.")

    normalized_raw = {
        normalize_stock_code(code): frame
        for code, frame in (raw_tick_dict or {}).items()
    }
    minute_index = build_trading_minute_index(trade_date)
    stock_frames = {
        code: _build_stock_minute_frame(normalized_raw.get(code), minute_index, trade_date)
        for code in stock_codes
    }

    wide = pd.concat(stock_frames, axis=1, names=["stock_code", "field"])
    wide = wide.swaplevel("stock_code", "field", axis=1)
    expected_columns = pd.MultiIndex.from_product(
        [CACHE_VALUE_FIELDS, stock_codes],
        names=["field", "stock_code"],
    )
    wide = wide.reindex(index=minute_index, columns=expected_columns)
    wide.index = wide.index.astype("int64")
    wide = wide.loc[~wide.index.duplicated(keep="last")]

    missing_stocks = [
        code
        for code in stock_codes
        if normalized_raw.get(code) is None or normalized_raw[code].empty
    ]
    return {
        "metadata": {
            "kind": CACHE_KIND,
            "schema_version": CACHE_SCHEMA_VERSION,
            "trade_date": trade_date,
            "stock_codes": stock_codes,
            "source_fields": list(RAW_CACHE_COLUMNS),
            "cached_value_fields": list(CACHE_VALUE_FIELDS),
            "time_index_dtype": "int64",
            "minute_alignment": MINUTE_ALIGNMENT,
            "minute_rows": int(len(wide)),
            "missing_stocks": missing_stocks,
        },
        "data": wide,
    }


def validate_minute_tick_cache(
    cache: dict,
    *,
    trade_date: str | None = None,
    stock_codes=None,
) -> dict:
    if not isinstance(cache, dict) or "metadata" not in cache or "data" not in cache:
        raise ValueError("Invalid minute tick cache object.")
    metadata = cache["metadata"]
    wide = cache["data"]
    if metadata.get("kind") != CACHE_KIND:
        raise ValueError(f"Unexpected cache kind: {metadata.get('kind')!r}.")
    if int(metadata.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unexpected cache schema version: {metadata.get('schema_version')!r}.")
    if not isinstance(wide, pd.DataFrame) or not isinstance(wide.columns, pd.MultiIndex):
        raise ValueError("Minute tick cache data must be a DataFrame with MultiIndex columns.")
    if wide.columns.names != ["field", "stock_code"]:
        raise ValueError(f"Unexpected cache column levels: {wide.columns.names!r}.")
    if wide.index.dtype != np.dtype("int64"):
        raise ValueError(f"Minute tick cache index must be int64, got {wide.index.dtype}.")
    if wide.index.has_duplicates:
        raise ValueError("Minute tick cache index contains duplicates.")
    missing_fields = sorted(set(CACHE_VALUE_FIELDS) - set(wide.columns.get_level_values("field")))
    if missing_fields:
        raise ValueError(f"Minute tick cache is missing fields: {missing_fields}.")

    if trade_date is not None and metadata.get("trade_date") != normalize_trade_date(trade_date):
        raise ValueError(
            f"Cache trade date mismatch: {metadata.get('trade_date')} != {normalize_trade_date(trade_date)}."
        )
    if stock_codes is not None:
        expected_codes = normalize_stock_codes(stock_codes)
        if metadata.get("stock_codes") != expected_codes:
            raise ValueError("Cache stock universe does not match the requested stock_codes.")
    return cache


def get_minute_field(
    cache: dict,
    field: str,
    *,
    stock_codes=None,
    copy: bool = True,
) -> pd.DataFrame:
    validate_minute_tick_cache(cache)
    if field not in CACHE_VALUE_FIELDS:
        raise KeyError(f"Unknown minute cache field: {field!r}.")
    frame = cache["data"].xs(field, axis=1, level="field", drop_level=True)
    if stock_codes is not None:
        frame = frame.reindex(columns=normalize_stock_codes(stock_codes))
    return frame.copy() if copy else frame


def load_minute_tick_cache(
    path: str | Path,
    *,
    trade_date: str | None = None,
    stock_codes=None,
) -> dict:
    path = Path(path)
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    return validate_minute_tick_cache(cache, trade_date=trade_date, stock_codes=stock_codes)


def save_minute_tick_cache(cache: dict, path: str | Path) -> Path:
    validate_minute_tick_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path
