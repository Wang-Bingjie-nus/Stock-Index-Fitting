"""同花顺 HTTP 指数数据导入辅助函数（供 Notebook 调用）。"""

from __future__ import annotations

import json
import os
import pickle
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://quantapi.51ifind.com/api/v1"

# ========== 指数配置：改 INDEX_CODE 即可切换 ==========
INDEX_CODE = "000016"  # 上证50
# INDEX_CODE = "000300"  # 沪深300
# INDEX_CODE = "000905"  # 中证500
# INDEX_CODE = "000852"  # 中证1000
# INDEX_CODE = "000688"  # 科创50

INDEX_META = {
    "000016": {"name": "上证50", "ths": "000016.SH", "futures": "IH00.CFE", "primary_etf": "510050.SH"},
    "000300": {"name": "沪深300", "ths": "000300.SH", "futures": "IF00.CFE", "primary_etf": "510300.SH"},
    "000905": {"name": "中证500", "ths": "000905.SH", "futures": "IC00.CFE", "primary_etf": "510500.SH"},
    "000852": {"name": "中证1000", "ths": "000852.SH", "futures": "IM00.CFE", "primary_etf": "512100.SH"},
    "000688": {"name": "科创50", "ths": "000688.SH", "futures": "IC00.CFE", "primary_etf": "588000.SH"},
}

EX_DIVIDEND_COLUMN_MAP = {
    "thscode": "股票代码",
    "time": "报告期_财报期末",
    "ths_ex_dividend_date_stock": "除权除息日",
    "ths_dividend_ps_before_tax_stock": "每股派息_税前",
    "ths_bonus_shares_ps_stock": "每股送股",
}

LIMIT_STATUS_COLUMN_MAP = {
    "thscode": "股票代码",
    "time": "状态日期",
    "ths_up_and_down_status_stock": "涨跌停状态",
    "is_limit_up": "前日是否涨停",
    "is_limit_down": "前日是否跌停",
    "is_suspended": "前日是否停牌",
    "can_trade_today_hint": "今日交易参考",
}


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def get_access_token(refresh_token: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/get_access_token",
        headers={"Content-Type": "application/json", "refresh_token": refresh_token},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errorcode") not in (0, "0", None):
        raise RuntimeError(f"获取 access_token 失败: {data.get('errmsg') or data}")
    token = data.get("data", {}).get("access_token")
    if not token:
        raise RuntimeError(f"响应中无 access_token: {data}")
    return token


def api_post(path: str, token: str, body: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    resp = requests.post(
        f"{BASE_URL}/{path}",
        json=body,
        headers={"Content-Type": "application/json", "access_token": token},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errorcode") not in (0, "0", None):
        raise RuntimeError(f"{path} 失败 [{data.get('errorcode')}]: {data.get('errmsg')}")
    return data


def resolve_trade_date(token: str, today: datetime | None = None) -> tuple[str, str]:
    """返回 (YYYYMMDD, YYYY-MM-DD)。优先今天；若非交易日则取最近一个交易日。"""
    today = today or datetime.now()
    ymd = today.strftime("%Y%m%d")
    dash = today.strftime("%Y-%m-%d")
    try:
        data = api_post(
            "get_trade_dates",
            token,
            {
                "marketcode": "212001",
                "functionpara": {
                    "dateType": "0",
                    "period": "D",
                    "offset": "-10",
                    "dateFormat": "0",
                    "output": "sequencedate",
                },
                "startdate": dash,
            },
            timeout=30,
        )
        times = (data.get("tables") or {}).get("time") or []
        eligible = [t for t in times if t.replace("-", "") <= ymd]
        if eligible:
            pick = eligible[-1]
            return pick.replace("-", ""), pick
    except Exception:
        pass
    return ymd, dash


def get_prior_trade_date(token: str, trade_date: str, trade_date_dash: str) -> tuple[str, str]:
    """返回当前交易日的前一个交易日 (YYYYMMDD, YYYY-MM-DD)。"""
    data = api_post(
        "get_trade_dates",
        token,
        {
            "marketcode": "212001",
            "functionpara": {
                "dateType": "0",
                "period": "D",
                "offset": "-10",
                "dateFormat": "0",
                "output": "sequencedate",
            },
            "startdate": trade_date_dash,
        },
        timeout=30,
    )
    times = (data.get("tables") or {}).get("time") or []
    prior = [t for t in times if t.replace("-", "") < trade_date]
    if not prior:
        raise RuntimeError(f"未找到 {trade_date} 之前的交易日")
    pick = prior[-1]
    return pick.replace("-", ""), pick


def load_constituents_from_csi(csi_dir: Path, index_code: str) -> pd.DataFrame:
    files = sorted(csi_dir.glob(f"{index_code}_样本权重_*.xls"))
    if not files:
        raise FileNotFoundError(
            f"未找到 {index_code} 权重文件，请先在 CSI 目录运行下载："
            f"python test_csiweb.py 或 download_csi_constituent('{index_code}')"
        )
    import sys

    project_root = csi_dir.parent.parent
    if str(project_root / "CSI") not in sys.path:
        sys.path.insert(0, str(project_root / "CSI"))
    from csiweb import read_csi_file

    df = read_csi_file(str(files[-1]))
    if df is None or df.empty:
        raise RuntimeError(f"读取权重文件失败: {files[-1]}")

    def to_thscode(row: pd.Series) -> str:
        code = row["成份券代码Constituent Code"]
        ex = str(row.get("交易所Exchange", ""))
        if "上海" in ex or "Shanghai" in ex:
            return f"{code}.SH"
        if "深圳" in ex or "Shenzhen" in ex:
            return f"{code}.SZ"
        return f"{code}.SH"

    out = df.copy()
    out["thscode"] = out.apply(to_thscode, axis=1)
    out["data_date"] = files[-1].stem.split("_")[-1]
    return out


def tables_to_long_df(tables: list[dict], value_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for item in tables:
        thscode = item.get("thscode")
        times = item.get("time") or [None]
        table = item.get("table") or {}
        n = max(len(times), max((len(table.get(c, [])) for c in value_cols), default=0))
        for i in range(n):
            row = {"thscode": thscode, "time": times[i] if i < len(times) else None}
            for c in value_cols:
                vals = table.get(c, [])
                row[c] = vals[i] if i < len(vals) else None
            rows.append(row)
    return pd.DataFrame(rows)


def fetch_etf_list(token: str, index_name: str, trade_date: str) -> pd.DataFrame:
    """跟踪该指数的 ETF/联接基金产品列表（不是申赎清单 PCF）。"""
    data = api_post(
        "smart_stock_picking",
        token,
        {"searchstring": f"ETF,跟踪指数,{index_name}", "searchtype": "fund"},
        timeout=120,
    )
    tables = data.get("tables") or []
    if not tables or not tables[0].get("table"):
        return pd.DataFrame(columns=["fund_code", "fund_name", "tracking_index", "query_date"])
    raw = tables[0]["table"]
    keys = list(raw.keys())
    df = pd.DataFrame(
        {
            "fund_code": raw.get(keys[0], []),
            "fund_name": raw.get(keys[1], [] if len(keys) < 2 else raw.get(keys[1], [])),
            "tracking_index": raw.get(keys[2], []) if len(keys) > 2 else None,
        }
    )
    df["query_date"] = trade_date
    df["is_listed_etf"] = df["fund_code"].str.endswith((".SH", ".SZ"))
    df["fund_type_tags"] = df["tracking_index"]
    return df


def filter_listed_etfs(df_products: pd.DataFrame) -> list[str]:
    if df_products.empty:
        return []
    codes = df_products.loc[df_products["is_listed_etf"], "fund_code"].dropna().tolist()
    return sorted(set(codes))


def fetch_ex_dividend(
    token: str,
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    chunk_size: int = 8,
) -> pd.DataFrame:
    """分批拉取除权除息，避免 date_sequence 单次超过账号 1W 条限制。"""
    indicators = [
        "ths_ex_dividend_date_stock",
        "ths_dividend_ps_before_tax_stock",
        "ths_bonus_shares_ps_stock",
    ]
    parts: list[pd.DataFrame] = []
    for i in range(0, len(stock_codes), chunk_size):
        batch = stock_codes[i : i + chunk_size]
        data = api_post(
            "date_sequence",
            token,
            {
                "codes": ",".join(batch),
                "startdate": start_date,
                "enddate": end_date,
                "functionpara": {"Fill": "Blank"},
                "indipara": [
                    {"indicator": ind, "indiparams": ["", "100", ""]} for ind in indicators
                ],
            },
            timeout=300,
        )
        part = tables_to_long_df(data.get("tables") or [], indicators)
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(how="all", subset=indicators)
    return format_ex_dividend_df(df)


def format_ex_dividend_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.rename(columns=EX_DIVIDEND_COLUMN_MAP, inplace=True)
    if "除权除息日" in out.columns:
        out["除权除息日"] = out["除权除息日"].apply(_normalize_ifind_date)
    return out


def _normalize_ifind_date(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.isdigit() and len(text) == 8:
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


def fetch_limit_up_down_status(token: str, stock_codes: list[str], status_date: str) -> pd.DataFrame:
    """拉取指定交易日的涨跌停/停牌状态（不是涨跌停价格）。"""
    indicator = "ths_up_and_down_status_stock"
    data = api_post(
        "date_sequence",
        token,
        {
            "codes": ",".join(stock_codes),
            "startdate": status_date,
            "enddate": status_date,
            "functionpara": {"Fill": "Blank"},
            "indipara": [{"indicator": indicator, "indiparams": ["", "100", ""]}],
        },
        timeout=300,
    )
    df = tables_to_long_df(data.get("tables") or [], [indicator])
    return format_limit_status_df(df, status_date)


def format_limit_status_df(df: pd.DataFrame, status_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    status_col = "ths_up_and_down_status_stock"
    out[status_col] = out[status_col].fillna("").astype(str).str.strip()
    out["is_limit_up"] = out[status_col] == "涨停"
    out["is_limit_down"] = out[status_col] == "跌停"
    out["is_suspended"] = out[status_col] == "停牌"
    out["can_trade_today_hint"] = out[status_col].map(_limit_status_to_hint)
    out["status_date"] = status_date
    out.rename(columns=LIMIT_STATUS_COLUMN_MAP, inplace=True)
    out["状态基准日_前一日"] = status_date
    return out


def _limit_status_to_hint(status: str) -> str:
    if status == "涨停":
        return "前日涨停，今日可能难以买入或波动加大"
    if status == "跌停":
        return "前日跌停，今日可能难以卖出或波动加大"
    if status == "停牌":
        return "前日停牌，今日可能继续停牌"
    if status in {"", "非涨跌停"}:
        return "前日正常交易"
    return f"前日状态：{status}"


def fetch_limit_prices(token: str, stock_codes: list[str], trade_date: str) -> pd.DataFrame:
    """当日涨跌停价格（辅助字段，不是「是否涨停」）。"""
    indicators = ["ths_max_up_stock", "ths_max_down_stock"]
    data = api_post(
        "date_sequence",
        token,
        {
            "codes": ",".join(stock_codes),
            "startdate": trade_date,
            "enddate": trade_date,
            "functionpara": {"Fill": "Blank"},
            "indipara": [
                {"indicator": ind, "indiparams": ["", "100", ""]} for ind in indicators
            ],
        },
        timeout=300,
    )
    df = tables_to_long_df(data.get("tables") or [], indicators)
    df.rename(
        columns={
            "ths_max_up_stock": "涨停价",
            "ths_max_down_stock": "跌停价",
        },
        inplace=True,
    )
    df.rename(columns={"thscode": "股票代码", "time": "日期"}, inplace=True)
    df["trade_date"] = trade_date
    return df


def _local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _find_text(root: ET.Element, names: set[str]) -> str | None:
    for elem in root.iter():
        if _local_name(elem.tag) in names and elem.text:
            text = elem.text.strip()
            if text:
                return text
    return None


def _find_all_components(root: ET.Element) -> list[ET.Element]:
    components: list[ET.Element] = []
    for elem in root.iter():
        name = _local_name(elem.tag)
        if name in {"Component", "Record", "Constituent"}:
            components.append(elem)
    return components


def _child_text(node: ET.Element, names: set[str]) -> str | None:
    for child in list(node):
        if _local_name(child.tag) in names and child.text:
            text = child.text.strip()
            if text:
                return text
    return None


def parse_sse_pcf_xml(xml_text: str, etf_code: str, trade_date: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """解析上交所 ssepcf XML，返回 (图3风格摘要 dict, 成份股 long DataFrame)。"""
    root = ET.fromstring(xml_text)
    pure_code = etf_code.split(".")[0]

    creation_unit = _find_text(root, {"CreationRedemptionUnit", "CreationUnit", "CreationRedemption"})
    nav_per_cu = _find_text(root, {"NAVperCU", "EstimateCashComponent", "PreCashComponent"})
    nav = _find_text(root, {"NAV", "NetAssetValue"})
    cash_component = _find_text(root, {"CashComponent", "PreCashComponent", "EstimateCashComponent"})
    fund_name = _find_text(root, {"FundName", "FundInstrumentName"})

    components = _find_all_components(root)
    rows: list[dict[str, Any]] = []
    for comp in components:
        code = _child_text(comp, {"InstrumentID", "UnderlyingSecurityID", "SecurityID", "ComponentCode"})
        if not code:
            continue
        qty = _child_text(comp, {"Quantity", "ComponentShare", "ShareAmount"})
        name = _child_text(comp, {"InstrumentName", "UnderlyingSecurityName", "SecurityName"})
        sub_flag = _child_text(comp, {"SubstitutionFlag", "CashSubstitutionFlag"})
        market = _child_text(comp, {"Market", "Exchange"})
        if market in {"1", "SH"}:
            thscode = f"{code}.SH"
        elif market in {"2", "SZ"}:
            thscode = f"{code}.SZ"
        elif code.startswith("6"):
            thscode = f"{code}.SH"
        else:
            thscode = f"{code}.SZ"
        rows.append(
            {
                "etf_code": etf_code,
                "trade_date": trade_date,
                "constituent_code": thscode,
                "constituent_name": name,
                "constituent_qty": int(float(qty)) if qty else None,
                "cash_substitution_flag": sub_flag,
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        etf_code: {
            "market": etf_code.split(".")[-1],
            "stock": pure_code,
            "基金名称": fund_name,
            "最小申购、赎回单位": int(float(creation_unit)) if creation_unit else None,
            "最小申购、赎回单位净值": float(nav_per_cu) if nav_per_cu else None,
            "基金份额净值": float(nav) if nav else None,
            "现金差额": float(cash_component) if cash_component else None,
            "成份股信息": [
                {
                    "成份股代码": r["constituent_code"],
                    "成份股名称": r["constituent_name"],
                    "成份股数量": r["constituent_qty"],
                    "现金替代标志": r["cash_substitution_flag"],
                }
                for r in rows
            ],
        }
    }
    return summary, df


def download_sse_pcf_xml(fund_code: str, trade_date: str) -> str | None:
    """尝试从上交所公开渠道下载 ssepcf XML。失败时返回 None。"""
    pure = fund_code.split(".")[0]
    ymd = trade_date.replace("-", "")
    dash = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.sse.com.cn/disclosure/fund/etflist/",
    }
    session.get("https://www.sse.com.cn/disclosure/fund/etflist/", headers=headers, timeout=30)

    candidates: list[tuple[str, dict[str, str] | None]] = [
        (f"https://www.sse.com.cn/disclosure/fund/etflist/ssepcf{pure}_{ymd}.xml", None),
        (f"https://www.sse.com.cn/disclosure/fund/etflist/ssepcf{pure}_{ymd}.txt", None),
        (
            "https://query.sse.com.cn/commonQuery.do",
            {
                "sqlId": "COMMON_SSE_CP_FUND_ETFPCF_L",
                "fundCode": pure,
                "queryDate": dash,
                "isPagination": "true",
                "pageHelp.pageSize": "25",
                "pageHelp.pageNo": "1",
            },
        ),
    ]
    for url, params in candidates:
        try:
            resp = session.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code != 200:
                continue
            text = resp.text.strip()
            if text.startswith("<?xml") or text.startswith("<"):
                return text
            if text.startswith("{"):
                payload = resp.json()
                result = payload.get("result")
                if isinstance(result, list) and result:
                    file_url = result[0].get("fileUrl") or result[0].get("FILE_URL")
                    if file_url:
                        file_resp = session.get(file_url, headers=headers, timeout=30)
                        if file_resp.status_code == 200 and file_resp.text.strip():
                            return file_resp.text
        except Exception:
            continue
    return None


def fetch_etf_pcf_from_xtquant(etf_codes: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    """若本机安装了迅投 xtquant，可读取图3风格的 ETF 申赎清单。"""
    from xtquant import xtdata

    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for code in etf_codes:
        info = xtdata.get_etf_info(code)
        if not info:
            continue
        summaries[code] = info
        for item in info.get("成份股信息", []):
            rows.append(
                {
                    "etf_code": code,
                    "trade_date": info.get("trade_date"),
                    "constituent_code": item.get("成份股代码"),
                    "constituent_name": item.get("成份股名称"),
                    "constituent_qty": item.get("成份股数量"),
                    "cash_substitution_flag": item.get("现金替代标志"),
                    "creation_unit": info.get("最小申购、赎回单位"),
                    "cash_component": info.get("现金差额"),
                }
            )
    return summaries, pd.DataFrame(rows)


def fetch_etf_pcf(
    etf_codes: list[str],
    trade_date: str,
    local_pcf_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    """
    获取 ETF 申赎清单（PCF，图3格式）。

    优先级：本地 XML 文件 > 上交所下载 > xtquant。
    返回 (summary_dict, long_df, warnings)。
    """
    summaries: dict[str, Any] = {}
    parts: list[pd.DataFrame] = []
    warnings: list[str] = []

    for code in etf_codes:
        xml_text: str | None = None
        if local_pcf_dir:
            pure = code.split(".")[0]
            for pattern in [f"ssepcf{pure}_{trade_date}.xml", f"ssepcf{pure}_{trade_date}.txt", f"{pure}_{trade_date}.xml"]:
                candidate = local_pcf_dir / pattern
                if candidate.is_file():
                    xml_text = candidate.read_text(encoding="utf-8", errors="ignore")
                    break
        if xml_text is None and code.endswith(".SH"):
            xml_text = download_sse_pcf_xml(code, trade_date)
        if xml_text:
            summary, df = parse_sse_pcf_xml(xml_text, code, trade_date)
            summaries.update(summary)
            parts.append(df)
            continue
        warnings.append(f"{code}: 未能从公开渠道获取 PCF（上交所 API/静态文件均无数据）")

    if not summaries:
        try:
            xt_summary, xt_df = fetch_etf_pcf_from_xtquant(etf_codes)
            if xt_summary:
                summaries.update(xt_summary)
                parts.append(xt_df)
                return summaries, pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), warnings
        except ImportError:
            warnings.append("本机未安装 xtquant；图3格式可请带教提供迅投 get_etf_info 环境，或手动放入 ssepcf XML。")
        except Exception as exc:
            warnings.append(f"xtquant get_etf_info 失败: {exc}")

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=[
            "etf_code",
            "trade_date",
            "constituent_code",
            "constituent_name",
            "constituent_qty",
            "cash_substitution_flag",
        ]
    )
    return summaries, df, warnings


def snap_to_df(data: dict[str, Any]) -> pd.DataFrame:
    tables = data.get("tables") or []
    if not tables:
        return pd.DataFrame()
    item = tables[0]
    times = item.get("time") or []
    table = item.get("table") or {}
    df = pd.DataFrame({"time": times})
    for k, v in table.items():
        df[k] = v
    df.insert(0, "thscode", item.get("thscode"))
    return df


def fetch_snap_shot(
    token: str, code: str, trade_date_dash: str, indicators: str
) -> pd.DataFrame:
    body = {
        "codes": code,
        "indicators": indicators,
        "starttime": f"{trade_date_dash} 09:30:00",
        "endtime": f"{trade_date_dash} 15:00:00",
    }
    data = api_post("snap_shot", token, body, timeout=300)
    return snap_to_df(data)


def fetch_snap_shot_with_fallback(
    token: str,
    code: str,
    trade_date: str,
    trade_date_dash: str,
    indicators: str,
) -> tuple[pd.DataFrame, str, str]:
    """若当日无 tick（非交易日或尚未开盘），自动回退到最近一个有数据的交易日。"""
    df = fetch_snap_shot(token, code, trade_date_dash, indicators)
    if not df.empty:
        return df, trade_date, trade_date_dash

    data = api_post(
        "get_trade_dates",
        token,
        {
            "marketcode": "212001",
            "functionpara": {
                "dateType": "0",
                "period": "D",
                "offset": "-10",
                "dateFormat": "0",
                "output": "sequencedate",
            },
            "startdate": trade_date_dash,
        },
        timeout=30,
    )
    times = (data.get("tables") or {}).get("time") or []
    prior = [t for t in times if t.replace("-", "") < trade_date]
    for pick in reversed(prior):
        pick_ymd = pick.replace("-", "")
        df = fetch_snap_shot(token, code, pick, indicators)
        if not df.empty:
            print(f"  当日无 tick，已回退到 {pick}")
            return df, pick_ymd, pick
    return df, trade_date, trade_date_dash


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def save_pkl(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
