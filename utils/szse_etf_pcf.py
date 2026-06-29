"""深交所 ETF 申赎清单（PCF）爬虫。

一个函数：拼 URL → 下载 GBK 文本 → 解析固定宽度列 → 返回 DataFrame。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd
import requests

SZSE_PCF_URL = "https://reportdocs.static.szse.cn/files/text/etf/ETF{code}{date}.txt"


def fetch_szse_pcf(etf_code: str, trade_date: str) -> pd.DataFrame:
    """下载并解析深交所 ETF 申赎清单。

    Args:
        etf_code: ETF 代码（纯数字，如 '159919'）
        trade_date: 交易日（YYYYMMDD 格式）

    Returns:
        DataFrame，列：etf_code, stock_code, stock_name,
        component_qty, substitution_flag, source, retrieved_at
    """
    url = SZSE_PCF_URL.format(code=etf_code, date=trade_date)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # GBK → UTF-8
    text = resp.content.decode("gbk", errors="replace")
    lines = text.splitlines()

    # 找到 "组合信息内容" 部分
    header_idx = None
    for i, line in enumerate(lines):
        if "组合信息内容" in line:
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError(f"未找到组合信息内容部分: {url}")

    records = []
    for line in lines[header_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        # 跳过表头和分隔线
        if "证券代码" in stripped or stripped.startswith("-"):
            continue

        # 按 2 个以上连续空格切分（深交所 PCF 为固定宽度格式）
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 4:
            continue

        code = parts[0]
        # 只处理 6 位数字股票代码
        if not re.match(r"^\d{6}$", code):
            continue
        # 过滤虚拟成份证券 159900（跨市场 ETF 的申赎现金汇总行）
        if code == "159900":
            continue

        name = parts[1]
        qty = int(parts[2].replace(",", ""))
        flag = parts[3]

        records.append({
            "stock_code": code,
            "stock_name": name.strip(),
            "component_qty": qty,
            "substitution_flag": flag,
        })

    if not records:
        raise RuntimeError(f"未解析到任何成份股: {url}")

    df = pd.DataFrame(records)
    df["etf_code"] = etf_code
    df["source"] = "SZSE"
    df["retrieved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return df[[
        "etf_code", "stock_code", "stock_name",
        "component_qty", "substitution_flag", "source", "retrieved_at",
    ]]
