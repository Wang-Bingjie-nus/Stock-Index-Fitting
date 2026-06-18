"""上交所 ETF 申赎清单（成份股信息）爬虫。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

SSE_QUERY_URL = "https://query.sse.com.cn/commonQuery.do"
SSE_LIST_REFERER = "https://www.sse.com.cn/disclosure/fund/etflist/"
SSE_DETAIL_REFERER = "https://www.sse.com.cn/disclosure/fund/etflist/detail.shtml?fundid={fund_id}"

SQL_ETF_LIST = "COMMON_SSE_PL_ETFGGSGSHQD_L"
SQL_ETF_BASIC = "COMMON_SSE_CP_JJLB_ETFJJGK_GGSGSHQD_JBXX_C"
SQL_ETF_COMPONENTS = "COMMON_SSE_CP_JJLB_ETFJJGK_GGSGSHQD_COMPONENT_C"

DEFAULT_STOCK_ETF_CLASSES = ("01", "03", "08", "09")

SSE_ETF_CLASS_NAME_MAP = {
    "01": "单市场股票（沪）ETF",
    "03": "跨市场股票（沪深京）ETF",
    "08": "跨市场股票（沪港深京）ETF",
    "09": "股票（科创板/含科创板）ETF",
}

# 上证50相关 ETF（上交所挂牌，关键词检索的补充）
DEFAULT_SSE50_ETF_CODES = [
    "510050",  # 华夏上证50ETF
    "510100",  # 易方达
    "510190",  # 华安
    "510600",  # 申万菱信
    "510680",  # 万家
    "510710",  # 博时
    "510800",  # 建信
    "510850",  # 工银
    "510950",  # 广发
    "512050",  # 华夏上证50ETF（若存在）
]

SUBSTITUTION_FLAG_MAP = {
    "0": "禁止",
    "1": "允许",
    "2": "必须",
    "3": "深市或京市退补",
    "4": "深市或京市必须",
    "5": "退补",
    "6": "必须",
    "7": "港市退补",
    "8": "港市必须",
}

MARKET_MAP = {
    "101": "上海证券交易所",
    "102": "深圳证券交易所",
    "103": "香港联合交易所",
    "105": "外汇交易中心",
    "106": "北京证券交易所",
    "9999": "其他",
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text.replace("\u00a0", "").replace(" ", "")


def _decode_substitution_flag(flag: Any, etf_version: str = "XML") -> str:
    key = str(flag).strip()
    if etf_version == "XML":
        return {"0": "禁止", "1": "允许", "2": "必须"}.get(key, "无")
    return SUBSTITUTION_FLAG_MAP.get(key, "无")


def _decode_market(market: Any) -> str:
    return MARKET_MAP.get(str(market).strip(), str(market))


def _format_trading_day(value: Any) -> str:
    text = re.sub(r"\D", "", str(value))
    if len(text) == 8:
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return str(value)


class SseEtfPcfScraper:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            }
        )

    def _warm_up(self) -> None:
        self.session.get(SSE_LIST_REFERER, timeout=30)

    def _query(self, params: dict[str, Any], fund_id: str | None = None) -> dict[str, Any]:
        referer = SSE_DETAIL_REFERER.format(fund_id=fund_id) if fund_id else SSE_LIST_REFERER
        headers = {"Referer": referer}
        resp = self.session.get(SSE_QUERY_URL, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("actionErrors"):
            raise RuntimeError(f"SSE 接口返回错误: {data.get('actionErrors')}")
        return data

    # def search_etf_codes(self, keywords: str = "上证50", page_size: int = 25) -> list[dict[str, str]]:
    #     """按关键字搜索 ETF 列表（股票 ETF 类别）。"""
    #     print('search_etf_codes', keywords)
    #     self._warm_up()
    #     params = {
    #         "sqlId": SQL_ETF_LIST,
    #         "isPagination": "true",
    #         "ETF_CLASS": "01",
    #         "type": "inParams",
    #         "KEY_WORDS": keywords,
    #         "FUND_CODE": "",
    #         "pageHelp.pageSize": str(page_size),
    #         "pageHelp.pageNo": "1",
    #         "pageHelp.beginPage": "1",
    #         "pageHelp.endPage": "1",
    #     }
    #     data = self._query(params)
    #     rows = data.get("result") or []
    #     out: list[dict[str, str]] = []
    #     for row in rows:
    #         code = str(row.get("FUNDID2", "")).strip()
    #         if not code:
    #             continue
    #         out.append(
    #             {
    #                 "etf_code": code,
    #                 "etf_name": str(row.get("ETF_FULLNAME", "")).strip(),
    #                 "fund_company": str(row.get("FUND_COMP_NAME", "")).strip(),
    #                 "trading_day": _format_trading_day(row.get("TRADING_DAY", "")),
    #                 "nav": _clean_text(row.get("NAV", "")),
    #             }
    #         )
    #     return out
    
    def search_etf_codes(
        self,
        keywords: str = "",
        etf_classes: tuple[str, ...] | list[str] = DEFAULT_STOCK_ETF_CLASSES,
        page_size: int = 1000,
    ) -> list[dict[str, str]]:
        """先按多个 ETF_CLASS 拉取全量列表，再在本地按关键词过滤。"""
        self._warm_up()

        keyword_text = str(keywords or "").strip().lower()
        out: list[dict[str, str]] = []
        seen_codes: set[str] = set()

        for etf_class in etf_classes:
            params = {
                "sqlId": SQL_ETF_LIST,
                "isPagination": "true",
                "ETF_CLASS": etf_class,
                "type": "inParams",
                "KEY_WORDS": "",
                "FUND_CODE": "",
                "pageHelp.pageSize": str(page_size),
                "pageHelp.pageNo": "1",
                "pageHelp.beginPage": "1",
                "pageHelp.endPage": "1",
            }
            data = self._query(params)
            rows = data.get("result") or data.get("pageHelp", {}).get("data") or []

            for row in rows:
                code = str(row.get("FUNDID2", "")).strip()
                if not code or code in seen_codes:
                    continue

                etf_name = str(row.get("ETF_FULLNAME", "")).strip()
                short_name = str(row.get("SECURITY_ABBR_A", "") or row.get("FUND_NAME", "")).strip()
                fund_company = str(row.get("FUND_COMP_NAME", "")).strip()

                searchable = " ".join([code, etf_name, short_name, fund_company]).lower()
                if keyword_text and keyword_text not in searchable:
                    continue

                seen_codes.add(code)
                out.append(
                    {
                        "etf_code": code,
                        "etf_name": etf_name,
                        "etf_short_name": short_name,
                        "fund_company": fund_company,
                        "trading_day": _format_trading_day(row.get("TRADING_DAY", "")),
                        "nav": _clean_text(row.get("NAV", "")),
                        "etf_class": str(row.get("ETF_CLASS", etf_class)).strip(),
                        "etf_class_name": SSE_ETF_CLASS_NAME_MAP.get(str(row.get("ETF_CLASS", etf_class)).strip(), ""),
                        "etf_type": str(row.get("ETF_TYPE", "")).strip(),
                    }
                )

        return sorted(out, key=lambda x: x["etf_code"])

    def fetch_basic_info(self, fund_id: str) -> dict[str, Any]:
        data = self._query(
            {
                "sqlId": SQL_ETF_BASIC,
                "FUNDID2": fund_id,
                "isPagination": "false",
            },
            fund_id=fund_id,
        )
        rows = data.get("result") or []
        if not rows:
            raise RuntimeError(f"未获取到 {fund_id} 的申购赎回基本信息")
        row = rows[0]
        return {
            "etf_code": fund_id,
            "fund_name": row.get("FUND_NAME"),
            "fund_company": row.get("FUND_COMP_NAME"),
            "trade_code": row.get("TRADE_CODE"),
            "trading_day": _format_trading_day(row.get("TRADING_DAY")),
            "pre_trading_day": _format_trading_day(row.get("PRE_TRADING_DAY")),
            "nav": _clean_text(row.get("NAV")),
            "nav_per_cu": _clean_text(row.get("NAVPERCU")),
            "pre_cash_component": _clean_text(row.get("PRE_CASH_COMPONENT")),
            "estimated_cash_component": _clean_text(row.get("ESTIMATED_CASH_COMPONENT")),
            "creation_redemption_unit": row.get("CREATION_REDEMPTION_UNIT"),
            "creation_redemption": row.get("CREATION_REDEMPTION"),
            "max_cash_ratio": row.get("MAX_CASH_RATIO"),
            "raw": row,
        }

    def fetch_components(self, fund_id: str) -> pd.DataFrame:
        data = self._query(
            {
                "sqlId": SQL_ETF_COMPONENTS,
                "FUNDID2": fund_id,
                "isPagination": "false",
            },
            fund_id=fund_id,
        )
        rows = data.get("result") or []
        if not rows:
            return pd.DataFrame(
                columns=[
                    "etf_code",
                    "证券代码",
                    "证券简称",
                    "股票数量_股",
                    "现金替代标志",
                    "申购现金替代溢价比率",
                    "赎回现金替代折价比率",
                    "替代金额_元",
                    "挂牌市场",
                ]
            )

        etf_version = str(rows[0].get("ETF_VERSION", "XML"))
        records: list[dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    "etf_code": fund_id,
                    "证券代码": row.get("INSTRUMENT_ID"),
                    "证券简称": row.get("INSTRUMENT_NAME"),
                    "股票数量_股": row.get("QUANTITY"),
                    "现金替代标志": _decode_substitution_flag(
                        row.get("SUBSTITUTION_FLAG"), etf_version
                    ),
                    "现金替代标志_原始": row.get("SUBSTITUTION_FLAG"),
                    "申购现金替代溢价比率": row.get("CREATION_PREMIUM_RATE"),
                    "赎回现金替代折价比率": row.get("REDEMPTION_DISCOUNT_RATE"),
                    "替代金额_元": row.get("SUBSTITUTION_CASH_AMOUNT"),
                    "挂牌市场": _decode_market(row.get("UNDERLYION_SECURITY_ID")),
                    "挂牌市场_代码": row.get("UNDERLYION_SECURITY_ID"),
                }
            )
        return pd.DataFrame(records)

    def scrape_etfs(
        self,
        etf_codes: list[str],
        etf_names: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
        """返回 (成份股表, 基本信息表, 原始 JSON 列表)。"""
        self._warm_up()
        etf_names = etf_names or {}
        component_parts: list[pd.DataFrame] = []
        basic_rows: list[dict[str, Any]] = []
        raw_payloads: list[dict[str, Any]] = []

        for code in etf_codes:
            basic = self.fetch_basic_info(code)
            basic["etf_name"] = etf_names.get(code, basic.get("fund_name"))
            df_part = self.fetch_components(code)
            if not df_part.empty:
                df_part.insert(1, "etf_name", basic.get("etf_name"))
                df_part.insert(2, "公告日期", basic.get("trading_day"))
            component_parts.append(df_part)
            basic_rows.append({k: v for k, v in basic.items() if k != "raw"})
            raw_payloads.append(
                {
                    code: {
                        "基本信息": {k: v for k, v in basic.items() if k != "raw"},
                        "成份股信息": df_part.to_dict(orient="records"),
                    }
                }
            )

        df_components = pd.concat(component_parts, ignore_index=True) if component_parts else pd.DataFrame()
        df_basic = pd.DataFrame(basic_rows)
        return df_components, df_basic, raw_payloads


# def resolve_sse50_etf_codes(scraper: SseEtfPcfScraper, keywords: str = "上证50") -> tuple[list[str], dict[str, str]]:
#     print('resolve_sse50_etf_codes', keywords)
#     found = scraper.search_etf_codes(keywords=keywords)
#     name_map = {item["etf_code"]: item["etf_name"] for item in found}
#     codes = sorted(set(name_map) | set(DEFAULT_SSE50_ETF_CODES))
#     return codes, name_map

def resolve_sse50_etf_codes(
    scraper: SseEtfPcfScraper,
    keywords: str = "上证50",
    etf_classes: tuple[str, ...] | list[str] = DEFAULT_STOCK_ETF_CLASSES,
) -> tuple[list[str], dict[str, str]]:
    found = scraper.search_etf_codes(keywords=keywords, etf_classes=etf_classes)
    name_map = {item["etf_code"]: item["etf_name"] for item in found}

    # 仅在上证50场景保留手工兜底；其他指数不要混入上证50 ETF。
    if str(keywords).strip() == "上证50":
        codes = sorted(set(name_map) | set(DEFAULT_SSE50_ETF_CODES))
    else:
        codes = sorted(name_map)

    return codes, name_map


def save_outputs(
    df_components: pd.DataFrame,
    df_basic: pd.DataFrame,
    raw_payloads: list[dict[str, Any]],
    output_dir: Path,
    index_code: str,
    trade_date: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"申赎清单_pcf_sse_{index_code}_{trade_date}"
    paths = {
        "csv": output_dir / f"{tag}.csv",
        "txt": output_dir / f"{tag}.txt",
        "basic_csv": output_dir / f"{tag}_基本信息.csv",
        "json": output_dir / f"{tag}.json",
    }

    df_components.to_csv(paths["csv"], index=False, encoding="utf-8-sig")
    df_basic.to_csv(paths["basic_csv"], index=False, encoding="utf-8-sig")

    merged_json: dict[str, Any] = {}
    for item in raw_payloads:
        merged_json.update(item)
    paths["json"].write_text(json.dumps(merged_json, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"数据来源: 上交所 query.sse.com.cn ({SQL_ETF_COMPONENTS})",
        f"指数代码: {index_code}",
        f"公告日期: {trade_date}",
        f"ETF 数量: {len(df_basic)}",
        f"成份股记录数: {len(df_components)}",
        "",
        "=== ETF 基本信息 ===",
        df_basic.to_string(index=False) if not df_basic.empty else "(空)",
        "",
        "=== 成份股样例（前 15 行）===",
        df_components.head(15).to_string(index=False) if not df_components.empty else "(空)",
    ]
    paths["txt"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def print_preview(df_components: pd.DataFrame, df_basic: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("上交所 ETF 申赎清单 — 成份股信息 预览")
    print("=" * 60)
    if not df_basic.empty:
        print("\n[基本信息]")
        print(df_basic[["etf_code", "etf_name", "trading_day", "creation_redemption_unit", "nav"]].to_string(index=False))
    if not df_components.empty:
        preview_cols = [
            "etf_code",
            "证券代码",
            "证券简称",
            "股票数量_股",
            "现金替代标志",
            "申购现金替代溢价比率",
            "赎回现金替代折价比率",
            "挂牌市场",
        ]
        cols = [c for c in preview_cols if c in df_components.columns]
        print(f"\n[成份股] 共 {len(df_components)} 条，展示前 10 条：")
        print(df_components[cols].head(10).to_string(index=False))
    print("=" * 60 + "\n")
