"""CSI index constituent/weight downloader.

This version does not guess the closeweight URL.  It follows the current
CSI website flow:

1. Visit the index detail page to initialize website context/cookies.
2. Query the same material API used by the detail page.
3. Read "样本权重" and "样本列表" filePath values from the API response.
4. Download the selected Excel file.

Example:
    download_csi_constituent("000300", "data/index", "closeweight")
    download_csi_constituent("000300", "data/index", "cons")
"""

from __future__ import annotations

import json
import os
import re
import ssl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import HTTPCookieProcessor, Request, build_opener
from http.cookiejar import CookieJar


CSINDEX_HOME = "https://www.csindex.com.cn"
CSINDEX_OSS_HOME = "https://oss-ch.csindex.com.cn"
INDEX_DETAIL_URL = CSINDEX_HOME + "/#/indices/family/detail?indexCode={index_code}"
INDEX_MATERIAL_API = CSINDEX_HOME + "/csindex-home/indexInfo/index-details-data"

DOWNLOAD_TYPES = {
    "cons": "样本列表",
    "closeweight": "样本权重",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}


@dataclass(frozen=True)
class CsiFileInfo:
    index_code: str
    download_type: str
    material_name: str
    file_name: str
    file_type: str
    file_url: str


def _validate_index_code(index_code: str) -> str:
    code = str(index_code).strip()
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError(f"index_code must be a six-digit string, got: {index_code!r}")
    return code


def _make_opener():
    context = ssl.create_default_context()
    return build_opener(HTTPCookieProcessor(CookieJar()))


def _request_bytes(opener, url: str, *, referer: str | None = None, timeout: int = 30) -> tuple[bytes, dict[str, str], str]:
    headers = dict(DEFAULT_HEADERS)
    if referer:
        headers["Referer"] = referer
    request = Request(url, headers=headers)
    with opener.open(request, timeout=timeout) as response:
        body = response.read()
        headers_out = {k.lower(): v for k, v in response.headers.items()}
        final_url = response.geturl()
    return body, headers_out, final_url


def _request_json(opener, url: str, *, referer: str | None = None, timeout: int = 30) -> dict[str, Any]:
    body, headers, _ = _request_bytes(opener, url, referer=referer, timeout=timeout)
    charset = "utf-8"
    content_type = headers.get("content-type", "")
    match = re.search(r"charset=([^;]+)", content_type, flags=re.I)
    if match:
        charset = match.group(1).strip()
    text = body.decode(charset, errors="replace")
    return json.loads(text)


def _normalize_file_url(file_path: str) -> str:
    path = str(file_path).strip()
    if not path:
        raise ValueError("empty CSI filePath")
    if path.startswith("//"):
        return "https:" + path
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return urljoin(CSINDEX_HOME, path)


def _choose_material_file(materials: dict[str, Any], index_code: str, download_type: str) -> CsiFileInfo:
    material_name = DOWNLOAD_TYPES[download_type]
    items = materials.get(material_name)
    if not items:
        available = ", ".join(k for k, v in materials.items() if v)
        raise RuntimeError(
            f"CSI material API returned no {material_name} for {index_code}. "
            f"Available materials: {available or '<none>'}"
        )

    if not isinstance(items, list):
        raise RuntimeError(f"Unexpected CSI material format for {material_name}: {type(items).__name__}")

    candidates = []
    for item in items:
        if not isinstance(item, dict):
            continue
        file_url = _normalize_file_url(item.get("filePath", ""))
        file_type = str(item.get("fileType", "")).lower().strip()
        file_name = str(item.get("fileName", "")).strip()
        if file_type in {"xls", "xlsx"} or file_url.lower().endswith((".xls", ".xlsx")):
            candidates.append((file_name, file_type, file_url))

    if not candidates:
        raise RuntimeError(f"CSI material {material_name} has no Excel file for {index_code}: {items}")

    # Prefer files whose URL/name contains the requested index code.  This avoids
    # selecting related derivative-index files when the API returns several.
    candidates.sort(
        key=lambda x: (
            index_code not in x[0] and index_code not in x[2],
            x[0],
            x[2],
        )
    )
    file_name, file_type, file_url = candidates[0]
    return CsiFileInfo(
        index_code=index_code,
        download_type=download_type,
        material_name=material_name,
        file_name=file_name,
        file_type=file_type or Path(file_url).suffix.lstrip(".").lower(),
        file_url=file_url,
    )


def resolve_csi_file_links(index_code: str, *, file_lang: int = 2, timeout: int = 30) -> dict[str, CsiFileInfo]:
    """Resolve current CSI constituent and closeweight Excel links.

    Args:
        index_code: six-digit CSI/SSE index code, e.g. "000300".
        file_lang: CSI website language flag. 2 means Chinese.
        timeout: HTTP timeout in seconds.

    Returns:
        Dict with keys "cons" and/or "closeweight".
    """
    code = _validate_index_code(index_code)
    opener = _make_opener()
    detail_url = INDEX_DETAIL_URL.format(index_code=code)

    # The page itself is a Vue app and does not contain the links, but visiting
    # it first mirrors the browser flow and initializes cookies if CSI adds them.
    _request_bytes(opener, detail_url, timeout=timeout)

    api_url = INDEX_MATERIAL_API + "?" + urlencode({"fileLang": file_lang, "indexCode": code})
    payload = _request_json(opener, api_url, referer=detail_url, timeout=timeout)
    if str(payload.get("code")) != "200":
        raise RuntimeError(f"CSI material API failed for {code}: {payload}")

    materials = payload.get("data") or {}
    if not isinstance(materials, dict):
        raise RuntimeError(f"CSI material API returned unexpected data for {code}: {type(materials).__name__}")

    links: dict[str, CsiFileInfo] = {}
    for download_type in DOWNLOAD_TYPES:
        links[download_type] = _choose_material_file(materials, code, download_type)
    return links


def _is_excel_response(content: bytes, content_type: str) -> bool:
    content_type = content_type.lower()
    if "application/vnd.ms-excel" in content_type:
        return True
    if "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in content_type:
        return True
    if content.startswith(b"\xd0\xcf\x11\xe0"):
        return True
    if content.startswith(b"PK\x03\x04"):
        return True
    if b"<?xml" in content[:200] and b"<Workbook" in content[:1000]:
        return True
    return False


def _output_filename(info: CsiFileInfo) -> str:
    today = datetime.now().strftime("%Y%m%d")
    suffix = ".xlsx" if info.file_url.lower().endswith(".xlsx") or info.file_type == "xlsx" else ".xls"
    label = "样本权重" if info.download_type == "closeweight" else "成分股"
    return f"{info.index_code}_{label}_{today}{suffix}"


def download_csi_constituent_v02(
    index_code: str,
    save_path: str | os.PathLike[str] | None = None,
    download_type: str = "cons",
    *,
    timeout: int = 30,
) -> str | None:
    """Download CSI constituent list or sample weights Excel.

    Args:
        index_code: six-digit index code, e.g. "000300".
        save_path: output directory. Defaults to "./csi_constituent".
        download_type: "cons" for sample list, "closeweight" for sample weights.
        timeout: HTTP timeout in seconds.

    Returns:
        Saved file path on success, otherwise None.
    """
    code = _validate_index_code(index_code)
    if download_type not in DOWNLOAD_TYPES:
        raise ValueError(f"download_type must be one of {sorted(DOWNLOAD_TYPES)}, got: {download_type!r}")

    output_dir = Path(save_path or "csi_constituent")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        links = resolve_csi_file_links(code, timeout=timeout)
        info = links[download_type]
        output_path = output_dir / _output_filename(info)

        print(f"正在下载指数 {code} {info.material_name}...")
        print(f"解析到真实链接: {info.file_url}")

        opener = _make_opener()
        content, headers, final_url = _request_bytes(
            opener,
            info.file_url,
            referer=INDEX_DETAIL_URL.format(index_code=code),
            timeout=timeout,
        )
        content_type = headers.get("content-type", "")
        if not _is_excel_response(content, content_type):
            debug_path = output_path.with_suffix(".html")
            debug_path.write_bytes(content)
            print(f"下载失败: 返回内容不是 Excel，已保存调试文件: {debug_path}")
            print(f"响应 Content-Type: {content_type}")
            print(f"响应 URL: {final_url}")
            return None

        # Adjust suffix if the server returns an xlsx zip while the URL says xls.
        if content.startswith(b"PK\x03\x04") and output_path.suffix.lower() != ".xlsx":
            output_path = output_path.with_suffix(".xlsx")

        output_path.write_bytes(content)
        print(f"下载完成: {output_path}")
        return str(output_path)

    except (HTTPError, URLError, TimeoutError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"下载失败: {exc}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download CSI index constituent/weight files.")
    parser.add_argument("index_code", help="six-digit index code, e.g. 000300")
    parser.add_argument("--save-path", default="csi_constituent", help="output directory")
    parser.add_argument(
        "--type",
        dest="download_type",
        choices=sorted(DOWNLOAD_TYPES),
        default="closeweight",
        help="file type to download",
    )
    args = parser.parse_args()
    download_csi_constituent_v02(args.index_code, args.save_path, args.download_type)
