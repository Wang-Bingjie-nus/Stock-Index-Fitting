"""数据集目录布局与命名约定（供导入、交叉验证共用）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# 数据类别 → (一级目录名, 默认数据源子目录)
CATEGORIES: dict[str, tuple[str, str]] = {
    "index_weight": ("01_指数权重", "csi"),
    "etf_products": ("02_etf/产品清单", "ifind"),
    "etf_pcf": ("02_etf/申赎清单_pcf", "ifind"),
    "ex_dividend": ("03_除权除息", "ifind"),
    "limit_status": ("04_涨跌停状态", "ifind"),
    "index_ticks": ("05_指数ticks", "ifind"),
    "futures_ticks": ("06_股指期货ticks", "ifind"),
}

# 交叉验证时可用的第二数据源目录名
ALT_SOURCES = ("xtquant", "sse", "zxqh", "nas", "pcf_xml")


@dataclass
class RunPaths:
    root: Path
    index_code: str
    index_name: str
    trade_date: str

    @property
    def run_dir(self) -> Path:
        return self.root / "data" / f"{self.index_code}_{self.index_name}" / self.trade_date

    @property
    def meta_dir(self) -> Path:
        return self.run_dir / "_meta"

    def category_dir(self, category: str, source: str | None = None) -> Path:
        if category not in CATEGORIES:
            raise KeyError(f"未知类别: {category}")
        rel, default_source = CATEGORIES[category]
        return self.run_dir / rel / (source or default_source)

    def file_name(self, label: str, source: str, ext: str = "csv") -> str:
        """统一命名：{标签}_{来源}_{指数}_{日期}.{ext}"""
        return f"{label}_{source}_{self.index_code}_{self.trade_date}.{ext}"

    def path_for(self, category: str, label: str, source: str, ext: str = "csv") -> Path:
        d = self.category_dir(category, source)
        d.mkdir(parents=True, exist_ok=True)
        return d / self.file_name(label, source, ext)


def resolve_dataset_root(start: Path | None = None) -> Path:
    """从 Notebook 或脚本位置推断数据集根目录。"""
    start = start or Path.cwd()
    if start.name == "_notebooks":
        return start.parent
    if (start / "data").is_dir() or (start / "_code").is_dir():
        return start
    if start.name == "数据集" or (start.parent / "_code").is_dir():
        return start
    candidate = start / "指数相关了解" / "数据集"
    if candidate.is_dir():
        return candidate
    return start


def build_run_paths(
    index_code: str,
    index_name: str,
    trade_date: str,
    dataset_root: Path | None = None,
) -> RunPaths:
    root = resolve_dataset_root(dataset_root)
    paths = RunPaths(root=root, index_code=index_code, index_name=index_name, trade_date=trade_date)
    paths.meta_dir.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        paths.category_dir(cat).mkdir(parents=True, exist_ok=True)
    for src in ALT_SOURCES:
        (paths.run_dir / "02_etf/申赎清单_pcf" / src).mkdir(parents=True, exist_ok=True)
    (paths.run_dir / "06_股指期货ticks" / "zxqh").mkdir(parents=True, exist_ok=True)
    return paths
