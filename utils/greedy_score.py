from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GreedyScoreConfig:
    use_amount_error: bool = True
    use_industry_error: bool = True
    use_size_error: bool = True
    use_covariance_error: bool = False
    amount_weight: float = 1.0
    industry_weight: float = 1.0
    size_weight: float = 1.0
    covariance_weight: float = 0.0
    max_over_budget_ratio: float = 1.005


@dataclass
class ExposureMatrices:
    stock_codes: list[str]
    stock_names: list[str]
    price: pd.Series
    index_weight: pd.Series
    target_amount: pd.Series
    target_stock_value: float
    industry_matrix: pd.DataFrame
    index_industry_weight: pd.Series
    size_matrix: pd.DataFrame
    index_size_weight: pd.Series
    covariance_matrix: pd.DataFrame | None = None
    metadata: dict | None = None


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


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns {missing}; actual={frame.columns.tolist()}")


def _one_hot(labels: pd.Series, categories: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=labels.index, columns=categories)
    for category in categories:
        matrix.loc[labels.eq(category), category] = 1.0
    return matrix


def _prepare_industry_labels(stock_codes: list[str], industry_frame: pd.DataFrame | None) -> tuple[pd.Series, pd.DataFrame]:
    if industry_frame is None or industry_frame.empty:
        labels = pd.Series("UNKNOWN", index=stock_codes, name="industry_label")
        detail = pd.DataFrame({"stock_code": stock_codes, "industry_label": labels.values})
        return labels, detail

    frame = industry_frame.copy()
    if "xt_stock_code" in frame.columns:
        frame["_stock_code_norm"] = frame["xt_stock_code"].map(normalize_stock_code)
    elif "stock_code" in frame.columns:
        frame["_stock_code_norm"] = frame["stock_code"].map(normalize_stock_code)
    else:
        raise ValueError("industry_frame must contain xt_stock_code or stock_code.")

    name_col = "industry_name" if "industry_name" in frame.columns else None
    code_col = "industry_code" if "industry_code" in frame.columns else None
    if name_col and code_col:
        frame["industry_label"] = frame[code_col].astype(str) + "_" + frame[name_col].astype(str)
    elif name_col:
        frame["industry_label"] = frame[name_col].astype(str)
    elif code_col:
        frame["industry_label"] = frame[code_col].astype(str)
    else:
        raise ValueError("industry_frame must contain industry_code or industry_name.")

    mapper = frame.drop_duplicates("_stock_code_norm").set_index("_stock_code_norm")["industry_label"]
    labels = pd.Series(stock_codes, index=stock_codes).map(mapper).fillna("UNKNOWN")
    labels.name = "industry_label"
    detail = pd.DataFrame({"stock_code": stock_codes, "industry_label": labels.values})
    return labels, detail


def _prepare_size_labels(stock_codes: list[str], mcap_frame: pd.DataFrame | None) -> tuple[pd.Series, pd.DataFrame]:
    if mcap_frame is None or mcap_frame.empty or "mcap" not in mcap_frame.columns:
        labels = pd.Series("UNKNOWN", index=stock_codes, name="size_label")
        detail = pd.DataFrame({"stock_code": stock_codes, "mcap": np.nan, "size_label": labels.values})
        return labels, detail

    frame = mcap_frame.copy()
    if "xt_stock_code" in frame.columns:
        frame["_stock_code_norm"] = frame["xt_stock_code"].map(normalize_stock_code)
    elif "stock_code" in frame.columns:
        frame["_stock_code_norm"] = frame["stock_code"].map(normalize_stock_code)
    else:
        raise ValueError("mcap_frame must contain xt_stock_code or stock_code.")
    frame["mcap"] = pd.to_numeric(frame["mcap"], errors="coerce")
    mcap = frame.dropna(subset=["mcap"]).drop_duplicates("_stock_code_norm").set_index("_stock_code_norm")["mcap"]

    labels = pd.Series("UNKNOWN", index=stock_codes, name="size_label", dtype=object)
    known_codes = [code for code in stock_codes if code in mcap.index]
    if known_codes:
        ranked = mcap.reindex(known_codes).sort_values(ascending=False)
        n = len(ranked)
        split1 = int(np.ceil(n / 3))
        split2 = int(np.ceil(2 * n / 3))
        labels.loc[ranked.index[:split1]] = "large"
        labels.loc[ranked.index[split1:split2]] = "mid"
        labels.loc[ranked.index[split2:]] = "small"

    detail = pd.DataFrame({
        "stock_code": stock_codes,
        "mcap": mcap.reindex(stock_codes).values,
        "size_label": labels.reindex(stock_codes).values,
    })
    return labels, detail


def build_exposure_matrices(
    portfolio_frame: pd.DataFrame,
    *,
    industry_frame: pd.DataFrame | None = None,
    mcap_frame: pd.DataFrame | None = None,
    target_stock_value: float | None = None,
    covariance_matrix: pd.DataFrame | None = None,
) -> ExposureMatrices:
    _require_columns(portfolio_frame, ["stock_code", "close_price", "raw_weight_pct"], "portfolio_frame")

    base = portfolio_frame.copy()
    base["stock_code"] = base["stock_code"].map(normalize_stock_code)
    base["close_price"] = pd.to_numeric(base["close_price"], errors="coerce")
    base["raw_weight_pct"] = pd.to_numeric(base["raw_weight_pct"], errors="coerce")
    base = base.dropna(subset=["stock_code", "close_price", "raw_weight_pct"])
    base = base[base["close_price"] > 0].drop_duplicates("stock_code").sort_values("stock_code")
    if base.empty:
        raise ValueError("portfolio_frame has no valid rows.")
    if (base["raw_weight_pct"] < 0).any():
        raise ValueError("portfolio_frame contains negative raw_weight_pct.")

    stock_codes = base["stock_code"].tolist()
    stock_names = base["stock_name"].astype(str).tolist() if "stock_name" in base.columns else stock_codes
    price = pd.Series(base["close_price"].to_numpy(dtype=float), index=stock_codes, name="price")

    raw_weight = pd.Series(base["raw_weight_pct"].to_numpy(dtype=float), index=stock_codes)
    weight_sum = raw_weight.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("raw_weight_pct sum must be positive.")
    index_weight = (raw_weight / weight_sum).rename("index_weight")

    if target_stock_value is None:
        if "target_stock_value" in base.columns:
            target_stock_value = float(pd.to_numeric(base["target_stock_value"], errors="coerce").dropna().iloc[0])
        else:
            target_stock_value = 1.0
    target_stock_value = float(target_stock_value)
    if not np.isfinite(target_stock_value) or target_stock_value <= 0:
        raise ValueError("target_stock_value must be positive.")
    target_amount = (index_weight * target_stock_value).rename("target_amount")

    industry_labels, industry_detail = _prepare_industry_labels(stock_codes, industry_frame)
    industry_categories = sorted(industry_labels.unique().tolist())
    industry_matrix = _one_hot(industry_labels, industry_categories)
    index_industry_weight = (index_weight @ industry_matrix).rename("index_industry_weight")

    size_labels, size_detail = _prepare_size_labels(stock_codes, mcap_frame)
    size_categories = [item for item in ["large", "mid", "small", "UNKNOWN"] if item in set(size_labels)]
    size_matrix = _one_hot(size_labels, size_categories)
    index_size_weight = (index_weight @ size_matrix).rename("index_size_weight")

    cov = None
    if covariance_matrix is not None:
        cov = covariance_matrix.copy()
        cov.index = [normalize_stock_code(item) for item in cov.index]
        cov.columns = [normalize_stock_code(item) for item in cov.columns]
        cov = cov.reindex(index=stock_codes, columns=stock_codes)

    metadata = {
        "stock_count": len(stock_codes),
        "industry_count": len(industry_categories),
        "size_count": len(size_categories),
        "industry_detail": industry_detail,
        "size_detail": size_detail,
    }
    return ExposureMatrices(
        stock_codes=stock_codes,
        stock_names=stock_names,
        price=price,
        index_weight=index_weight,
        target_amount=target_amount,
        target_stock_value=target_stock_value,
        industry_matrix=industry_matrix,
        index_industry_weight=index_industry_weight,
        size_matrix=size_matrix,
        index_size_weight=index_size_weight,
        covariance_matrix=cov,
        metadata=metadata,
    )


def quantities_from_portfolio(portfolio_frame: pd.DataFrame, qty_col: str = "target_qty", stock_codes: list[str] | None = None) -> pd.Series:
    _require_columns(portfolio_frame, ["stock_code", qty_col], "portfolio_frame")
    frame = portfolio_frame[["stock_code", qty_col]].copy()
    frame["stock_code"] = frame["stock_code"].map(normalize_stock_code)
    frame[qty_col] = pd.to_numeric(frame[qty_col], errors="coerce").fillna(0.0)
    qty = frame.drop_duplicates("stock_code").set_index("stock_code")[qty_col].astype(float)
    if stock_codes is not None:
        qty = qty.reindex(stock_codes).fillna(0.0)
    return qty.rename("qty")


def compute_portfolio_state(qty: pd.Series | dict[str, float], matrices: ExposureMatrices) -> dict:
    qty_series = pd.Series(qty, dtype=float)
    qty_series.index = [normalize_stock_code(item) for item in qty_series.index]
    qty_series = qty_series.reindex(matrices.stock_codes).fillna(0.0)
    amount = (qty_series * matrices.price).rename("amount")
    invested_amount = float(amount.sum())
    if invested_amount > 0:
        portfolio_weight = (amount / invested_amount).rename("portfolio_weight")
    else:
        portfolio_weight = pd.Series(0.0, index=matrices.stock_codes, name="portfolio_weight")
    active_weight = (portfolio_weight - matrices.index_weight).rename("active_weight")
    return {
        "qty": qty_series,
        "amount": amount,
        "invested_amount": invested_amount,
        "invested_ratio": invested_amount / matrices.target_stock_value,
        "portfolio_weight": portfolio_weight,
        "active_weight": active_weight,
    }


def compute_score(qty: pd.Series | dict[str, float], matrices: ExposureMatrices, config: GreedyScoreConfig | None = None) -> dict:
    config = config or GreedyScoreConfig()
    state = compute_portfolio_state(qty, matrices)
    amount = state["amount"]
    portfolio_weight = state["portfolio_weight"]
    active_weight = state["active_weight"]
    invested_ratio = float(state["invested_ratio"])

    amount_diff = ((amount - matrices.target_amount) / matrices.target_stock_value).rename("amount_diff")
    amount_error = float(amount_diff.abs().sum())

    portfolio_industry_weight = (portfolio_weight @ matrices.industry_matrix).rename("portfolio_industry_weight")
    industry_diff = portfolio_industry_weight - matrices.index_industry_weight
    industry_error = float(industry_diff.abs().sum())

    portfolio_size_weight = (portfolio_weight @ matrices.size_matrix).rename("portfolio_size_weight")
    size_diff = portfolio_size_weight - matrices.index_size_weight
    size_error = float(size_diff.abs().sum())

    covariance_error = 0.0
    if config.use_covariance_error:
        if matrices.covariance_matrix is None:
            raise ValueError("covariance_matrix is required when use_covariance_error=True.")
        cov = matrices.covariance_matrix.to_numpy(dtype=float)
        aw = active_weight.to_numpy(dtype=float)
        covariance_error = float(aw.T @ cov @ aw)

    under_budget_gap = float(max(0.0, 1.0 - invested_ratio))
    over_budget_gap = float(max(0.0, invested_ratio - 1.0))

    total_score = 0.0
    amount_score_contribution = 0.0
    industry_score_contribution = 0.0
    size_score_contribution = 0.0
    covariance_score_contribution = 0.0
    if config.use_amount_error:
        amount_score_contribution = config.amount_weight * amount_error
        total_score += amount_score_contribution
    if config.use_industry_error:
        industry_score_contribution = config.industry_weight * industry_error
        total_score += industry_score_contribution
    if config.use_size_error:
        size_score_contribution = config.size_weight * size_error
        total_score += size_score_contribution
    if config.use_covariance_error:
        covariance_score_contribution = config.covariance_weight * covariance_error
        total_score += covariance_score_contribution

    contribution_denom = abs(total_score) if abs(total_score) > 1e-15 else np.nan

    return {
        "total_score": float(total_score),
        "amount_error": amount_error,
        "industry_error": industry_error,
        "size_error": size_error,
        "covariance_error": covariance_error,
        "amount_score_contribution": float(amount_score_contribution),
        "industry_score_contribution": float(industry_score_contribution),
        "size_score_contribution": float(size_score_contribution),
        "covariance_score_contribution": float(covariance_score_contribution),
        "amount_score_contribution_pct": float(amount_score_contribution / contribution_denom) if np.isfinite(contribution_denom) else np.nan,
        "industry_score_contribution_pct": float(industry_score_contribution / contribution_denom) if np.isfinite(contribution_denom) else np.nan,
        "size_score_contribution_pct": float(size_score_contribution / contribution_denom) if np.isfinite(contribution_denom) else np.nan,
        "covariance_score_contribution_pct": float(covariance_score_contribution / contribution_denom) if np.isfinite(contribution_denom) else np.nan,
        "under_budget_gap": under_budget_gap,
        "over_budget_gap": over_budget_gap,
        "invested_amount": float(state["invested_amount"]),
        "invested_ratio": invested_ratio,
        "portfolio_weight_sum": float(portfolio_weight.sum()),
        "index_weight_sum": float(matrices.index_weight.sum()),
        "portfolio_industry_weight_sum": float(portfolio_industry_weight.sum()),
        "index_industry_weight_sum": float(matrices.index_industry_weight.sum()),
        "portfolio_size_weight_sum": float(portfolio_size_weight.sum()),
        "index_size_weight_sum": float(matrices.index_size_weight.sum()),
        "portfolio_industry_weight": portfolio_industry_weight,
        "index_industry_weight": matrices.index_industry_weight,
        "portfolio_size_weight": portfolio_size_weight,
        "index_size_weight": matrices.index_size_weight,
    }


def score_to_frame(score: dict) -> pd.DataFrame:
    scalar_keys = [
        "total_score",
        "amount_error",
        "industry_error",
        "size_error",
        "covariance_error",
        "amount_score_contribution",
        "industry_score_contribution",
        "size_score_contribution",
        "covariance_score_contribution",
        "amount_score_contribution_pct",
        "industry_score_contribution_pct",
        "size_score_contribution_pct",
        "covariance_score_contribution_pct",
        "under_budget_gap",
        "over_budget_gap",
        "invested_amount",
        "invested_ratio",
        "portfolio_weight_sum",
        "index_weight_sum",
        "portfolio_industry_weight_sum",
        "index_industry_weight_sum",
        "portfolio_size_weight_sum",
        "index_size_weight_sum",
    ]
    return pd.DataFrame([{"metric": key, "value": score[key]} for key in scalar_keys])
