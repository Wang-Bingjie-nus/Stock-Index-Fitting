from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .minute_tick_cache_v03 import normalize_stock_code, normalize_trade_date
from .pareto_risk_optimizer_v04 import (
    ParetoOptimizerConfig,
    optimize_portfolio_pareto_risk,
)


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def floor_legal_quantity(quantity: float, minimum: int, step: int) -> int:
    if not np.isfinite(quantity) or quantity < minimum:
        return 0
    return int(minimum + np.floor((quantity - minimum) / step) * step)


def build_component_rules_from_weights(
    df_weights: pd.DataFrame,
    df_security_rules: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the same board-level buy rules used by correlation_v04."""

    require_columns(
        df_weights,
        ["stock_code", "stock_name", "raw_weight_pct"],
        "index weights",
    )
    require_columns(
        df_security_rules,
        ["exchange", "board", "buy_min_qty", "buy_qty_step"],
        "security rules",
    )

    frame = df_weights.copy()
    frame["stock_code"] = frame["stock_code"].map(normalize_stock_code)
    if frame["stock_code"].duplicated().any():
        raise ValueError("index weights contain duplicate stock codes.")

    frame["exchange"] = frame["stock_code"].str[-2:]
    frame["board"] = "MAIN"
    frame.loc[frame["stock_code"].str.startswith("68"), "board"] = "STAR"
    frame.loc[frame["stock_code"].str.startswith("30"), "board"] = "CHINEXT"
    frame.loc[frame["stock_code"].str.startswith("920"), "exchange"] = "BJ"
    frame.loc[frame["exchange"].eq("BJ"), "board"] = "MAIN"
    if (~frame["exchange"].isin(["SH", "SZ", "BJ"])).any():
        bad = frame.loc[~frame["exchange"].isin(["SH", "SZ", "BJ"]), "stock_code"]
        raise RuntimeError(f"Unsupported exchange found: {bad.tolist()[:20]}")

    rules = df_security_rules.drop(columns=["import_time"], errors="ignore").copy()
    rules["exchange"] = rules["exchange"].astype(str).str.strip().str.upper()
    rules["board"] = rules["board"].astype(str).str.strip().str.upper()
    if rules.duplicated(["exchange", "board"]).any():
        raise ValueError("security rules contain duplicate exchange/board keys.")

    frame = frame.merge(
        rules,
        on=["exchange", "board"],
        how="left",
        validate="many_to_one",
    )
    if frame[["buy_min_qty", "buy_qty_step"]].isna().any(axis=None):
        bad = frame.loc[
            frame[["buy_min_qty", "buy_qty_step"]].isna().any(axis=1),
            ["stock_code", "exchange", "board"],
        ]
        raise RuntimeError(
            f"Some components cannot match trading rules: {bad.to_dict('records')[:20]}"
        )
    frame["buy_min_qty"] = pd.to_numeric(frame["buy_min_qty"], errors="raise").astype(int)
    frame["buy_qty_step"] = pd.to_numeric(frame["buy_qty_step"], errors="raise").astype(int)
    return frame


def prepare_theoretical_portfolio(
    df_weights: pd.DataFrame,
    df_market_snapshot: pd.DataFrame,
    df_security_rules: pd.DataFrame,
    *,
    pricing_date: str,
    target_value: float,
    label: str = "initial",
    import_time: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build component rules and the theoretical amount/quantity table."""

    pricing_date = normalize_trade_date(pricing_date)
    target_value = float(target_value)
    if not np.isfinite(target_value) or target_value <= 0:
        raise ValueError("target_value must be positive and finite.")

    component_rules = build_component_rules_from_weights(
        df_weights,
        df_security_rules,
    )
    market = df_market_snapshot.copy()
    require_columns(market, ["stock_code", "close_price"], "market snapshot")
    market["stock_code"] = market["stock_code"].map(normalize_stock_code)
    market["close_price"] = pd.to_numeric(market["close_price"], errors="coerce")
    if market["stock_code"].duplicated().any():
        raise ValueError("market snapshot contains duplicate stock codes.")

    weight_codes = set(component_rules["stock_code"])
    market_codes = set(market["stock_code"])
    if weight_codes != market_codes:
        raise ValueError(
            "weight and market stock universes differ: "
            f"only_weights={sorted(weight_codes - market_codes)[:20]}, "
            f"only_market={sorted(market_codes - weight_codes)[:20]}"
        )

    theoretical = component_rules[
        [
            "stock_code",
            "stock_name",
            "raw_weight_pct",
            "exchange",
            "board",
            "buy_min_qty",
            "buy_qty_step",
        ]
    ].merge(
        market[["stock_code", "close_price"]],
        on="stock_code",
        validate="one_to_one",
    )
    theoretical["raw_weight_pct"] = pd.to_numeric(
        theoretical["raw_weight_pct"],
        errors="coerce",
    )
    raw_sum = float(theoretical["raw_weight_pct"].sum())
    if (
        theoretical["raw_weight_pct"].isna().any()
        or (theoretical["raw_weight_pct"] < 0).any()
        or raw_sum <= 0
    ):
        raise ValueError("raw_weight_pct contains invalid values.")
    if theoretical["close_price"].isna().any() or (theoretical["close_price"] <= 0).any():
        raise ValueError("market snapshot contains invalid close prices.")

    theoretical["raw_weight"] = theoretical["raw_weight_pct"] / raw_sum
    theoretical["target_stock_value"] = target_value
    theoretical["theoretical_amount"] = target_value * theoretical["raw_weight"]
    theoretical["theoretical_qty"] = (
        theoretical["theoretical_amount"] / theoretical["close_price"]
    )
    theoretical["pricing_date"] = pricing_date
    theoretical["portfolio_label"] = str(label)
    theoretical["import_time"] = import_time or datetime.now().astimezone().strftime(
        "%Y%m%d-%H%M"
    )
    return component_rules, theoretical


def build_target_portfolio_for_weights(
    df_weights: pd.DataFrame,
    pricing_date: str,
    target_value: float,
    df_security_rules: pd.DataFrame,
    *,
    allow_over_budget: bool = False,
    max_over_budget_ratio: float = 1.0,
    label: str = "rebalance",
    market_snapshot: pd.DataFrame | None = None,
    theoretical_portfolio: pd.DataFrame | None = None,
    excepted_code_ls: list[str] | None = None,
) -> dict:
    """Build the amount-greedy base exactly as in correlation_v04."""

    pricing_date = normalize_trade_date(pricing_date)
    target_value = float(target_value)
    if not np.isfinite(target_value) or target_value <= 0:
        raise RuntimeError(f"target_value must be positive for {label}: {target_value}")
    budget_limit = target_value * (
        float(max_over_budget_ratio) if allow_over_budget else 1.0
    )
    if budget_limit < target_value:
        raise RuntimeError("max_over_budget_ratio must be >= 1.0.")

    excepted_code_set = {
        normalize_stock_code(code) for code in (excepted_code_ls or [])
    }
    component_rules = build_component_rules_from_weights(
        df_weights,
        df_security_rules,
    )

    if theoretical_portfolio is None:
        if market_snapshot is None:
            raise ValueError(
                "market_snapshot is required when theoretical_portfolio is not supplied."
            )
        component_rules, theoretical = prepare_theoretical_portfolio(
            df_weights,
            market_snapshot,
            df_security_rules,
            pricing_date=pricing_date,
            target_value=target_value,
            label=label,
        )
    else:
        theoretical = theoretical_portfolio.copy()
        require_columns(
            theoretical,
            [
                "stock_code",
                "stock_name",
                "raw_weight_pct",
                "exchange",
                "board",
                "buy_min_qty",
                "buy_qty_step",
                "close_price",
                "raw_weight",
                "target_stock_value",
                "theoretical_amount",
                "theoretical_qty",
            ],
            f"{label} theoretical portfolio",
        )
        theoretical["stock_code"] = theoretical["stock_code"].map(
            normalize_stock_code
        )
        theoretical["close_price"] = pd.to_numeric(
            theoretical["close_price"], errors="coerce"
        )
        if theoretical["stock_code"].duplicated().any():
            raise ValueError(f"{label} theoretical portfolio has duplicate codes.")
        if theoretical["close_price"].isna().any() or (
            theoretical["close_price"] <= 0
        ).any():
            raise RuntimeError(f"Invalid precomputed close prices for {label}.")
        theoretical["portfolio_label"] = label
        theoretical["pricing_date"] = pricing_date
        market_snapshot = (
            theoretical[["stock_code", "close_price", "pricing_date"]].copy()
            if market_snapshot is None
            else market_snapshot.copy()
        )

    theoretical["is_excepted_code"] = theoretical["stock_code"].isin(
        excepted_code_set
    )
    target = theoretical.copy()
    target["initial_floor_qty"] = [
        0 if is_excepted else floor_legal_quantity(qty, int(minimum), int(step))
        for qty, minimum, step, is_excepted in zip(
            target["theoretical_qty"],
            target["buy_min_qty"],
            target["buy_qty_step"],
            target["is_excepted_code"],
        )
    ]
    target["greedy_added_qty"] = 0
    invested = float((target["initial_floor_qty"] * target["close_price"]).sum())

    while invested > budget_limit + 1e-8:
        reductions = []
        for idx, row in target.loc[target["initial_floor_qty"] > 0].iterrows():
            before_qty = int(row["initial_floor_qty"])
            remove_qty = int(
                row["buy_min_qty"]
                if before_qty == row["buy_min_qty"]
                else row["buy_qty_step"]
            )
            after_qty = before_qty - remove_qty
            theoretical_amount = float(row["theoretical_amount"])
            error_denom = max(abs(theoretical_amount), 1e-12)
            before_error = abs(
                before_qty * row["close_price"] - theoretical_amount
            ) / error_denom
            after_error = abs(
                after_qty * row["close_price"] - theoretical_amount
            ) / error_denom
            reductions.append(
                (after_error - before_error, row["stock_code"], idx, remove_qty)
            )
        if not reductions:
            raise RuntimeError(f"Cannot reduce {label} basket under budget limit.")
        _, _, idx, remove_qty = sorted(
            reductions,
            key=lambda item: (item[0], item[1]),
        )[0]
        invested -= remove_qty * float(target.at[idx, "close_price"])
        target.at[idx, "initial_floor_qty"] -= remove_qty

    greedy_steps = []
    while True:
        candidates = []
        for idx, row in target.iterrows():
            if bool(row.get("is_excepted_code", False)):
                continue
            before_qty = int(row["initial_floor_qty"] + row["greedy_added_qty"])
            add_qty = int(
                row["buy_min_qty"] if before_qty == 0 else row["buy_qty_step"]
            )
            cost = add_qty * float(row["close_price"])
            if invested + cost > budget_limit + 1e-8:
                continue
            theoretical_amount = float(row["theoretical_amount"])
            error_denom = max(abs(theoretical_amount), 1e-12)
            before_error = abs(
                before_qty * row["close_price"] - theoretical_amount
            ) / error_denom
            after_error = abs(
                (before_qty + add_qty) * row["close_price"] - theoretical_amount
            ) / error_denom
            improvement = before_error - after_error
            candidates.append(
                (improvement, after_error, row["stock_code"], idx, add_qty, cost)
            )
        if not candidates:
            break
        positive = [item for item in candidates if item[0] > 1e-12]
        if positive:
            selected = sorted(positive, key=lambda item: (-item[0], item[2]))[0]
        else:
            selected = sorted(candidates, key=lambda item: (item[1], item[2]))[0]
        improvement, _, stock_code, idx, add_qty, cost = selected
        target.at[idx, "greedy_added_qty"] += add_qty
        invested += cost
        greedy_steps.append(
            {
                "portfolio_label": label,
                "step_no": len(greedy_steps) + 1,
                "stock_code": stock_code,
                "added_qty": add_qty,
                "improvement": improvement,
                "cost": cost,
            }
        )

    target["target_qty"] = target["initial_floor_qty"] + target["greedy_added_qty"]
    target["target_market_value"] = target["target_qty"] * target["close_price"]
    target["is_held"] = target["target_qty"] > 0
    if target["target_market_value"].sum() > budget_limit + 1e-8:
        raise RuntimeError(f"{label} target basket exceeds budget limit.")
    for row in target.itertuples(index=False):
        qty = int(row.target_qty)
        if qty != 0 and (
            qty < int(row.buy_min_qty)
            or (qty - int(row.buy_min_qty)) % int(row.buy_qty_step) != 0
        ):
            raise RuntimeError(
                f"Illegal target quantity in {label}: {row.stock_code} {qty}"
            )

    summary = {
        "portfolio_label": label,
        "pricing_date": pricing_date,
        "target_stock_value": target_value,
        "budget_limit": budget_limit,
        "invested_amount": float(target["target_market_value"].sum()),
        "remaining_cash": float(target_value - target["target_market_value"].sum()),
        "invested_ratio": float(target["target_market_value"].sum() / target_value),
        "held_stock_count": int((target["target_qty"] > 0).sum()),
        "zero_qty_stock_count": int((target["target_qty"] == 0).sum()),
        "excepted_code_count": len(excepted_code_set),
        "excepted_component_count": int(target["is_excepted_code"].sum()),
        "excepted_weight_pct": float(
            target.loc[target["is_excepted_code"], "raw_weight_pct"].sum()
        ),
        "excepted_target_amount": float(
            target.loc[target["is_excepted_code"], "theoretical_amount"].sum()
        ),
        "greedy_step_count": len(greedy_steps),
    }
    return {
        "df_component_rules": component_rules,
        "df_market_snapshot": market_snapshot,
        "df_theoretical_portfolio": theoretical,
        "df_target_portfolio": target,
        "greedy_steps": pd.DataFrame(greedy_steps),
        "summary": summary,
    }


def build_pareto_portfolio_for_weights(
    df_weights: pd.DataFrame,
    pricing_date: str,
    target_value: float,
    df_security_rules: pd.DataFrame,
    *,
    covariance_matrix: pd.DataFrame,
    pareto_config: ParetoOptimizerConfig,
    allow_over_budget: bool,
    max_over_budget_ratio: float,
    label: str,
    market_snapshot: pd.DataFrame | None = None,
    theoretical_portfolio: pd.DataFrame | None = None,
    excepted_code_ls: list[str] | None = None,
) -> dict:
    """Amount-greedy base followed by the optimized v04 Pareto risk search."""

    if not allow_over_budget:
        raise ValueError(
            "v04 requires allow_over_budget=True for the confirmed funding interval."
        )

    base_build = build_target_portfolio_for_weights(
        df_weights,
        pricing_date,
        target_value,
        df_security_rules,
        allow_over_budget=True,
        max_over_budget_ratio=max_over_budget_ratio,
        label=f"{label}_amount_base",
        market_snapshot=market_snapshot,
        theoretical_portfolio=theoretical_portfolio,
        excepted_code_ls=excepted_code_ls,
    )
    base_target = base_build["df_target_portfolio"].copy()
    base_invested = float(base_target["target_market_value"].sum())
    budget_limit = float(target_value) * float(max_over_budget_ratio)
    if base_invested < float(target_value) - 1e-8:
        raise RuntimeError(
            "Amount greedy base cannot reach target inside the funding interval: "
            f"invested={base_invested:,.2f}, target={target_value:,.2f}, "
            f"cap={budget_limit:,.2f}"
        )
    if base_invested > budget_limit + 1e-8:
        raise RuntimeError(
            f"Amount greedy base exceeds cap: {base_invested:,.2f} > {budget_limit:,.2f}"
        )

    pareto_build = optimize_portfolio_pareto_risk(
        base_target,
        covariance_matrix,
        target_value=float(target_value),
        max_over_budget_ratio=float(max_over_budget_ratio),
        excepted_code_ls=excepted_code_ls,
        config=pareto_config,
        label=label,
    )
    summary = base_build["summary"].copy()
    summary.update(pareto_build["summary"])
    summary["amount_greedy_step_count"] = int(len(base_build["greedy_steps"]))
    summary["excepted_code_count"] = len(excepted_code_ls or [])

    return {
        "df_component_rules": base_build["df_component_rules"],
        "df_market_snapshot": base_build["df_market_snapshot"],
        "df_theoretical_portfolio": base_build["df_theoretical_portfolio"],
        "df_amount_greedy_base_target": base_target,
        "df_target_portfolio": pareto_build["df_target_portfolio"],
        "greedy_steps": pareto_build["selected_path"],
        "pareto_frontier": pareto_build["pareto_frontier"],
        "pareto_round_summary": pareto_build["round_summary"],
        "summary": summary,
        "base_state": pareto_build["base_state"],
        "selected_state": pareto_build["selected_state"],
    }


__all__ = [
    "ParetoOptimizerConfig",
    "build_component_rules_from_weights",
    "build_pareto_portfolio_for_weights",
    "build_target_portfolio_for_weights",
    "floor_legal_quantity",
    "prepare_theoretical_portfolio",
    "require_columns",
]
