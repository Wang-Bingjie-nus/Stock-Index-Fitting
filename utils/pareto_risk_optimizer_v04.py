from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

import numpy as np
import pandas as pd

from .risk_model_v02 import MINUTES_PER_TRADING_DAY, TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class ParetoOptimizerConfig:
    risk_candidate_count: int = 10
    amount_candidate_count: int = 10
    beam_width: int = 20
    max_rounds: int = 50
    stale_rounds_to_stop: int = 2
    legal_neighbor_steps: int = 2
    annualization_periods: int = MINUTES_PER_TRADING_DAY * TRADING_DAYS_PER_YEAR
    objective_tolerance: float = 1e-14


@dataclass
class PortfolioState:
    qty: np.ndarray
    invested_amount: float
    amount_error: float
    tracking_variance: float
    tracking_error_annual: float
    round_no: int
    action: dict = field(default_factory=dict)
    parent: "PortfolioState | None" = None
    state_id: str = ""
    active_weight: np.ndarray | None = None
    covariance_active: np.ndarray | None = None


def _normalize_stock_code(value) -> str:
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


def _state_id(qty: np.ndarray) -> str:
    payload = np.rint(qty).astype(np.int64, copy=False).tobytes()
    return hashlib.md5(payload).hexdigest()[:16]


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _is_legal_quantity(qty: int, minimum: int, step: int) -> bool:
    return qty == 0 or (qty >= minimum and (qty - minimum) % step == 0)


def _legal_totals_near(
    desired_total: float,
    *,
    minimum: int,
    step: int,
    lower: int,
    upper: int,
    neighbors: int,
) -> list[int]:
    if minimum <= 0 or step <= 0 or upper < lower:
        return []

    candidates: set[int] = set()
    if lower <= 0 <= upper:
        candidates.add(0)

    legal_lower = max(lower, minimum)
    if legal_lower <= upper:
        k_lower = max(0, math.ceil((legal_lower - minimum) / step))
        k_upper = math.floor((upper - minimum) / step)
        if k_lower <= k_upper:
            k_center = int(round((desired_total - minimum) / step))
            for k in range(k_center - neighbors, k_center + neighbors + 1):
                if k_lower <= k <= k_upper:
                    candidates.add(minimum + k * step)
            candidates.add(minimum + k_lower * step)
            candidates.add(minimum + k_upper * step)

    return sorted(candidates)


def _top_k_available(
    values: np.ndarray,
    available: np.ndarray,
    count: int,
    *,
    largest: bool = False,
) -> list[int]:
    """Return deterministic top-k indices without sorting the full universe."""

    candidates = np.flatnonzero(available)
    count = min(max(int(count), 0), len(candidates))
    if count == 0:
        return []

    candidate_values = np.asarray(values, dtype=float)[candidates]
    ranking_values = -candidate_values if largest else candidate_values
    ranking_values = np.where(np.isfinite(ranking_values), ranking_values, np.inf)

    if count < len(candidates):
        partition = np.argpartition(ranking_values, count - 1)[:count]
        cutoff = float(ranking_values[partition].max())
        strict = np.flatnonzero(ranking_values < cutoff)
        tied = np.flatnonzero(ranking_values == cutoff)
        tied = tied[np.argsort(candidates[tied], kind="stable")]
        selected_positions = np.concatenate([strict, tied[: count - len(strict)]])
    else:
        selected_positions = np.arange(len(candidates))

    selected_indices = candidates[selected_positions]
    selected_values = ranking_values[selected_positions]
    order = np.lexsort((selected_indices, selected_values))
    return selected_indices[order].astype(int).tolist()


def _local_pareto_positions(
    amount_error: np.ndarray,
    tracking_error: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Return positions on the two-objective Pareto front of one parent batch."""

    count = len(amount_error)
    if count == 0:
        return np.empty(0, dtype=np.int64)

    serial = np.arange(count, dtype=np.int64)
    order = np.lexsort((serial, tracking_error, amount_error))
    ordered_tracking = tracking_error[order]
    prior_best = np.minimum.accumulate(ordered_tracking)
    keep = np.ones(count, dtype=bool)
    if count > 1:
        keep[1:] = ordered_tracking[1:] < prior_best[:-1] - tolerance
    return order[keep]


class _ParetoPortfolioOptimizer:
    def __init__(
        self,
        base_portfolio: pd.DataFrame,
        covariance_matrix: pd.DataFrame,
        *,
        target_value: float,
        max_over_budget_ratio: float,
        excepted_code_ls: list[str] | None,
        config: ParetoOptimizerConfig,
        label: str,
    ) -> None:
        required = [
            "stock_code",
            "close_price",
            "raw_weight_pct",
            "buy_min_qty",
            "buy_qty_step",
            "target_qty",
        ]
        _require_columns(base_portfolio, required, "base_portfolio")

        frame = base_portfolio.copy()
        frame["stock_code"] = frame["stock_code"].map(_normalize_stock_code)
        frame = frame.drop_duplicates("stock_code").sort_values("stock_code").reset_index(drop=True)
        for column in ["close_price", "raw_weight_pct", "buy_min_qty", "buy_qty_step", "target_qty"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")

        self.frame = frame
        self.codes = frame["stock_code"].tolist()
        self.names = frame.get("stock_name", pd.Series(self.codes)).astype(str).tolist()
        self.n = len(frame)
        self.price = frame["close_price"].to_numpy(dtype=float)
        self.minimum = frame["buy_min_qty"].to_numpy(dtype=int)
        self.step = frame["buy_qty_step"].to_numpy(dtype=int)
        self.target_value = float(target_value)
        self.max_ratio = float(max_over_budget_ratio)
        self.budget_limit = self.target_value * self.max_ratio
        self.config = config
        self.label = str(label)

        raw_weight = frame["raw_weight_pct"].to_numpy(dtype=float)
        if raw_weight.sum() <= 0:
            raise ValueError("raw_weight_pct must sum to a positive value.")
        self.index_weight = raw_weight / raw_weight.sum()
        self.target_amount = self.index_weight * self.target_value

        cov = covariance_matrix.copy()
        cov.index = [_normalize_stock_code(item) for item in cov.index]
        cov.columns = [_normalize_stock_code(item) for item in cov.columns]
        cov = cov.reindex(index=self.codes, columns=self.codes)
        if cov.isna().any().any():
            raise ValueError("covariance_matrix is missing stocks required by base_portfolio.")
        self.covariance = cov.to_numpy(dtype=float, copy=True)
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        self.covariance_index = self.covariance @ self.index_weight
        self.index_variance = float(self.index_weight @ self.covariance_index)

        excepted = {_normalize_stock_code(code) for code in (excepted_code_ls or [])}
        self.is_excepted = np.array([code in excepted for code in self.codes], dtype=bool)
        base_qty = np.rint(frame["target_qty"].to_numpy(dtype=float)).astype(np.int64)
        if np.any(base_qty[self.is_excepted] != 0):
            bad = [self.codes[index] for index in np.flatnonzero(self.is_excepted & (base_qty != 0))]
            raise ValueError(f"Excepted stocks have non-zero base quantities: {bad}")
        self.base_state = self._state_from_qty(base_qty, round_no=0, action={"action_type": "amount_greedy_base"})
        self._validate_state(self.base_state)
        self._discovered_state_ids = {self.base_state.state_id}

    def _hydrate(self, state: PortfolioState) -> PortfolioState:
        if state.active_weight is None or state.covariance_active is None:
            amount = state.qty.astype(float) * self.price
            weight = amount / state.invested_amount
            active = weight - self.index_weight
            state.active_weight = active
            state.covariance_active = self.covariance @ active
        return state

    def _state_from_qty(
        self,
        qty: np.ndarray,
        *,
        round_no: int,
        action: dict,
        parent: PortfolioState | None = None,
    ) -> PortfolioState:
        qty = np.rint(qty).astype(np.int64, copy=True)
        amount = qty.astype(float) * self.price
        invested = float(amount.sum())
        if invested <= 0:
            raise ValueError("Portfolio invested amount must be positive.")
        active = amount / invested - self.index_weight
        covariance_active = self.covariance @ active
        variance = max(float(active @ covariance_active), 0.0)
        amount_error = float(np.abs((amount - self.target_amount) / self.target_value).sum())
        return PortfolioState(
            qty=qty,
            invested_amount=invested,
            amount_error=amount_error,
            tracking_variance=variance,
            tracking_error_annual=math.sqrt(variance * self.config.annualization_periods),
            round_no=int(round_no),
            action=action,
            parent=parent,
            state_id=_state_id(qty),
            active_weight=active,
            covariance_active=covariance_active,
        )

    def _validate_state(self, state: PortfolioState) -> None:
        tolerance = 1e-8
        if state.invested_amount < self.target_value - tolerance:
            raise ValueError(
                f"Portfolio is below target: {state.invested_amount:,.2f} < {self.target_value:,.2f}"
            )
        if state.invested_amount > self.budget_limit + tolerance:
            raise ValueError(
                f"Portfolio exceeds budget cap: {state.invested_amount:,.2f} > {self.budget_limit:,.2f}"
            )
        if np.any(state.qty[self.is_excepted] != 0):
            raise ValueError("Portfolio contains an excepted stock.")
        for index, qty in enumerate(state.qty):
            if not _is_legal_quantity(int(qty), int(self.minimum[index]), int(self.step[index])):
                raise ValueError(f"Illegal quantity: {self.codes[index]} qty={qty}")

    def _collect_candidate_actions(self, state: PortfolioState) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        buy_indices, sell_indices = self._screen_candidate_indices(state)
        actions: set[tuple[int, int, int, int, int]] = set()
        raw_count = 0

        for buy_index in buy_indices:
            desired = self._continuous_add_amount(state, buy_index)
            for add_qty in self._add_qty_options(state, buy_index, desired):
                raw_count += 1
                actions.add((0, buy_index, int(add_qty), -1, 0))

        for sell_index in sell_indices:
            desired = self._continuous_reduce_amount(state, sell_index)
            for remove_qty in self._remove_qty_options(state, sell_index, desired):
                raw_count += 1
                actions.add((1, sell_index, -int(remove_qty), -1, 0))

        for sell_index in sell_indices:
            for buy_index in buy_indices:
                if sell_index == buy_index:
                    continue
                desired = self._continuous_swap_amount(state, sell_index, buy_index)
                for sell_qty, buy_qty in self._swap_options(
                    state, sell_index, buy_index, desired
                ):
                    raw_count += 1
                    actions.add((2, sell_index, -int(sell_qty), buy_index, int(buy_qty)))

        if not actions:
            empty = np.empty(0, dtype=np.int64)
            return {
                "action_type": empty,
                "index_a": empty,
                "delta_qty_a": empty,
                "index_b": empty,
                "delta_qty_b": empty,
            }, {"raw_candidate_count": raw_count, "unique_action_count": 0}

        action_array = np.asarray(sorted(actions), dtype=np.int64)
        return {
            "action_type": action_array[:, 0],
            "index_a": action_array[:, 1],
            "delta_qty_a": action_array[:, 2],
            "index_b": action_array[:, 3],
            "delta_qty_b": action_array[:, 4],
        }, {
            "raw_candidate_count": raw_count,
            "unique_action_count": len(action_array),
        }

    @staticmethod
    def _legal_quantity_mask(
        qty: np.ndarray,
        minimum: np.ndarray,
        step: np.ndarray,
    ) -> np.ndarray:
        return (qty == 0) | ((qty >= minimum) & ((qty - minimum) % step == 0))

    def _score_candidate_actions(
        self,
        parent: PortfolioState,
        actions: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        count = len(actions["action_type"])
        if count == 0:
            return {**actions, "invested_amount": np.empty(0), "amount_error": np.empty(0),
                    "tracking_variance": np.empty(0), "tracking_error_annual": np.empty(0)}

        index_a = actions["index_a"]
        delta_qty_a = actions["delta_qty_a"]
        index_b = actions["index_b"]
        delta_qty_b = actions["delta_qty_b"]
        has_b = index_b >= 0
        safe_index_b = np.where(has_b, index_b, index_a)

        new_qty_a = parent.qty[index_a] + delta_qty_a
        new_qty_b = parent.qty[safe_index_b] + np.where(has_b, delta_qty_b, 0)
        legal_a = self._legal_quantity_mask(
            new_qty_a, self.minimum[index_a], self.step[index_a]
        )
        legal_b = ~has_b | self._legal_quantity_mask(
            new_qty_b, self.minimum[safe_index_b], self.step[safe_index_b]
        )
        allowed_a = ~(self.is_excepted[index_a] & (new_qty_a != 0))
        allowed_b = ~has_b | ~(self.is_excepted[safe_index_b] & (new_qty_b != 0))

        delta_amount_a = delta_qty_a.astype(float) * self.price[index_a]
        delta_amount_b = np.where(
            has_b,
            delta_qty_b.astype(float) * self.price[safe_index_b],
            0.0,
        )
        invested = parent.invested_amount + delta_amount_a + delta_amount_b
        tolerance = 1e-8
        feasible = (
            (new_qty_a >= 0)
            & (~has_b | (new_qty_b >= 0))
            & legal_a
            & legal_b
            & allowed_a
            & allowed_b
            & (invested >= self.target_value - tolerance)
            & (invested <= self.budget_limit + tolerance)
        )
        positions = np.flatnonzero(feasible)
        if len(positions) == 0:
            return {
                key: value[:0] for key, value in actions.items()
            } | {
                "invested_amount": np.empty(0),
                "amount_error": np.empty(0),
                "tracking_variance": np.empty(0),
                "tracking_error_annual": np.empty(0),
            }

        scored = {key: value[positions] for key, value in actions.items()}
        index_a = index_a[positions]
        safe_index_b = safe_index_b[positions]
        has_b = has_b[positions]
        delta_amount_a = delta_amount_a[positions]
        delta_amount_b = delta_amount_b[positions]
        invested = invested[positions]

        old_amount = parent.qty.astype(float) * self.price
        old_a = old_amount[index_a]
        target_a = self.target_amount[index_a]
        amount_error = parent.amount_error + (
            np.abs(old_a + delta_amount_a - target_a) - np.abs(old_a - target_a)
        ) / self.target_value
        old_b = old_amount[safe_index_b]
        target_b = self.target_amount[safe_index_b]
        amount_error += np.where(
            has_b,
            (
                np.abs(old_b + delta_amount_b - target_b)
                - np.abs(old_b - target_b)
            )
            / self.target_value,
            0.0,
        )

        parent = self._hydrate(parent)
        ratio = parent.invested_amount / invested
        dense_b = ratio - 1.0
        active_cov_index = float(parent.active_weight @ self.covariance_index)
        dense_variance = (
            np.square(ratio) * parent.tracking_variance
            + np.square(dense_b) * self.index_variance
            + 2.0 * ratio * dense_b * active_cov_index
        )

        sparse_weight_a = delta_amount_a / invested
        sparse_weight_b = delta_amount_b / invested
        covariance_dense_a = (
            ratio * parent.covariance_active[index_a]
            + dense_b * self.covariance_index[index_a]
        )
        covariance_dense_b = (
            ratio * parent.covariance_active[safe_index_b]
            + dense_b * self.covariance_index[safe_index_b]
        )
        sparse_linear = 2.0 * (
            sparse_weight_a * covariance_dense_a
            + sparse_weight_b * covariance_dense_b
        )
        sparse_quadratic = (
            np.square(sparse_weight_a) * self.covariance[index_a, index_a]
            + 2.0
            * sparse_weight_a
            * sparse_weight_b
            * self.covariance[index_a, safe_index_b]
            + np.square(sparse_weight_b)
            * self.covariance[safe_index_b, safe_index_b]
        )
        variance = np.maximum(dense_variance + sparse_linear + sparse_quadratic, 0.0)
        tracking_error = np.sqrt(variance * self.config.annualization_periods)

        scored.update(
            invested_amount=invested,
            amount_error=amount_error,
            tracking_variance=variance,
            tracking_error_annual=tracking_error,
        )
        return scored

    def _materialize_candidates(
        self,
        parent: PortfolioState,
        scored: dict[str, np.ndarray],
        *,
        round_no: int,
    ) -> tuple[list[PortfolioState], int]:
        candidates = []
        duplicate_count = 0
        action_names = ("add", "reduce", "swap")

        for position in range(len(scored["action_type"])):
            index_a = int(scored["index_a"][position])
            delta_qty_a = int(scored["delta_qty_a"][position])
            index_b = int(scored["index_b"][position])
            delta_qty_b = int(scored["delta_qty_b"][position])
            trades = [(index_a, delta_qty_a)]
            if index_b >= 0:
                trades.append((index_b, delta_qty_b))

            new_qty = parent.qty.copy()
            for index, delta_qty in trades:
                new_qty[index] += delta_qty
            state_id = _state_id(new_qty)
            if state_id in self._discovered_state_ids:
                duplicate_count += 1
                continue
            self._discovered_state_ids.add(state_id)

            invested = float(scored["invested_amount"][position])
            amount_error = float(scored["amount_error"][position])
            variance = float(scored["tracking_variance"][position])
            tracking_error = float(scored["tracking_error_annual"][position])
            trade_amounts = {
                index: delta_qty * self.price[index] for index, delta_qty in trades
            }
            action = {
                "action_type": action_names[int(scored["action_type"][position])],
                "trade_count": len(trades),
                "net_trade_amount": float(sum(trade_amounts.values())),
                "invested_before": parent.invested_amount,
                "invested_after": invested,
                "amount_error_before": parent.amount_error,
                "amount_error_after": amount_error,
                "tracking_error_before": parent.tracking_error_annual,
                "tracking_error_after": tracking_error,
            }
            for trade_position, (index, delta_qty) in enumerate(sorted(trades), start=1):
                action[f"stock_code_{trade_position}"] = self.codes[index]
                action[f"stock_name_{trade_position}"] = self.names[index]
                action[f"delta_qty_{trade_position}"] = delta_qty
                action[f"trade_amount_{trade_position}"] = float(trade_amounts[index])

            candidates.append(
                PortfolioState(
                    qty=new_qty,
                    invested_amount=invested,
                    amount_error=amount_error,
                    tracking_variance=variance,
                    tracking_error_annual=tracking_error,
                    round_no=int(round_no),
                    action=action,
                    parent=parent,
                    state_id=state_id,
                )
            )
        return candidates, duplicate_count

    def _screen_candidate_indices(self, state: PortfolioState) -> tuple[list[int], list[int]]:
        state = self._hydrate(state)
        amount = state.qty.astype(float) * self.price
        covariance_active = state.covariance_active

        buy_available = ~self.is_excepted
        buy_risk = _top_k_available(
            covariance_active,
            buy_available,
            self.config.risk_candidate_count,
        )
        buy_amount = _top_k_available(
            self.target_amount - amount,
            buy_available,
            self.config.amount_candidate_count,
            largest=True,
        )

        sell_available = state.qty > 0
        sell_risk = _top_k_available(
            covariance_active,
            sell_available,
            self.config.risk_candidate_count,
            largest=True,
        )
        sell_amount = _top_k_available(
            amount - self.target_amount,
            sell_available,
            self.config.amount_candidate_count,
            largest=True,
        )

        buy = list(dict.fromkeys(buy_risk + buy_amount))
        sell = list(dict.fromkeys(sell_risk + sell_amount))
        return buy, sell

    def _continuous_add_amount(self, state: PortfolioState, index: int) -> float:
        state = self._hydrate(state)
        active = state.active_weight
        cov_active = state.covariance_active
        active_cov_index = float(active @ self.covariance_index)
        active_cov_weight = state.tracking_variance + active_cov_index
        covariance_weight_i = cov_active[index] + self.covariance_index[index]
        weight_variance = state.tracking_variance + 2.0 * active_cov_index + self.index_variance
        a_q_d = cov_active[index] - active_cov_weight
        d_q_d = self.covariance[index, index] - 2.0 * covariance_weight_i + weight_variance
        if d_q_d <= 0:
            return 0.0
        t_star = max(0.0, -a_q_d / d_q_d)
        slack = max(0.0, self.budget_limit - state.invested_amount)
        t_max = slack / (state.invested_amount + slack) if slack > 0 else 0.0
        t_star = min(t_star, t_max, 1.0 - 1e-12)
        return state.invested_amount * t_star / max(1.0 - t_star, 1e-12)

    def _continuous_reduce_amount(self, state: PortfolioState, index: int) -> float:
        state = self._hydrate(state)
        active = state.active_weight
        cov_active = state.covariance_active
        active_cov_index = float(active @ self.covariance_index)
        active_cov_weight = state.tracking_variance + active_cov_index
        covariance_weight_i = cov_active[index] + self.covariance_index[index]
        weight_variance = state.tracking_variance + 2.0 * active_cov_index + self.index_variance
        a_q_d = active_cov_weight - cov_active[index]
        d_q_d = weight_variance - 2.0 * covariance_weight_i + self.covariance[index, index]
        if d_q_d <= 0:
            return 0.0
        t_star = max(0.0, -a_q_d / d_q_d)
        max_remove_amount = min(
            state.qty[index] * self.price[index],
            state.invested_amount - self.target_value,
        )
        if max_remove_amount <= 0:
            return 0.0
        t_max = max_remove_amount / max(state.invested_amount - max_remove_amount, 1e-12)
        t_star = min(t_star, t_max)
        return state.invested_amount * t_star / (1.0 + t_star)

    def _continuous_swap_amount(self, state: PortfolioState, sell_index: int, buy_index: int) -> float:
        state = self._hydrate(state)
        d_q_d = (
            self.covariance[sell_index, sell_index]
            + self.covariance[buy_index, buy_index]
            - 2.0 * self.covariance[sell_index, buy_index]
        )
        if d_q_d <= 0:
            return 0.0
        a_q_d = state.covariance_active[buy_index] - state.covariance_active[sell_index]
        amount = max(0.0, -state.invested_amount * a_q_d / d_q_d)
        return min(amount, state.qty[sell_index] * self.price[sell_index])

    def _add_qty_options(self, state: PortfolioState, index: int, desired_amount: float) -> list[int]:
        current = int(state.qty[index])
        max_add_qty = int(math.floor((self.budget_limit - state.invested_amount) / self.price[index] + 1e-12))
        if max_add_qty <= 0:
            return []
        totals = _legal_totals_near(
            current + desired_amount / self.price[index],
            minimum=int(self.minimum[index]),
            step=int(self.step[index]),
            lower=current + 1,
            upper=current + max_add_qty,
            neighbors=self.config.legal_neighbor_steps,
        )
        return sorted({total - current for total in totals if total > current})

    def _remove_qty_options(self, state: PortfolioState, index: int, desired_amount: float) -> list[int]:
        current = int(state.qty[index])
        max_remove_qty = int(
            min(current, math.floor((state.invested_amount - self.target_value) / self.price[index] + 1e-12))
        )
        if max_remove_qty <= 0:
            return []
        totals = _legal_totals_near(
            current - desired_amount / self.price[index],
            minimum=int(self.minimum[index]),
            step=int(self.step[index]),
            lower=max(0, current - max_remove_qty),
            upper=current - 1,
            neighbors=self.config.legal_neighbor_steps,
        )
        return sorted({current - total for total in totals if total < current})

    def _swap_options(
        self,
        state: PortfolioState,
        sell_index: int,
        buy_index: int,
        desired_amount: float,
    ) -> list[tuple[int, int]]:
        sell_current = int(state.qty[sell_index])
        sell_totals = _legal_totals_near(
            sell_current - desired_amount / self.price[sell_index],
            minimum=int(self.minimum[sell_index]),
            step=int(self.step[sell_index]),
            lower=0,
            upper=sell_current - 1,
            neighbors=self.config.legal_neighbor_steps,
        )
        sell_deltas = sorted({sell_current - total for total in sell_totals if total < sell_current})
        options: set[tuple[int, int]] = set()
        buy_current = int(state.qty[buy_index])

        for sell_qty in sell_deltas:
            sell_amount = sell_qty * self.price[sell_index]
            invested_after_sell = state.invested_amount - sell_amount
            min_buy_amount = max(0.0, self.target_value - invested_after_sell)
            max_buy_amount = self.budget_limit - invested_after_sell
            if max_buy_amount <= 0:
                continue
            min_buy_qty = int(math.ceil(min_buy_amount / self.price[buy_index] - 1e-12))
            max_buy_qty = int(math.floor(max_buy_amount / self.price[buy_index] + 1e-12))
            if max_buy_qty <= 0:
                continue
            buy_totals = _legal_totals_near(
                buy_current + sell_amount / self.price[buy_index],
                minimum=int(self.minimum[buy_index]),
                step=int(self.step[buy_index]),
                lower=buy_current + max(1, min_buy_qty),
                upper=buy_current + max_buy_qty,
                neighbors=self.config.legal_neighbor_steps,
            )
            for buy_total in buy_totals:
                if buy_total > buy_current:
                    options.add((sell_qty, buy_total - buy_current))
        return sorted(options)

    def _generate_candidates(
        self,
        state: PortfolioState,
        round_no: int,
    ) -> tuple[list[PortfolioState], dict[str, int]]:
        actions, stats = self._collect_candidate_actions(state)
        scored = self._score_candidate_actions(state, actions)
        feasible_count = len(scored["action_type"])
        local_positions = _local_pareto_positions(
            scored["amount_error"],
            scored["tracking_error_annual"],
            tolerance=self.config.objective_tolerance,
        )
        local_front = {
            key: values[local_positions] for key, values in scored.items()
        }
        candidates, duplicate_count = self._materialize_candidates(
            state,
            local_front,
            round_no=round_no,
        )
        stats.update(
            feasible_candidate_count=feasible_count,
            local_pareto_candidate_count=len(local_positions),
            materialized_candidate_count=len(candidates),
            duplicate_state_count=duplicate_count,
        )
        return candidates, stats

    def _pareto_front(self, states: list[PortfolioState]) -> list[PortfolioState]:
        unique: dict[str, PortfolioState] = {}
        for state in states:
            # Preserve the earliest path to a quantity vector. Replacing an
            # archived state with a later round-trip can create cyclic paths.
            unique.setdefault(state.state_id, state)
        ordered = sorted(
            unique.values(),
            key=lambda state: (state.amount_error, state.tracking_error_annual, state.state_id),
        )
        frontier = []
        best_tracking = math.inf
        tolerance = self.config.objective_tolerance
        for state in ordered:
            if state.tracking_error_annual < best_tracking - tolerance:
                frontier.append(state)
                best_tracking = state.tracking_error_annual
        return frontier

    @staticmethod
    def _crowding_distance(states: list[PortfolioState]) -> np.ndarray:
        count = len(states)
        if count <= 2:
            return np.full(count, np.inf)
        distance = np.zeros(count, dtype=float)
        objectives = np.array(
            [[state.amount_error, state.tracking_error_annual] for state in states],
            dtype=float,
        )
        for column in range(objectives.shape[1]):
            order = np.argsort(objectives[:, column])
            distance[order[0]] = np.inf
            distance[order[-1]] = np.inf
            span = objectives[order[-1], column] - objectives[order[0], column]
            if span <= 0:
                continue
            for position in range(1, count - 1):
                left = objectives[order[position - 1], column]
                right = objectives[order[position + 1], column]
                distance[order[position]] += (right - left) / span
        return distance

    def _select_beam(
        self,
        frontier: list[PortfolioState],
        *,
        excluded_state_ids: set[str] | None = None,
    ) -> list[PortfolioState]:
        excluded_state_ids = excluded_state_ids or set()
        available = [
            state for state in frontier if state.state_id not in excluded_state_ids
        ]
        if len(available) <= self.config.beam_width:
            return available

        crowding = self._crowding_distance(frontier)
        order = sorted(
            range(len(frontier)),
            key=lambda index: (
                -crowding[index],
                frontier[index].amount_error,
                frontier[index].tracking_error_annual,
                frontier[index].state_id,
            ),
        )
        selected = []
        for index in order:
            state = frontier[index]
            if state.state_id in excluded_state_ids:
                continue
            selected.append(state)
            if len(selected) >= self.config.beam_width:
                break
        return selected

    @staticmethod
    def _select_ideal_point(frontier: list[PortfolioState]) -> tuple[PortfolioState, pd.DataFrame]:
        frame = pd.DataFrame(
            [
                {
                    "state_id": state.state_id,
                    "round_no": state.round_no,
                    "invested_amount": state.invested_amount,
                    "amount_error": state.amount_error,
                    "tracking_variance": state.tracking_variance,
                    "tracking_error_annual": state.tracking_error_annual,
                    "action_type": state.action.get("action_type"),
                }
                for state in frontier
            ]
        )
        for source, target in [
            ("amount_error", "amount_error_normalized"),
            ("tracking_error_annual", "tracking_error_normalized"),
        ]:
            values = frame[source].to_numpy(dtype=float)
            span = values.max() - values.min()
            frame[target] = (values - values.min()) / span if span > 0 else 0.0
        frame["ideal_point_distance"] = np.sqrt(
            np.square(frame["amount_error_normalized"])
            + np.square(frame["tracking_error_normalized"])
        )
        frame = frame.sort_values(
            ["ideal_point_distance", "tracking_error_annual", "amount_error", "state_id"]
        ).reset_index(drop=True)
        selected_id = str(frame.iloc[0]["state_id"])
        selected = next(state for state in frontier if state.state_id == selected_id)
        frame["is_selected"] = frame["state_id"].eq(selected_id)
        return selected, frame

    def run(self) -> dict:
        base = self.base_state
        self._discovered_state_ids = {base.state_id}
        beam = [base]
        archive = [base]
        previous_ids = {base.state_id}
        expanded_state_ids: set[str] = set()
        stale_rounds = 0
        round_rows = []

        for round_no in range(1, self.config.max_rounds + 1):
            input_beam_count = len(beam)
            generated = []
            aggregate = {
                "raw_candidate_count": 0,
                "unique_action_count": 0,
                "feasible_candidate_count": 0,
                "local_pareto_candidate_count": 0,
                "materialized_candidate_count": 0,
                "duplicate_state_count": 0,
            }
            for state in beam:
                if state.state_id in expanded_state_ids:
                    raise RuntimeError(f"State expanded more than once: {state.state_id}")
                expanded_state_ids.add(state.state_id)
                state_candidates, state_stats = self._generate_candidates(state, round_no)
                generated.extend(state_candidates)
                for key in aggregate:
                    aggregate[key] += int(state_stats[key])
            frontier = self._pareto_front(archive + generated + [base])
            frontier_ids = {state.state_id for state in frontier}
            new_count = len(frontier_ids - previous_ids)
            stale_rounds = stale_rounds + 1 if new_count == 0 else 0
            archive = frontier
            beam = self._select_beam(
                frontier,
                excluded_state_ids=expanded_state_ids,
            )
            previous_ids = frontier_ids

            round_rows.append({
                "round_no": round_no,
                "input_beam_count": input_beam_count,
                "expanded_state_count": input_beam_count,
                "cumulative_expanded_state_count": len(expanded_state_ids),
                "raw_candidate_count": aggregate["raw_candidate_count"],
                "unique_action_count": aggregate["unique_action_count"],
                "feasible_candidate_count": aggregate["feasible_candidate_count"],
                "local_pareto_candidate_count": aggregate["local_pareto_candidate_count"],
                "generated_candidate_count": len(generated),
                "materialized_candidate_count": len(generated),
                "duplicate_state_count": aggregate["duplicate_state_count"],
                "pareto_frontier_count": len(frontier),
                "new_pareto_state_count": new_count,
                "best_amount_error": min(state.amount_error for state in frontier),
                "best_tracking_error_annual": min(state.tracking_error_annual for state in frontier),
            })
            if round_no == 1 or round_no % 5 == 0:
                print(
                    f"[pareto {self.label}] round={round_no}, "
                    f"raw={aggregate['raw_candidate_count']:,}, "
                    f"local={aggregate['local_pareto_candidate_count']:,}, "
                    f"materialized={len(generated):,}, "
                    f"frontier={len(frontier)}, beam={len(beam)}, new={new_count}, "
                    f"best_amount={round_rows[-1]['best_amount_error']:.8f}, "
                    f"best_TE={round_rows[-1]['best_tracking_error_annual']:.6%}"
                )
            if stale_rounds >= self.config.stale_rounds_to_stop or not beam:
                break

        final_frontier = self._pareto_front(archive + [base])
        selected, frontier_frame = self._select_ideal_point(final_frontier)
        self._validate_state(selected)

        path = []
        cursor = selected
        while cursor is not None:
            if cursor.parent is not None:
                path.append({"round_no": cursor.round_no, "state_id": cursor.state_id, **cursor.action})
            cursor = cursor.parent
        path_frame = pd.DataFrame(list(reversed(path)))

        target = self.frame.copy()
        target["amount_greedy_qty"] = self.base_state.qty
        target["pareto_qty_delta"] = selected.qty - self.base_state.qty
        target["target_qty"] = selected.qty
        target["target_market_value"] = target["target_qty"] * target["close_price"]
        target["is_excepted_code"] = self.is_excepted
        target["is_held"] = target["target_qty"] > 0
        target["greedy_method"] = "amount_greedy_pareto_risk"

        summary = {
            "portfolio_label": self.label,
            "greedy_method": "amount_greedy_pareto_risk",
            "target_stock_value": self.target_value,
            "budget_limit": self.budget_limit,
            "base_invested_amount": self.base_state.invested_amount,
            "final_invested_amount": selected.invested_amount,
            "final_invested_ratio": selected.invested_amount / self.target_value,
            "base_amount_error": self.base_state.amount_error,
            "final_amount_error": selected.amount_error,
            "base_tracking_variance": self.base_state.tracking_variance,
            "final_tracking_variance": selected.tracking_variance,
            "base_tracking_error_annual": self.base_state.tracking_error_annual,
            "final_tracking_error_annual": selected.tracking_error_annual,
            "pareto_frontier_count": len(final_frontier),
            "pareto_round_count": len(round_rows),
            "selected_state_id": selected.state_id,
            "selected_path_length": len(path_frame),
            "pareto_expanded_state_count": len(expanded_state_ids),
            "pareto_raw_candidate_count": int(
                sum(row["raw_candidate_count"] for row in round_rows)
            ),
            "pareto_materialized_candidate_count": int(
                sum(row["generated_candidate_count"] for row in round_rows)
            ),
            "held_stock_count": int((selected.qty > 0).sum()),
            "zero_qty_stock_count": int((selected.qty == 0).sum()),
            "excepted_component_count": int(self.is_excepted.sum()),
        }
        return {
            "df_target_portfolio": target,
            "summary": summary,
            "base_state": self.base_state,
            "selected_state": selected,
            "pareto_frontier": frontier_frame,
            "selected_path": path_frame,
            "round_summary": pd.DataFrame(round_rows),
        }


def optimize_portfolio_pareto_risk(
    base_portfolio: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
    *,
    target_value: float,
    max_over_budget_ratio: float = 1.005,
    excepted_code_ls: list[str] | None = None,
    config: ParetoOptimizerConfig | None = None,
    label: str = "pareto_risk",
) -> dict:
    config = config or ParetoOptimizerConfig()
    optimizer = _ParetoPortfolioOptimizer(
        base_portfolio,
        covariance_matrix,
        target_value=target_value,
        max_over_budget_ratio=max_over_budget_ratio,
        excepted_code_ls=excepted_code_ls,
        config=config,
        label=label,
    )
    return optimizer.run()


def build_size_exposure_report(
    target_portfolio: pd.DataFrame,
    mcap_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build a report-only large/mid/small exposure table."""

    _require_columns(
        target_portfolio,
        ["stock_code", "raw_weight_pct", "target_market_value"],
        "target_portfolio",
    )
    _require_columns(mcap_frame, ["stock_code", "mcap"], "mcap_frame")

    portfolio = target_portfolio.copy()
    portfolio["stock_code"] = portfolio["stock_code"].map(_normalize_stock_code)
    mcap = mcap_frame.copy()
    mcap["stock_code"] = mcap["stock_code"].map(_normalize_stock_code)
    mcap["mcap"] = pd.to_numeric(mcap["mcap"], errors="coerce")
    mcap = mcap.dropna(subset=["mcap"]).drop_duplicates("stock_code")

    ranked = mcap.sort_values("mcap", ascending=False).reset_index(drop=True)
    count = len(ranked)
    split1 = int(math.ceil(count / 3))
    split2 = int(math.ceil(2 * count / 3))
    ranked["size_label"] = "small"
    ranked.loc[: split1 - 1, "size_label"] = "large"
    ranked.loc[split1 : split2 - 1, "size_label"] = "mid"

    merged = portfolio.merge(ranked[["stock_code", "mcap", "size_label"]], on="stock_code", how="left")
    merged["size_label"] = merged["size_label"].fillna("UNKNOWN")
    merged["index_weight"] = pd.to_numeric(merged["raw_weight_pct"], errors="coerce").fillna(0.0)
    merged["index_weight"] /= merged["index_weight"].sum()
    invested = float(pd.to_numeric(merged["target_market_value"], errors="coerce").fillna(0.0).sum())
    merged["portfolio_weight"] = merged["target_market_value"] / invested if invested > 0 else 0.0

    report = merged.groupby("size_label", as_index=False).agg(
        stock_count=("stock_code", "size"),
        index_size_weight=("index_weight", "sum"),
        portfolio_size_weight=("portfolio_weight", "sum"),
        mcap_min=("mcap", "min"),
        mcap_max=("mcap", "max"),
    )
    report["active_size_weight"] = report["portfolio_size_weight"] - report["index_size_weight"]
    return report.sort_values("active_size_weight", key=lambda values: values.abs(), ascending=False).reset_index(drop=True)
