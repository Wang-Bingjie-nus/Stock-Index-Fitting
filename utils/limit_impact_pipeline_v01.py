from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .limit_impact_v01 import (
    classify_unavailable_stocks,
    evaluate_baskets_at_0931,
    extract_opening_minute_snapshot,
    fetch_xt_historical_trading_status,
    get_next_trading_date,
    get_recent_trading_dates,
    unavailable_codes_from_report,
)
from .minute_cache_loader_v03 import load_stock_minute_cache_cached
from .minute_tick_cache_v03 import (
    load_unavailable_minute_tick_cache,
    normalize_stock_code,
    normalize_trade_date,
    unavailable_minute_tick_cache_path,
)
from .portfolio_construction_v04 import (
    ParetoOptimizerConfig,
    build_pareto_portfolio_for_weights,
    prepare_theoretical_portfolio,
    require_columns,
)
from .risk_model_v03 import build_shrunk_risk_model_from_daily_loader


@dataclass(frozen=True)
class LimitImpactPipelineConfig:
    index_code: str
    index_name: str
    xt_index_code: str
    target_stock_value: float = 4_500_000.0
    rule_file_path: str = "security_buy_rules.csv"
    manual_unavailable_codes: tuple[str, ...] = field(default_factory=tuple)
    risk_matrix_mode: str = "correlation"
    risk_lookback_days: int = 5
    risk_half_life_days: float = 3.0
    risk_price_col: str = "lastPrice"
    pareto_risk_candidate_count: int = 10
    pareto_amount_candidate_count: int = 10
    pareto_beam_width: int = 20
    pareto_max_rounds: int = 50
    pareto_stale_rounds_to_stop: int = 3
    pareto_legal_neighbor_steps: int = 3
    allow_over_budget: bool = True
    max_over_budget_ratio: float = 1.005
    source_tick_root: Path = Path(".")
    price_validation_tolerance: float = 0.0001
    limit_price_tolerance: float = 1e-6
    include_suspended_as_unavailable: bool = True

    def __post_init__(self) -> None:
        if self.risk_matrix_mode not in {"correlation", "covariance"}:
            raise ValueError("risk_matrix_mode must be 'correlation' or 'covariance'.")
        if float(self.target_stock_value) <= 0:
            raise ValueError("target_stock_value must be positive.")
        if int(self.risk_lookback_days) <= 0:
            raise ValueError("risk_lookback_days must be positive.")
        if float(self.max_over_budget_ratio) < 1.0:
            raise ValueError("max_over_budget_ratio cannot be below 1.0.")


def filter_available_date_runs(
    date_runs: Iterable[Any],
    unavailable_dates: Iterable[str],
) -> list[Any]:
    """Drop every D->D+1 interval touching an unavailable trading date."""

    unavailable = {
        normalize_trade_date(trade_date) for trade_date in unavailable_dates
    }
    return [
        run
        for run in date_runs
        if normalize_trade_date(run.construction_date) not in unavailable
        and normalize_trade_date(run.evaluation_date) not in unavailable
    ]


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _sql_in(values: Iterable[Any]) -> str:
    values = list(values)
    if not values:
        raise ValueError("SQL IN values cannot be empty.")
    return ",".join(_sql_literal(value) for value in values)


def _strip_market_suffix(code: Any) -> str:
    return str(code).split(".")[0].zfill(6)


def fetch_gogoal_index_close(
    gogoal_query_fn: Callable[..., pd.DataFrame],
    index_code: str,
    trade_date: str,
) -> pd.DataFrame:
    trade_date_dash = pd.Timestamp(normalize_trade_date(trade_date)).strftime(
        "%Y-%m-%d"
    )
    return gogoal_query_fn(
        f"""
        SELECT trade_date, index_code, index_name, tclose, lclose, is_valid
        FROM qt_idx_daily
        WHERE index_code = {_sql_literal(index_code)}
          AND trade_date = {_sql_literal(trade_date_dash)}
          AND is_valid = 1
        """,
        output_format="dataframe",
    )


def fetch_gogoal_stock_closes(
    gogoal_query_fn: Callable[..., pd.DataFrame],
    stock_codes: Iterable[str],
    trade_date: str,
) -> pd.DataFrame:
    stripped = [_strip_market_suffix(code) for code in stock_codes]
    trade_date_dash = pd.Timestamp(normalize_trade_date(trade_date)).strftime(
        "%Y-%m-%d"
    )
    raw = gogoal_query_fn(
        f"""
        SELECT trade_date, stock_code, stock_name, tclose, lclose, is_valid
        FROM qt_stk_daily
        WHERE trade_date = {_sql_literal(trade_date_dash)}
          AND stock_code IN ({_sql_in(stripped)})
          AND is_valid = 1
        """,
        output_format="dataframe",
    )
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "stock_code",
                "stock_name_gogoal",
                "gogoal_close",
                "gogoal_lclose",
            ]
        )
    raw = raw.copy()
    raw["stock_code"] = raw["stock_code"].map(normalize_stock_code)
    return raw.rename(
        columns={
            "stock_name": "stock_name_gogoal",
            "tclose": "gogoal_close",
            "lclose": "gogoal_lclose",
        }
    )[["stock_code", "stock_name_gogoal", "gogoal_close", "gogoal_lclose"]]


def read_projected_index_weights(
    project_root: Path,
    index_code: str,
    trade_date: str,
) -> pd.DataFrame:
    trade_date = normalize_trade_date(trade_date)
    path = (
        Path(project_root)
        / "data"
        / "weights_projection"
        / f"{index_code}-{trade_date}.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Projected index weight file is absent for {trade_date}: {path}"
        )
    raw = pd.read_csv(path)
    require_columns(
        raw,
        ["stock_code", "stock_name", "projected_weight_pct"],
        f"projected weights {path}",
    )
    weights = pd.DataFrame(
        {
            "stock_code": raw["stock_code"].map(normalize_stock_code),
            "stock_name": raw["stock_name"].astype(str).str.strip(),
            "raw_weight_pct": pd.to_numeric(
                raw["projected_weight_pct"], errors="coerce"
            ),
            "weight_date": trade_date,
            "weight_source": "PROJECTED_LOCAL",
        }
    )
    weights = (
        weights.dropna(subset=["stock_code", "raw_weight_pct"])
        .drop_duplicates("stock_code")
        .sort_values("stock_code")
        .reset_index(drop=True)
    )
    if weights.empty or (weights["raw_weight_pct"] <= 0).any():
        raise RuntimeError(
            f"Projected weights contain no valid positive weights: {path}"
        )
    return weights


class LimitImpactDateRun:
    """Stateful execution of Tasks 3-12 for one construction date."""

    def __init__(
        self,
        *,
        construction_date: str,
        config: LimitImpactPipelineConfig,
        project_root: Path,
        batch_run_dir: Path,
        import_time: str,
        xtdata_client: Any,
        gogoal_query_fn: Callable[..., pd.DataFrame],
        daily_loader: Callable[[str], pd.DataFrame],
    ) -> None:
        self.construction_date = normalize_trade_date(construction_date)
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.batch_run_dir = Path(batch_run_dir).resolve()
        self.import_time = str(import_time)
        self.xtdata = xtdata_client
        self.gogoal_query = gogoal_query_fn
        self.daily_loader = daily_loader
        self.manual_unavailable_codes = sorted(
            {normalize_stock_code(code) for code in config.manual_unavailable_codes}
        )

        calendar = self.xtdata.get_trading_calendar(
            "SH",
            start_time=self.construction_date,
            end_time=self.construction_date,
        )
        calendar_dates = {
            re.sub(r"\D", "", str(value))[:8] for value in (calendar or [])
        }
        if self.construction_date not in calendar_dates:
            raise ValueError(
                "construction_date is not an SH trading date: "
                f"{self.construction_date}"
            )

        self.evaluation_date = get_next_trading_date(
            self.xtdata, self.construction_date
        )
        self.run_dir = self.batch_run_dir / "date_runs" / self.construction_date
        self.dirs = {
            "inputs": self.run_dir / "01_inputs",
            "status": self.run_dir / "02_d1_status",
            "baskets": self.run_dir / "03_baskets",
            "reports": self.run_dir / "04_reports",
        }
        for directory in self.dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        self.tick_cache_dir = (
            self.project_root
            / "data"
            / self.config.index_code
            / "_tick_cache_correlation_v03"
        )

    @property
    def transition_dates(self) -> tuple[str, str]:
        return self.construction_date, self.evaluation_date

    def existing_unavailable_transition_dates(self) -> set[str]:
        """Return D/D+1 dates that already carry any valid unavailable marker."""

        unavailable_dates = set()
        for trade_date in self.transition_dates:
            marker_path = unavailable_minute_tick_cache_path(
                self.tick_cache_dir,
                trade_date,
            )
            if not marker_path.is_file():
                continue
            marker = load_unavailable_minute_tick_cache(
                marker_path,
                trade_date=trade_date,
            )
            metadata = marker["metadata"]
            print(
                f"[UNAVAILABLE CACHE HIT] transition preflight {trade_date}: "
                f"reason={metadata['reason']}; "
                f"details={metadata.get('details', {})}"
            )
            unavailable_dates.add(trade_date)
        return unavailable_dates

    def preflight_transition_minute_caches(self) -> set[str]:
        """Build or validate D/D+1 caches before expensive portfolio work."""

        for role, trade_date in zip(
            ("construction", "evaluation"),
            self.transition_dates,
        ):
            minute_cache = load_stock_minute_cache_cached(
                trade_date,
                self.stock_codes,
                index_code=self.config.index_code,
                cache_dir=self.tick_cache_dir,
                source_tick_root=Path(self.config.source_tick_root),
                label=(
                    f"transition preflight {role} "
                    f"{self.construction_date} {self.evaluation_date}"
                ),
                source_missing_policy="skip",
            )
            if minute_cache is None:
                return {trade_date}
            del minute_cache
        return set()

    def load_inputs(self) -> None:
        self.df_index_weights = read_projected_index_weights(
            self.project_root,
            self.config.index_code,
            self.construction_date,
        )
        self.stock_codes = self.df_index_weights["stock_code"].tolist()

        rule_path = self.project_root / "dataloader" / self.config.rule_file_path
        if not rule_path.is_file():
            raise FileNotFoundError(f"Trading rule file is absent: {rule_path}")
        self.df_security_rules = pd.read_csv(rule_path)
        require_columns(
            self.df_security_rules,
            ["exchange", "board", "buy_min_qty", "buy_qty_step"],
            "security rules",
        )
        for column in ["exchange", "board"]:
            self.df_security_rules[column] = (
                self.df_security_rules[column].astype(str).str.strip().str.upper()
            )
        for column in ["buy_min_qty", "buy_qty_step"]:
            self.df_security_rules[column] = pd.to_numeric(
                self.df_security_rules[column], errors="raise"
            ).astype(int)

        if self.df_index_weights["stock_code"].duplicated().any():
            raise RuntimeError("Index weights contain duplicate stocks.")
        if self.df_security_rules.duplicated(["exchange", "board"]).any():
            raise RuntimeError(
                "Trading rules contain duplicate exchange/board combinations."
            )

        self.df_index_weights.to_csv(
            self.dirs["inputs"] / "index_weights.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.df_security_rules.to_csv(
            self.dirs["inputs"] / "security_rules.csv",
            index=False,
            encoding="utf-8-sig",
        )

    def load_construction_prices(self) -> None:
        df_gogoal_index = fetch_gogoal_index_close(
            self.gogoal_query,
            self.config.index_code,
            self.construction_date,
        )
        if df_gogoal_index.empty:
            raise RuntimeError(
                "Go-Goal index daily data is empty: "
                f"{self.config.index_code} {self.construction_date}"
            )
        self.construction_index_close = float(
            pd.to_numeric(df_gogoal_index["tclose"], errors="coerce").dropna().iloc[-1]
        )

        self.xtdata.download_history_data(
            self.config.xt_index_code,
            period="1d",
            start_time=self.construction_date,
            end_time=self.construction_date,
        )
        xt_index_daily = self.xtdata.get_market_data_ex(
            field_list=["close"],
            stock_list=[self.config.xt_index_code],
            period="1d",
            start_time=self.construction_date,
            end_time=self.construction_date,
            count=-1,
            dividend_type="none",
            fill_data=False,
        )
        xt_index_frame = xt_index_daily.get(self.config.xt_index_code)
        if xt_index_frame is None or xt_index_frame.empty:
            raise RuntimeError("XtQuant index daily data is empty.")
        xt_index_close = float(
            pd.to_numeric(xt_index_frame["close"], errors="coerce").dropna().iloc[-1]
        )
        index_close_deviation = abs(
            self.construction_index_close - xt_index_close
        ) / abs(self.construction_index_close)
        if index_close_deviation > self.config.price_validation_tolerance:
            raise RuntimeError(
                "Index close cross-validation failed: "
                f"gogoal={self.construction_index_close}, xt={xt_index_close}, "
                f"deviation={index_close_deviation}"
            )

        df_gogoal_close = fetch_gogoal_stock_closes(
            self.gogoal_query,
            self.stock_codes,
            self.construction_date,
        )
        if df_gogoal_close.empty:
            raise RuntimeError("Go-Goal component closes are empty.")
        df_gogoal_close["gogoal_close"] = pd.to_numeric(
            df_gogoal_close["gogoal_close"], errors="coerce"
        )

        nas_daily = self.daily_loader(self.construction_date)
        if nas_daily is None or nas_daily.empty:
            raise RuntimeError("NAS component daily data is empty.")
        require_columns(nas_daily, ["xt_stock_code", "tclose"], "NAS daily data")
        df_nas_close = nas_daily.loc[
            nas_daily["xt_stock_code"].isin(self.stock_codes),
            ["xt_stock_code", "tclose"],
        ].copy()
        df_nas_close.columns = ["stock_code", "nas_close"]
        df_nas_close["nas_close"] = pd.to_numeric(
            df_nas_close["nas_close"], errors="coerce"
        )

        self.df_market_snapshot = (
            pd.DataFrame({"stock_code": self.stock_codes})
            .merge(df_gogoal_close, on="stock_code", how="left")
            .merge(df_nas_close, on="stock_code", how="left")
        )
        self.df_market_snapshot["relative_deviation"] = (
            self.df_market_snapshot["gogoal_close"]
            - self.df_market_snapshot["nas_close"]
        ).abs() / self.df_market_snapshot["gogoal_close"].abs()
        self.df_market_snapshot["close_price"] = self.df_market_snapshot["gogoal_close"]
        self.df_market_snapshot["pricing_date"] = self.construction_date

        invalid_market = (
            self.df_market_snapshot[["gogoal_close", "nas_close"]].isna().any(axis=1)
            | (self.df_market_snapshot[["gogoal_close", "nas_close"]] <= 0).any(axis=1)
            | (
                self.df_market_snapshot["relative_deviation"]
                > self.config.price_validation_tolerance
            )
        )
        if invalid_market.any():
            bad = self.df_market_snapshot.loc[
                invalid_market,
                [
                    "stock_code",
                    "gogoal_close",
                    "nas_close",
                    "relative_deviation",
                ],
            ]
            raise RuntimeError(
                "Construction-date component close validation failed: "
                f"{bad.head(20).to_dict('records')}"
            )

        self.df_market_snapshot.to_csv(
            self.dirs["inputs"] / "market_snapshot_d.csv",
            index=False,
            encoding="utf-8-sig",
        )

    def build_risk_model(self) -> None:
        (
            self.df_component_rules,
            self.df_theoretical_portfolio,
        ) = prepare_theoretical_portfolio(
            self.df_index_weights,
            self.df_market_snapshot,
            self.df_security_rules,
            pricing_date=self.construction_date,
            target_value=self.config.target_stock_value,
            label="initial",
            import_time=self.import_time,
        )

        self.risk_dates = get_recent_trading_dates(
            self.xtdata,
            self.construction_date,
            self.config.risk_lookback_days,
        )
        risk_cache_by_date = {}
        self.skipped_risk_dates = []
        for risk_date in self.risk_dates:
            cache = load_stock_minute_cache_cached(
                risk_date,
                self.stock_codes,
                index_code=self.config.index_code,
                cache_dir=self.tick_cache_dir,
                source_tick_root=Path(self.config.source_tick_root),
                label=f"risk cache {risk_date}",
                source_missing_policy="skip",
            )
            if cache is None:
                self.skipped_risk_dates.append(risk_date)
            else:
                risk_cache_by_date[risk_date] = cache

        self.used_risk_dates = list(risk_cache_by_date)
        if not self.used_risk_dates:
            raise RuntimeError(
                "Risk model has no usable dates: "
                f"requested={self.risk_dates}, "
                f"skipped={self.skipped_risk_dates}"
            )

        self.risk_model = build_shrunk_risk_model_from_daily_loader(
            self.used_risk_dates,
            self.stock_codes,
            lambda risk_date: risk_cache_by_date.pop(str(risk_date)),
            price_col=self.config.risk_price_col,
            half_life_days=self.config.risk_half_life_days,
            min_valid_dates=1,
        )
        self.risk_model.summary["requested_risk_dates"] = self.risk_dates
        self.risk_model.summary["used_risk_dates"] = self.used_risk_dates
        self.risk_model.summary["skipped_risk_dates"] = self.skipped_risk_dates
        self.risk_matrix = getattr(self.risk_model, self.config.risk_matrix_mode).copy()

        self.df_component_rules.to_csv(
            self.dirs["inputs"] / "component_rules.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.df_theoretical_portfolio.to_csv(
            self.dirs["inputs"] / "theoretical_portfolio.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.risk_matrix.reset_index().rename(columns={"index": "stock_code"}).to_csv(
            self.dirs["inputs"] / f"risk_{self.config.risk_matrix_mode}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    def build_basket1(self) -> None:
        self.pareto_config = ParetoOptimizerConfig(
            risk_candidate_count=self.config.pareto_risk_candidate_count,
            amount_candidate_count=self.config.pareto_amount_candidate_count,
            beam_width=self.config.pareto_beam_width,
            max_rounds=self.config.pareto_max_rounds,
            stale_rounds_to_stop=self.config.pareto_stale_rounds_to_stop,
            legal_neighbor_steps=self.config.pareto_legal_neighbor_steps,
        )
        self.basket1_build = build_pareto_portfolio_for_weights(
            self.df_index_weights,
            self.construction_date,
            self.config.target_stock_value,
            self.df_security_rules,
            covariance_matrix=self.risk_matrix,
            pareto_config=self.pareto_config,
            allow_over_budget=self.config.allow_over_budget,
            max_over_budget_ratio=self.config.max_over_budget_ratio,
            label="basket1",
            market_snapshot=self.df_market_snapshot,
            theoretical_portfolio=self.df_theoretical_portfolio,
            excepted_code_ls=[],
        )
        self.basket1 = self.basket1_build["df_target_portfolio"].copy()
        self.basket1["portfolio_label"] = "basket1"
        self.basket1_invested_amount = float(self.basket1["target_market_value"].sum())
        if not (
            self.config.target_stock_value - 1e-8
            <= self.basket1_invested_amount
            <= self.config.target_stock_value * self.config.max_over_budget_ratio + 1e-8
        ):
            raise RuntimeError(
                f"basket1 invested amount is invalid: {self.basket1_invested_amount}"
            )
        self._write_basket("basket1", self.basket1)

    def load_d1_status(self) -> bool:
        d1_minute_cache = load_stock_minute_cache_cached(
            self.evaluation_date,
            self.stock_codes,
            index_code=self.config.index_code,
            cache_dir=self.tick_cache_dir,
            source_tick_root=Path(self.config.source_tick_root),
            label=f"D+1 opening cache {self.construction_date}",
            source_missing_policy="skip",
        )
        if d1_minute_cache is None:
            return False
        self.df_opening_snapshot = extract_opening_minute_snapshot(
            d1_minute_cache,
            self.stock_codes,
            self.evaluation_date,
        )
        self.df_xt_status_raw = fetch_xt_historical_trading_status(
            self.xtdata,
            self.stock_codes,
            self.evaluation_date,
        )
        self.df_trading_status = classify_unavailable_stocks(
            self.df_xt_status_raw,
            self.df_opening_snapshot,
            manual_unavailable_codes=self.manual_unavailable_codes,
            price_tolerance=self.config.limit_price_tolerance,
            include_suspended=self.config.include_suspended_as_unavailable,
        )
        self.status_unavailable_codes = unavailable_codes_from_report(
            self.df_trading_status
        )
        self.all_unavailable_codes = sorted(
            set(self.status_unavailable_codes) | set(self.manual_unavailable_codes)
        )
        self.manual_not_in_universe = sorted(
            set(self.manual_unavailable_codes) - set(self.stock_codes)
        )

        self.df_xt_status_raw.to_csv(
            self.dirs["status"] / "xt_status_raw_d1.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.df_opening_snapshot.to_csv(
            self.dirs["status"] / "opening_snapshot_0931.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.df_trading_status.to_csv(
            self.dirs["status"] / "trading_status_0931.csv",
            index=False,
            encoding="utf-8-sig",
        )
        self.status_summary = (
            self.df_trading_status.groupby("unavailable_reason", dropna=False)
            .size()
            .rename("stock_count")
            .reset_index()
            .sort_values(
                ["stock_count", "unavailable_reason"],
                ascending=[False, True],
            )
        )
        return True

    def build_basket3(self) -> None:
        self.basket3_build = build_pareto_portfolio_for_weights(
            self.df_index_weights,
            self.construction_date,
            self.config.target_stock_value,
            self.df_security_rules,
            covariance_matrix=self.risk_matrix,
            pareto_config=self.pareto_config,
            allow_over_budget=self.config.allow_over_budget,
            max_over_budget_ratio=self.config.max_over_budget_ratio,
            label="basket3",
            market_snapshot=self.df_market_snapshot,
            theoretical_portfolio=self.df_theoretical_portfolio,
            excepted_code_ls=self.all_unavailable_codes,
        )
        self.basket3 = self.basket3_build["df_target_portfolio"].copy()
        self.basket3["portfolio_label"] = "basket3"
        unexpected = self.basket3.loc[
            self.basket3["stock_code"].isin(self.all_unavailable_codes)
            & (pd.to_numeric(self.basket3["target_qty"], errors="coerce").fillna(0) > 0)
        ]
        if not unexpected.empty:
            raise RuntimeError(
                "basket3 still holds unavailable stocks: "
                f"{unexpected['stock_code'].tolist()[:20]}"
            )
        self._write_basket("basket3", self.basket3)

    def load_index_opening_price(self) -> None:
        self.xtdata.download_history_data(
            self.config.xt_index_code,
            period="1m",
            start_time=self.evaluation_date,
            end_time=self.evaluation_date,
        )
        xt_index_minute = self.xtdata.get_market_data_ex(
            field_list=["close"],
            stock_list=[self.config.xt_index_code],
            period="1m",
            start_time=self.evaluation_date,
            end_time=self.evaluation_date,
            count=-1,
            dividend_type="none",
            fill_data=False,
        )
        df_index_minute_d1 = xt_index_minute.get(self.config.xt_index_code)
        if df_index_minute_d1 is None or df_index_minute_d1.empty:
            raise RuntimeError(
                "XtQuant index 1m data is empty: "
                f"{self.config.xt_index_code} {self.evaluation_date}"
            )
        index_minute_time = pd.to_datetime(df_index_minute_d1.index)
        opening_timestamp = pd.Timestamp(self.evaluation_date).replace(
            hour=9, minute=31
        )
        opening_rows = df_index_minute_d1.loc[
            index_minute_time.floor("min") == opening_timestamp
        ]
        if opening_rows.empty:
            raise RuntimeError(
                "Index has no 09:31 close: "
                f"{self.config.xt_index_code} {self.evaluation_date}"
            )
        self.opening_index_close = float(
            pd.to_numeric(opening_rows["close"], errors="coerce").dropna().iloc[-1]
        )
        if not np.isfinite(self.opening_index_close) or self.opening_index_close <= 0:
            raise RuntimeError(
                f"Index 09:31 close is invalid: {self.opening_index_close}"
            )

    def evaluate(self) -> pd.DataFrame:
        (
            self.df_deviation_summary,
            self.basket_valuation_details,
        ) = evaluate_baskets_at_0931(
            {
                "basket1": self.basket1,
                "basket3": self.basket3,
            },
            self.df_opening_snapshot,
            construction_index_close=self.construction_index_close,
            opening_index_close=self.opening_index_close,
            common_base_amount=self.basket1_invested_amount,
        )
        self.df_deviation_summary.insert(1, "construction_date", self.construction_date)
        self.df_deviation_summary.insert(2, "evaluation_date", self.evaluation_date)
        self.df_deviation_summary.to_csv(
            self.dirs["reports"] / "two_basket_deviation_0931.csv",
            index=False,
            encoding="utf-8-sig",
        )

        status_columns = [
            "stock_code",
            "is_limit_up_0931",
            "is_limit_down_0931",
            "is_suspended",
            "is_manual_unavailable",
            "is_unavailable",
            "unavailable_reason",
        ]
        for basket_name, detail in self.basket_valuation_details.items():
            detail = detail.merge(
                self.df_trading_status[status_columns],
                on="stock_code",
                how="left",
                validate="one_to_one",
            )
            self.basket_valuation_details[basket_name] = detail
            detail.to_csv(
                self.dirs["reports"] / f"{basket_name}_valuation_0931.csv",
                index=False,
                encoding="utf-8-sig",
            )
        return self.df_deviation_summary.copy()

    def finalize(self) -> dict[str, Any]:
        assert set(self.df_deviation_summary["basket"]) == {
            "basket1",
            "basket3",
        }
        assert len(self.df_trading_status) == len(self.df_index_weights)
        assert set(self.df_trading_status["stock_code"]) == set(
            self.df_index_weights["stock_code"]
        )
        for basket in [self.basket1, self.basket3]:
            assert not basket.loc[basket["target_qty"] > 0].empty
        assert not (
            set(self.basket3.loc[self.basket3["target_qty"] > 0, "stock_code"])
            & set(self.all_unavailable_codes)
        )

        self.run_manifest = {
            "index_code": self.config.index_code,
            "index_name": self.config.index_name,
            "construction_date": self.construction_date,
            "evaluation_date": self.evaluation_date,
            "target_stock_value": self.config.target_stock_value,
            "basket1_invested_amount": self.basket1_invested_amount,
            "manual_unavailable_codes": self.manual_unavailable_codes,
            "manual_not_in_universe": self.manual_not_in_universe,
            "status_unavailable_codes": self.status_unavailable_codes,
            "all_unavailable_codes": self.all_unavailable_codes,
            "risk_matrix_mode": self.config.risk_matrix_mode,
            "requested_risk_dates": self.risk_dates,
            "used_risk_dates": self.used_risk_dates,
            "skipped_risk_dates": self.skipped_risk_dates,
            "status_source": (
                "XtQuant meta 9506 + 1d suspendFlag/absent daily row "
                "+ 09:31 minute close"
            ),
            "stock_opening_price_source": (
                "NAS XtQuant raw tick -> basket_tick_v03 minute cache"
            ),
            "index_opening_price_source": "XtQuant 1m close",
            "run_dir": str(self.run_dir),
        }
        with (self.dirs["reports"] / "run_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                self.run_manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        return dict(self.run_manifest)

    def run_all(self) -> pd.DataFrame:
        self.load_inputs()
        self.load_construction_prices()
        self.build_risk_model()
        self.build_basket1()
        self.load_d1_status()
        self.build_basket3()
        self.load_index_opening_price()
        summary = self.evaluate()
        self.finalize()
        return summary

    def _write_basket(self, basket_name: str, basket: pd.DataFrame) -> None:
        basket.to_csv(
            self.dirs["baskets"] / f"{basket_name}_full.csv",
            index=False,
            encoding="utf-8-sig",
        )
        basket.loc[basket["target_qty"] > 0].to_csv(
            self.dirs["baskets"] / f"{basket_name}_holdings.csv",
            index=False,
            encoding="utf-8-sig",
        )


__all__ = [
    "LimitImpactDateRun",
    "LimitImpactPipelineConfig",
    "filter_available_date_runs",
    "fetch_gogoal_index_close",
    "fetch_gogoal_stock_closes",
    "read_projected_index_weights",
]
