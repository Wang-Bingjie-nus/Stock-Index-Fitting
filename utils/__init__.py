"""Notebook-facing public API for the completed stock-index workflows."""

from .adjustment_schedule import (
    adjustment_interval_from_config,
    find_unavailable_marker_dates,
    first_trading_dates_of_later_months,
    resolve_adjusting_date_candidates,
    resolve_interval_adjusting_dates,
)
from .cache_generator import (
    CSI300_INDEX_CODE,
    generate_csi300_caches,
    get_previous_closest_trading_date,
    merge_partition_caches_for_range,
    preview_generation_plan,
    resolve_trading_dates,
    select_complete_trade_dates,
)
from .csi_reader import read_csi_file
from .db_query import safe_query as gogoal_query
from .downloader_v02 import download_csi_constituent_v02
from .exposure_deviation import calculate_exposure_deviation
from .file_utils import get_tick_file_path
from .greedy_score import (
    GreedyScoreConfig,
    build_exposure_matrices,
    compute_portfolio_state,
    compute_score,
    quantities_from_portfolio,
    score_to_frame,
)
from .limit_impact_pipeline_v01 import (
    LimitImpactDateRun,
    LimitImpactPipelineConfig,
    filter_available_date_runs,
    preflight_transition_daily_data,
)
from .limit_impact_v01 import combine_task11_summaries, expand_construction_date_list
from .minute_tick_cache_v03 import (
    build_minute_tick_cache,
    build_trading_minute_index,
    build_unavailable_minute_tick_cache,
    load_minute_tick_cache,
    load_unavailable_minute_tick_cache,
    save_minute_tick_cache,
    save_unavailable_minute_tick_cache,
    unavailable_minute_tick_cache_path,
)
from .pareto_risk_optimizer_v04 import (
    ParetoOptimizerConfig as ParetoOptimizerConfigV04,
    build_size_exposure_report as build_size_exposure_report_v04,
    optimize_portfolio_pareto_risk as optimize_portfolio_pareto_risk_v04,
)
from .reader import read_daily_data, read_stocks_ticks
from .risk_model_v03 import build_shrunk_risk_model_from_daily_loader
from .tick_analysis_v03 import (
    build_minute_tracking_analysis as build_minute_tracking_analysis_v03,
    save_tracking_outputs as save_tracking_outputs_v03,
)
from .tick_analysis_v10 import (
    build_minute_tracking_analysis as build_minute_tracking_analysis_v10,
    combine_minute_tracking_results,
    merge_corporate_action_sources,
    save_tracking_outputs as save_tracking_outputs_v10,
    standardize_corporate_actions,
)
from .weight_projection import (
    fetch_stock_closes_range,
    load_or_download_weight_source,
    make_projection_output_dir,
    normalize_date_key,
    project_weights_by_close_and_actions,
    save_projection_outputs,
    select_weight_source_for_target,
)

__all__ = [
    "CSI300_INDEX_CODE",
    "GreedyScoreConfig",
    "LimitImpactDateRun",
    "LimitImpactPipelineConfig",
    "ParetoOptimizerConfigV04",
    "adjustment_interval_from_config",
    "build_exposure_matrices",
    "build_minute_tick_cache",
    "build_minute_tracking_analysis_v03",
    "build_minute_tracking_analysis_v10",
    "build_shrunk_risk_model_from_daily_loader",
    "build_size_exposure_report_v04",
    "build_trading_minute_index",
    "build_unavailable_minute_tick_cache",
    "calculate_exposure_deviation",
    "combine_minute_tracking_results",
    "combine_task11_summaries",
    "compute_portfolio_state",
    "compute_score",
    "download_csi_constituent_v02",
    "expand_construction_date_list",
    "fetch_stock_closes_range",
    "filter_available_date_runs",
    "find_unavailable_marker_dates",
    "first_trading_dates_of_later_months",
    "generate_csi300_caches",
    "get_previous_closest_trading_date",
    "get_tick_file_path",
    "gogoal_query",
    "load_minute_tick_cache",
    "load_or_download_weight_source",
    "load_unavailable_minute_tick_cache",
    "make_projection_output_dir",
    "merge_corporate_action_sources",
    "merge_partition_caches_for_range",
    "normalize_date_key",
    "optimize_portfolio_pareto_risk_v04",
    "preview_generation_plan",
    "preflight_transition_daily_data",
    "project_weights_by_close_and_actions",
    "quantities_from_portfolio",
    "read_csi_file",
    "read_daily_data",
    "read_stocks_ticks",
    "resolve_adjusting_date_candidates",
    "resolve_interval_adjusting_dates",
    "resolve_trading_dates",
    "save_minute_tick_cache",
    "save_projection_outputs",
    "save_tracking_outputs_v03",
    "save_tracking_outputs_v10",
    "save_unavailable_minute_tick_cache",
    "score_to_frame",
    "select_complete_trade_dates",
    "select_weight_source_for_target",
    "standardize_corporate_actions",
    "unavailable_minute_tick_cache_path",
]
