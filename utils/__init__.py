"""
NAS存储文件读取模块

提供从NAS存储读取日行情和tick数据的功能，独立于ggbase模块。

导出的函数：
-----------
read_daily_data(date_input, base_dir)
    读取指定日期的日行情数据
    
read_sector_ticks(date_input, sector, base_dir)
    读取指定板块的tick数据
    支持板块: SHZB, SHKCB, SZZB, SZCYB, BJ（不区分大小写）
    
read_stocks_ticks(date_input, stock_codes, base_dir)
    根据日期和股票代码列表批量读取tick数据（优化版本，每个文件只读一次）
    
get_tick_file_path(date_input, stock_code, base_dir)
    根据日期和股票代码获取tick文件路径
    
get_daybar_file_path(date_input, base_dir)
    根据日期获取日行情文件路径
    
parse_date(date_input)
    解析日期输入，支持多种格式
"""

from .reader import read_daily_data, read_sector_ticks, read_stocks_ticks, parse_date
from .file_utils import get_tick_file_path, get_daybar_file_path
# from .downloader import download_csi_constituent
from .csi_reader import read_csi_file
from .db_query import safe_query
from .quick_query import quick_query
from .dataset_layout import build_run_paths
from .sse_etf_pcf_scraper import SseEtfPcfScraper, print_preview, resolve_sse50_etf_codes, save_outputs
from .szse_etf_pcf import fetch_szse_pcf
from .exposure_deviation import calculate_exposure_deviation
# from .tick_analysis import build_index_tick, compute_tracking_error, plot_tracking, build_basket_ticks
from .downloader_v02 import download_csi_constituent_v02


__all__ = [
    'read_daily_data',
    'read_sector_ticks',
    'read_stocks_ticks',
    'get_tick_file_path',
    'get_daybar_file_path',
    'parse_date',
    'download_csi_constituent', 
    'read_csi_file',
    'safe_query', 
    'quick_query',
    'build_run_paths',
    'fetch_szse_pcf',
    'calculate_exposure_deviation',
    'build_index_tick',
    'compute_tracking_error',
    'plot_tracking',
    'build_basket_ticks'
]
