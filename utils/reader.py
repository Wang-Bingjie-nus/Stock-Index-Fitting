import pandas as pd
import os
import pickle
from datetime import datetime, date

from .file_utils import get_tick_file_path, get_daybar_file_path


def parse_date(date_input):
    """
    解析日期输入，支持多种格式
    
    参数:
        date_input: 可以是 datetime对象、date对象或字符串
        
    返回:
        datetime: 解析后的datetime对象
    """
    if isinstance(date_input, datetime):
        return date_input
    elif isinstance(date_input, date):
        return datetime(date_input.year, date_input.month, date_input.day)
    elif isinstance(date_input, str):
        date_str = date_input.strip()
        # 支持的日期格式: YYYY-MM-DD, YYYYMMDD, YYYY/MM/DD
        for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"无法解析日期格式: {date_input}")
    else:
        raise TypeError(f"不支持的日期类型: {type(date_input)}")


def read_daily_data(date_input, base_dir=r'Z:\高频行情迅投\日行情'):
    """
    读取日行情数据文件
    
    参数:
        date_input: 日期，可以是 datetime/date 对象或字符串
        base_dir: 基础目录，默认为 Z:\高频行情迅投\日行情
    
    返回:
        pd.DataFrame 或 None (文件不存在时)
    """
    file_path = get_daybar_file_path(date_input, base_dir)
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return None
    
    print(f"成功读取日行情数据")
    print(f"  文件路径: {file_path}")
    print(f"  数据形状: {df.shape}")
    return df


def read_sector_ticks(date_input, sector=None, base_dir=r'Z:\高频行情迅投\ticks'):
    """
    读取指定板块的tick数据文件
    
    参数:
        date_input: 日期，可以是 datetime/date 对象或字符串
        sector: 板块类型，可选值: SHZB, SHKCB, SZZB, SZCYB, BJ（不区分大小写）
                如不传，则默认读取BJ（北交所）
        base_dir: 基础目录，默认为 Z:\高频行情迅投\ticks
    
    返回:
        dict 或 None (文件不存在时)
        返回 dict，键为股票代码，值为对应的 tick 数据 DataFrame
    """
    date_obj = parse_date(date_input)
    date_str = date_obj.strftime('%Y%m%d')
    year = date_obj.strftime('%Y')
    month = date_obj.strftime('%m')
    
    # 板块映射
    sector_map = {
        'SHZB': ('SH', 'sh_zb'),
        'SHKCB': ('SH', 'sh_kcb'),
        'SZZB': ('SZ', 'sz_zb'),
        'SZCYB': ('SZ', 'sz_cyb'),
        'BJ': ('BJ', 'bj')
    }
    
    # 处理板块参数
    if sector is None:
        sector = 'BJ'
    else:
        sector = sector.upper()
        if sector not in sector_map:
            valid_sectors = ', '.join(sector_map.keys())
            print(f"无效的板块: {sector}，可选值: {valid_sectors}")
            return None
    
    exchange, market_suffix = sector_map[sector]
    
    # 构建文件路径
    file_name = f"{date_str}_tick_{market_suffix}.pkl"
    file_path = os.path.join(base_dir, exchange, year, month, file_name)
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return None
    
    print(f"成功读取: {file_path}")
    print(f"板块: {sector}")
    print(f"股票数量: {len(data)}")
    
    return data


def read_stocks_ticks(date_input, stock_codes, base_dir=r'Z:\高频行情迅投\ticks'):
    """
    根据股票代码列表批量读取tick数据（优化版本）
    
    优化策略：
    1. 根据股票代码分类，识别它们属于哪个文件
    2. 每个文件只读取一次
    3. 从文件中提取所有需要的股票数据
    
    参数:
        date_input: 日期，可以是 datetime/date 对象或字符串
        stock_codes: 股票代码列表，如 ['600000.SH', '000001.SZ', '920001.BJ']
        base_dir: 基础目录，默认为 Z:\高频行情迅投\ticks
    
    返回:
        dict: {stock_code: pd.DataFrame}，键为股票代码，值为对应的tick数据DataFrame
              未找到的股票代码对应值为 None
    """
    if not stock_codes:
        print("股票代码列表为空")
        return {}
    
    # 将股票代码按所属文件分组
    file_stocks_map = {}
    
    for stock_code in stock_codes:
        try:
            file_path = get_tick_file_path(date_input, stock_code, base_dir)
            if file_path not in file_stocks_map:
                file_stocks_map[file_path] = []
            file_stocks_map[file_path].append(stock_code)
        except Exception as e:
            print(f"获取股票 {stock_code} 的文件路径失败: {e}")
    
    # 读取每个文件并提取所需数据
    result = {}
    total_read_files = 0
    
    for file_path, stocks_in_file in file_stocks_map.items():
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            for stock in stocks_in_file:
                result[stock] = None
            continue
        
        try:
            with open(file_path, 'rb') as f:
                file_data = pickle.load(f)
            total_read_files += 1
            
            for stock_code in stocks_in_file:
                if stock_code in file_data:
                    df = file_data[stock_code]
                    if isinstance(df, pd.DataFrame):
                        result[stock_code] = df
                        print(f"✓ {stock_code}: 读取成功")
                    else:
                        result[stock_code] = None
                        print(f"✗ {stock_code}: 数据格式错误，不是DataFrame类型")
                else:
                    result[stock_code] = None
                    print(f"✗ {stock_code}: 文件中未找到该股票")
        
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            for stock in stocks_in_file:
                result[stock] = None
    
    print(f"\n批量读取完成，共读取 {total_read_files} 个文件，{len(result)} 只股票")
    
    return result
