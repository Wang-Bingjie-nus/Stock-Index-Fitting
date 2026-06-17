import os
from datetime import datetime, date


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
        for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"无法解析日期格式: {date_input}")
    else:
        raise TypeError(f"不支持的日期类型: {type(date_input)}")


def get_tick_file_path(date_input, stock_code, base_dir=r'Z:\高频行情迅投\ticks'):
    """
    根据日期和股票代码获取tick文件路径
    
    参数:
        date_input: 日期，可以是 datetime/date 对象或字符串
        stock_code: 股票代码，如 '600000.SH'
        base_dir: 基础目录，默认为 Z:\高频行情迅投\ticks
    
    返回:
        str: 文件路径
    """
    date_obj = parse_date(date_input)
    date_str = date_obj.strftime('%Y%m%d')
    year = date_obj.strftime('%Y')
    month = date_obj.strftime('%m')
    
    stock_code = stock_code.upper()
    
    if stock_code.endswith('.SH'):
        pure_code = stock_code.replace('.SH', '')
        if pure_code.startswith('68'):
            filename = f"{date_str}_tick_sh_kcb.pkl"
            save_dir = os.path.join(base_dir, 'SH', year, month)
        else:
            filename = f"{date_str}_tick_sh_zb.pkl"
            save_dir = os.path.join(base_dir, 'SH', year, month)
        return os.path.join(save_dir, filename)
    
    elif stock_code.endswith('.SZ'):
        pure_code = stock_code.replace('.SZ', '')
        if pure_code.startswith('30'):
            filename = f"{date_str}_tick_sz_cyb.pkl"
            save_dir = os.path.join(base_dir, 'SZ', year, month)
        else:
            filename = f"{date_str}_tick_sz_zb.pkl"
            save_dir = os.path.join(base_dir, 'SZ', year, month)
        return os.path.join(save_dir, filename)
    
    elif stock_code.endswith('.BJ'):
        filename = f"{date_str}_tick_bj.pkl"
        save_dir = os.path.join(base_dir, 'BJ', year, month)
        return os.path.join(save_dir, filename)
    
    elif stock_code.isdigit():
        if stock_code.startswith('92'):
            filename = f"{date_str}_tick_bj.pkl"
            save_dir = os.path.join(base_dir, 'BJ', year, month)
        elif stock_code.startswith('68'):
            filename = f"{date_str}_tick_sh_kcb.pkl"
            save_dir = os.path.join(base_dir, 'SH', year, month)
        elif stock_code.startswith('6'):
            filename = f"{date_str}_tick_sh_zb.pkl"
            save_dir = os.path.join(base_dir, 'SH', year, month)
        elif stock_code.startswith('30'):
            filename = f"{date_str}_tick_sz_cyb.pkl"
            save_dir = os.path.join(base_dir, 'SZ', year, month)
        else:
            filename = f"{date_str}_tick_sz_zb.pkl"
            save_dir = os.path.join(base_dir, 'SZ', year, month)
        return os.path.join(save_dir, filename)
    
    else:
        raise ValueError(f"无法识别的股票代码格式: {stock_code}")


def get_daybar_file_path(date_input, base_dir=r'Z:\高频行情迅投\日行情'):
    """
    根据日期获取日行情文件路径
    
    参数:
        date_input: 日期，可以是 datetime/date 对象或字符串
        base_dir: 基础目录，默认为 Z:\高频行情迅投\日行情
    
    返回:
        str: 文件路径
    """
    date_obj = parse_date(date_input)
    year = date_obj.year
    month = date_obj.month
    file_name = f"A_DAYBAR_{date_obj.strftime('%Y%m%d')}.pkl"
    return os.path.join(base_dir, str(year), f"{month:02d}", file_name)
