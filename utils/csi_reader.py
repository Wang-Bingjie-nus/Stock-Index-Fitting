"""读取模块 - 读取指数Excel文件并返回DataFrame"""

import pandas as pd
import os

def read_csi_file(file_path: str):
    """
    读取指数Excel文件并返回DataFrame，对部分字段进行格式转换
    
    :param file_path: Excel文件路径
    :return: 转换后的DataFrame
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print("目录错了")
        return None
    
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print(df.head())
        
        # 1. 将日期Date字段转换为日期格式
        if '日期Date' in df.columns:
            df['日期Date'] = pd.to_datetime(df['日期Date'], format='%Y%m%d')
        
        # 2. 将指数代码和成分券代码转换为6位数字的字符串
        if '指数代码 Index Code' in df.columns:
            df['指数代码 Index Code'] = df['指数代码 Index Code'].apply(lambda x: f"{x:06d}")
        
        if '成份券代码Constituent Code' in df.columns:
            df['成份券代码Constituent Code'] = df['成份券代码Constituent Code'].apply(lambda x: f"{x:06d}")
        
        # 3. 对特定指数进行行数校验
        if '指数代码 Index Code' in df.columns and not df.empty:
            index_code = df['指数代码 Index Code'].iloc[0]
            expected_rows = {
                '000016': 50,    # 上证50
                '000300': 300,   # 沪深300
                '000905': 500,   # 中证500
                '000852': 1000   # 中证1000
            }
            
            if index_code in expected_rows:
                actual_rows = len(df)
                expected = expected_rows[index_code]
                if actual_rows != expected:
                    # 行数不符合预期，但仍然返回DataFrame
                    print(f"警告：指数 {index_code} 行数 {actual_rows} 不符合预期 {expected}")
                    pass
        
        return df
        
    except Exception as e:
        return None
