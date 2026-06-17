#!/usr/bin/env python3
"""
朝阳永续数据库查询脚本
基于dbquery.py进行数据库查询操作
"""

import pandas as pd
from sqlalchemy import create_engine, exc, text
import socket
import sys
import json
import argparse
from typing import Optional, Dict, Any, List

# 定义内部主机列表
INNER_HOSTS = ['commandocapitaldatabase']

class CommandoDBEngine:
    def __init__(self, inner_host: bool = True, database: str = 'ggbase'):
        self.username = 'queryuser'
        self.password = 'Ggbase4User%40Query'
        self.dbname = database
        self.hostname = self.get_machine_info()

        # 内网连接参数
        self.inner_config = {
            'host': '192.168.1.30',
            'port': 3306
        }
        
        # 外网连接参数
        self.outer_config = {
            'host': '111.207.128.74',
            'port': 33069
        }

        if inner_host or (self.hostname in INNER_HOSTS):
            self.first_config = self.inner_config
            self.second_config = self.outer_config
        else:
            self.first_config = self.outer_config
            self.second_config = self.inner_config
        
        self.engine = self.create_database_engine()

    def get_machine_info(self):
        """获取当前机器的主机名。"""
        return socket.gethostname()

    def try_connect(self, config, label):
        """尝试使用指定配置连接数据库"""
        try:
            # 修复：密码中包含@符号，需要进行URL编码
            # 注意：密码中的%40已经是编码后的@，不需要再次编码
            connection_url = f"mysql+pymysql://{self.username}:{self.password}@{config['host']}:{config['port']}/{self.dbname}"
            print(f"正在尝试{label}连接: {config['host']}:{config['port']}", file=sys.stderr)
            engine = create_engine(connection_url, pool_recycle=3600)
            
            # 测试连接
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            
            print(f"成功使用{label}连接数据库", file=sys.stderr)
            return engine
        except Exception as e:
            print(f"{label}连接失败: {e}", file=sys.stderr)
            return None

    def create_database_engine(self):
        """创建数据库引擎，先试first_config，再试second_config。"""
        try:
            # 先尝试first_config连接
            engine = self.try_connect(self.first_config, "first_config")
            
            # 如果first_config连接失败，尝试second_config连接
            if engine is None:
                engine = self.try_connect(self.second_config, "second_config")
            
            if engine is None:
                print("所有连接尝试均失败", file=sys.stderr)
                return None
            
            return engine
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            return None

    def query_to_dataframe(self, sql: str) -> pd.DataFrame:
        """执行SQL查询并返回DataFrame"""
        if self.engine is None:
            raise ConnectionError("数据库引擎未初始化")
        
        with self.engine.connect() as connection:
            result = connection.execute(text(sql))
            columns = result.keys()
            data = result.fetchall()
            df = pd.DataFrame(data, columns=columns)
            return df

    def query_to_dict(self, sql: str) -> List[Dict[str, Any]]:
        """执行SQL查询并返回字典列表"""
        df = self.query_to_dataframe(sql)
        return df.to_dict('records')

def safe_query(sql: str, output_format: str = 'dataframe', inner_host: bool = True) -> Any:
    """
    安全执行数据库查询
    
    Args:
        sql: SQL查询语句
        output_format: 输出格式，支持 'dataframe', 'dict', 'json'
        inner_host: 是否优先使用内网连接
    
    Returns:
        根据output_format返回不同格式的数据
    """
    # 安全检查：只允许SELECT、SHOW、DESC语句
    sql_lower = sql.strip().lower()
    if not (sql_lower.startswith('select') or 
            sql_lower.startswith('show') or 
            sql_lower.startswith('desc')):
        raise ValueError("只允许SELECT/SHOW/DESC查询语句")
    
    db_engine = CommandoDBEngine(inner_host=inner_host)
    
    if db_engine.engine is None:
        raise ConnectionError("数据库连接失败")
    
    if output_format == 'dataframe':
        return db_engine.query_to_dataframe(sql)
    elif output_format == 'dict':
        return db_engine.query_to_dict(sql)
    elif output_format == 'json':
        result = db_engine.query_to_dict(sql)
        return json.dumps(result, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"不支持的输出格式: {output_format}")

def main():
    parser = argparse.ArgumentParser(description='朝阳永续数据库查询工具')
    parser.add_argument('sql', type=str, help='SQL查询语句')
    parser.add_argument('--format', choices=['csv', 'json', 'table'], default='table', 
                       help='输出格式 (默认: table)')
    parser.add_argument('--inner', action='store_true', default=True,
                       help='优先使用内网连接 (默认: True)')
    parser.add_argument('--outer', action='store_false', dest='inner',
                       help='优先使用外网连接')
    
    args = parser.parse_args()
    
    try:
        # 执行查询
        df = safe_query(args.sql, output_format='dataframe', inner_host=args.inner)
        
        # 根据格式输出
        if args.format == 'csv':
            df.to_csv(sys.stdout, index=False, encoding='utf-8')
        elif args.format == 'json':
            json_str = df.to_json(orient='records', force_ascii=False, indent=2)
            print(json_str)
        else:  # table
            print(df.to_string())
            
    except Exception as e:
        print(f"查询失败: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()