#!/usr/bin/env python3
"""
朝阳永续数据库快速查询脚本
简化版查询工具
"""

import sys
import os

# 修复：确保可以正确导入db_query模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from db_query import safe_query
except ImportError as e:
    print(f"导入模块失败: {e}", file=sys.stderr)
    sys.exit(1)

def quick_query(sql: str, limit: int = 100) -> str:
    """
    快速查询数据库，默认限制返回记录数
    
    Args:
        sql: SQL查询语句
        limit: 最大返回记录数
    
    Returns:
        格式化后的查询结果字符串
    """
    # 添加LIMIT子句（如果还没有）
    sql_lower = sql.strip().lower()
    if 'limit' not in sql_lower:
        sql = f"{sql.rstrip(';')} LIMIT {limit}"
    
    try:
        # 使用JSON格式输出
        result = safe_query(sql, output_format='json')
        return result
    except Exception as e:
        return f"查询失败: {str(e)}"

def main():
    if len(sys.argv) < 2:
        print("用法: python quick_query.py 'SQL查询语句' [limit]")
        sys.exit(1)
    
    sql = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    result = quick_query(sql, limit)
    print(result)

if __name__ == '__main__':
    main()