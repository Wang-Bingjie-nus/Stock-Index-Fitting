"""下载模块 - 从中证指数官网下载指数相关Excel文件"""

import requests
import os
from datetime import datetime

def download_csi_constituent(index_code: str, save_path: str = None, download_type: str = "cons"):
    """
    从中证指数官网下载指数相关Excel文件
    
    :param index_code: 指数代码（如 000300 沪深300，000905 中证500）
    :param save_path: 保存路径，默认当前目录下的 csi_constituent 文件夹
    :param download_type: 下载类型，cons=成分股, closeweight=样本权重
    :return: 下载成功返回文件路径，失败返回None
    """
    # 设置保存路径
    if save_path is None:
        save_path = "csi_constituent"
    
    # 创建文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 生成文件名（包含指数代码+日期+类型）
    today = datetime.now().strftime("%Y%m%d")
    if download_type == "closeweight":
        filename = f"{index_code}_样本权重_{today}.xls"
    else:
        filename = f"{index_code}_成分股_{today}.xls"
    full_path = os.path.join(save_path, filename)

    # 构建请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    # 创建会话，先访问指数详情页获取cookie
    session = requests.Session()
    index_detail_url = f"https://www.csindex.com.cn/indices/family/detail?indexCode={index_code}"
    
    try:
        print(f'正在准备下载指数 {index_code} {"样本权重" if download_type == "closeweight" else "成分股"}...')
        print(f"访问指数详情页: {index_detail_url}")
        
        # 访问指数详情页
        detail_response = session.get(index_detail_url, headers=headers, timeout=30)
        print(f"详情页响应状态码: {detail_response.status_code}")
        
        # 构建下载链接
        if download_type == "closeweight":
            base_url = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/closeweight/"
            file_url = f"{base_url}{index_code}closeweight.xls"
        else:
            base_url = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/"
            file_url = f"{base_url}{index_code}cons.xls"
        
        print(f"下载链接: {file_url}")
        
        # 发送下载请求，使用同一个会话
        response = session.get(file_url, headers=headers, timeout=30, allow_redirects=True)
        
        # 打印调试信息
        print(f"响应状态码: {response.status_code}")
        print(f"响应URL: {response.url}")
        
        # 检查响应内容类型
        content_type = response.headers.get('Content-Type', '')
        print(f"内容类型: {content_type}")
        
        # 检查是否是Excel文件
        if 'application/vnd.ms-excel' in content_type or 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type:
            # 保存文件
            with open(full_path, "wb") as f:
                f.write(response.content)
            print(f"✅ 下载完成！文件保存至：{full_path}")
            return full_path
        elif response.status_code == 200:
            # 检查内容是否包含Excel文件的特征
            content = response.content
            if b'<?xml version="1.0"' in content and b'<Workbook' in content:
                # 这是一个Excel XML文件
                with open(full_path, "wb") as f:
                    f.write(content)
                print(f"✅ 下载完成！文件保存至：{full_path}")
                return full_path
            elif b'PK\x03\x04' in content:
                # 这是一个ZIP格式的Excel文件（.xlsx）
                xlsx_path = full_path.replace('.xls', '.xlsx')
                with open(xlsx_path, "wb") as f:
                    f.write(content)
                print(f"✅ 下载完成！文件保存至：{xlsx_path}")
                return xlsx_path
            else:
                # 保存HTML内容以便分析
                html_path = full_path.replace('.xls', '.html')
                with open(html_path, "wb") as f:
                    f.write(content)
                print(f"❌ 下载失败：返回的不是Excel文件，而是HTML内容")
                print(f"HTML内容已保存至：{html_path}")
                
                # 尝试使用不同的下载链接格式
                print("尝试使用备用下载链接...")
                alternative_urls = [
                    f"https://www.csindex.com.cn/csindex-webapp/search/cons?indexCode={index_code}&type={download_type}",
                    f"https://www.csindex.com.cn/csindex-webapp/index-detail/{index_code}/constituent"
                ]
                
                for alt_url in alternative_urls:
                    print(f"尝试备用链接: {alt_url}")
                    alt_response = session.get(alt_url, headers=headers, timeout=30)
                    print(f"备用链接响应状态码: {alt_response.status_code}")
                    
                    if alt_response.status_code == 200:
                        alt_content = alt_response.content
                        content_type = alt_response.headers.get('Content-Type', '')
                        if 'application/vnd.ms-excel' in content_type or 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type or b'<?xml version="1.0"' in alt_content or b'PK\x03\x04' in alt_content:
                            with open(full_path, "wb") as f:
                                f.write(alt_content)
                            print(f"✅ 备用链接下载完成！文件保存至：{full_path}")
                            return full_path
                
                return None
        else:
            print(f"❌ 下载失败：服务器返回状态码 {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ 下载失败：{str(e)}")
        print("请检查指数代码是否正确，或网络是否正常")
        return None
