# 股指期货与股票篮子对冲系统

本仓库用于开发股指期货与股票篮子对冲、跟踪误差控制及偏差分析系统。

## 当前状态

项目处于结构搭建阶段。各模块目前仅包含接口或占位实现，禁止直接用于实盘交易。

## 目录

- `src/hedge_system/`：核心 Python 包
- `config/`：非敏感配置模板
- `tests/`：单元、集成和回测测试
- `docs/`：架构、字段字典和决策记录
- `scripts/`：本地运行脚本
- `data/`：本地数据目录，不提交真实数据
- `outputs/`、`logs/`：本地运行产物，不提交真实产物

## 本地开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

复制 `.env.example` 为本地 `.env` 后再配置接口凭证。不要提交 `.env`。

