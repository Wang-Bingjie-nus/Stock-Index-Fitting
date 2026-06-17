# 股指单日静态拟合原型：AI 程序生成说明

## 一、生成任务

你是一名熟悉 Python、Jupyter Notebook、XtQuant、同花顺 HTTP、A 股交易规则、公开网页爬取、指数复制和组合优化的高级量化开发工程师。

请在当前项目中实际生成可运行的原型程序，而不是只提供方案、伪代码或代码片段。

必须生成：

```text
Stock-Index-Fitting/
├── 股指拟合.ipynb
└── index_fitting/
    ├── __init__.py
    ├── runtime.py
    ├── csi_adapter.py
    ├── xtquant_adapter.py
    ├── ifind_http.py
    ├── sse_etf_pcf.py
    ├── trading_rules_scraper.py
    └── dataset_layout.py
```

### 代码职责

| 文件 | 职责 |
|---|---|
| `股指拟合.ipynb` | 参数配置、主流程编排、统一校验、拟合算法、偏差计算、报告生成和结果展示 |
| `runtime.py` | 日志、运行上下文和 `ManualReviewRequired` |
| `dataset_layout.py` | 构造本次运行目录及文件路径 |
| `csi_adapter.py` | 薄封装现有 `CSI.csiweb`，下载并读取指数权重与成分文件 |
| `xtquant_adapter.py` | 迅投连接、指数日行情、主 ETF PCF、除权除息 |
| `ifind_http.py` | 同花顺 HTTP Token、交易日、相关 ETF、行情、除权除息、交易状态 |
| `sse_etf_pcf.py` | 上交所主 ETF 申赎清单爬虫 |
| `trading_rules_scraper.py` | 上交所、深交所官方交易数量规则爬取与唯一解析 |

外部接口、文件读取和网页爬虫必须放在独立 `.py` 模块中。Notebook 只调用模块，不重复实现外部数据获取细节。

## 二、第一版目标与边界

### 核心目标

第一版只完成单日静态拟合：

1. 自动确定严格早于程序运行自然日的最近交易日，作为 `build_date`。
2. 获取程序运行时公开最新的指数样本权重。
3. 获取并交叉验证构建日指数与成分股未复权收盘价。
4. 构建符合交易数量规则的目标股票篮子。
5. 分析理论股票篮子与目标股票篮子的静态拟合偏差。

```text
target_stock_value = index_close × contract_multiplier × index_units
```

### 第一版不做

- 不输入具体股指期货合约代码。
- 不计算期货损益、基差、保证金、交易成本、滑点或市场冲击。
- 不执行交易，不模拟成交。
- 不进行多日回测。
- ETF、公司行为和构建日交易状态不参与目标数量计算。
- 不使用 ETF 替代指数成分股。
- 不自动推断 `contract_multiplier`。
- 不生成图表。

## 三、关键业务口径

### 日期与运行目录

| 名称 | 类型 | 含义 | 格式 |
|---|---|---|---|
| `build_date` | date | 严格早于运行自然日的最近交易日 | `YYYYMMDD` |
| `import_time` | string | 本次程序运行目录键 | `YYYYMMDD-HHMM` |

不得把 `build_date` 与 `import_time` 混用。第一版不处理一分钟内重复运行。

本次运行全部文件必须收拢到：

```text
data/{index_code}/{import_time}/
├── 01_index/
├── 02_etf/
├── 03_corporate_actions/
├── 04_market/
├── 05_trading_status/
├── 06_trading_rules/
├── reports/
└── logs/
```

所有需要标识本次运行的结构统一使用 `import_time`，不得另设运行编号字段。

### 权重与价格

- 指数权重仅来自程序运行时通过 `CSI.csiweb` 下载的公开最新样本权重。
- 权重直接使用原始值，不归一化。
- 指数构建日收盘点位：迅投与同花顺交叉验证，通过后采用迅投值。
- 成分股构建日未复权收盘价：`nas_reader.read_daily_data(build_date)` 与同花顺交叉验证，通过后采用 NAS 字段 `tclose`。
- 双源收盘价相对偏差阈值均为 `0.01%`。
- 任一侧缺失、值非正数或偏差超过阈值时阻断。

### 主 ETF

- 主 ETF 由用户通过 `primary_etf_code` 显式指定。
- 若未指定且 `index_code == "000016"`，默认使用 `510050.SH`。
- 其他指数未指定主 ETF 时，不得自动猜测，必须进入人工审核。
- 同花顺仅用于获取跟踪目标指数的 ETF 产品清单，并验证主 ETF 属于该清单。
- 上交所与迅投仅对主 ETF PCF 进行严格交叉验证。

### 公司行为

- 仅查询历史一天与未来一天的除权除息事件。
- 迅投与同花顺结果不做一致性裁决，而是按 `stock_code + ex_date` 合并。
- 两侧同时存在时，迅投非空字段优先，同花顺补充空字段。
- 单侧事件直接保留。
- 任一接口异常阻断；正常空结果不阻断。

### 构建日交易状态

- 通过同花顺获取成分股构建日涨跌停与停牌状态。
- 状态只保存并用于后续复盘，不参与拟合。
- 不获取或推算涨跌停价格。
- 接口异常阻断；单只股票正常返回空状态允许继续。

## 四、主函数接口

```python
run_index_fitting(
    index_code: str,
    index_units: int,
    contract_multiplier: float,
    primary_etf_code: str | None = None,
    refresh_trade_rules: bool = False,
    debug: bool = False,
)
```

### 输入

| 字段名称 | 字段类型 | 字段含义 | 输入格式 |
|---|---|---|---|
| index_code | string | 目标指数代码 | 必填，六位代码，例如 `000016` |
| index_units | int | 用于确定目标股票金额的指数单位数 | 必填，正整数 |
| contract_multiplier | float | 用户指定的标准合约乘数 | 必填，正数 |
| primary_etf_code | string 或 null | 用于 PCF 严格验证的主 ETF | 带市场后缀；上证50未提供时默认 `510050.SH` |
| refresh_trade_rules | bool | 是否强制重新爬取官方交易规则 | 默认 `false` |
| debug | bool | 是否启用详细日志与调试文件 | 默认 `false` |

### 成功返回值

主函数只返回核心结果，不返回全部业务中间产物：

| 字段名称 | 字段类型 | 字段含义 | 输出格式 |
|---|---|---|---|
| run_summary | DataFrame | 本次运行参数、状态与摘要 | 单行表 |
| validation_report | DataFrame | 全流程校验记录 | 每项校验一行 |
| theoretical_portfolio | DataFrame | 未考虑交易数量规则的理论股票篮子 | 每只成分股一行 |
| target_portfolio | DataFrame | 两步贪心法生成的目标股票篮子 | 每只成分股一行 |
| deviation_report | DataFrame | 理论篮子与目标篮子的静态偏差 | 每只成分股一行 |
| portfolio_summary | dict | 组合级拟合评价 | JSON 可序列化字典 |
| output_paths | dict | 本次运行关键文件路径 | 绝对路径字典 |

## 五、实现原则

- Task 必须按本文整数序号顺序实现和调用。
- 每个 Task 写成独立函数，并保留 `debug=False` 参数。
- 主函数显式传递上一步输出，不依赖隐藏全局变量。
- 每个函数必须具有中文 docstring，说明参数、返回值、数据口径、异常和限制。
- 所有 DataFrame 字段名称必须稳定。
- `output_paths` 中存在对应 `*_csv` 路径的 Task 结构化输出，必须由产生该输出的 Task 在正常返回前立即写入；Task14 不依赖未显式传入的内存中间结果。
- 外部接口实际返回字段必须通过现有代码或最小探测确认，不得凭空编造。
- Token 只能从环境变量或已有 `.env` 读取，不得硬编码或写入日志。
- 关键失败必须保存已取得的证据后抛出 `ManualReviewRequired`。
- 不允许自动降级、删除异常股票或使用未经确认的替代数据。
- Notebook 必须可从上到下运行，最后一个单元默认调用主函数。

## 六、人工审核与统一校验

```python
class ManualReviewRequired(RuntimeError):
    def __init__(
        self,
        message: str,
        import_time: str,
        validation_report: pd.DataFrame,
        output_paths: dict,
    ):
        ...
```

### 校验记录字段

| 字段名称 | 字段类型 | 字段含义 | 输出格式 |
|---|---|---|---|
| check_name | string | 稳定校验名称 | 英文 snake_case |
| level | string | 严重级别 | `ERROR` / `WARNING` / `INFO` |
| status | string | 校验状态 | `PASS` / `FAIL` |
| actual_value | string | 实际结果 | 标量或 JSON 文本 |
| expected_value | string | 期望结果 | 标量或 JSON 文本 |
| tolerance | string | 允许误差 | 无则为空 |
| message | string | 说明 |  |
| checked_at | datetime | 校验时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | `YYYYMMDD-HHMM` |

### 统一阻断规则

以下情况必须阻断：

- 迅投或同花顺连接失败。
- CSI 权重或成分文件获取失败、无法解析或成分集合不一致。
- 无法确定构建日。
- 指数构建日收盘价双源验证失败。
- 任一成分股构建日收盘价双源验证失败。
- 官方交易规则无法唯一解析。
- 任一成分股无法唯一匹配交易规则。
- 主 ETF 不在同花顺目标指数相关 ETF 清单内。
- 主 ETF 上交所 PCF 或迅投 PCF 获取失败或两侧不一致。
- 公司行为任一数据源接口异常。
- 构建日交易状态接口异常。

以下情况不阻断：

- 公司行为正常返回空结果。
- 公司行为只有一侧存在事件。
- 单只股票交易状态正常返回空值。
- 未来一天没有公司行为事件。

## 七、外部数据源与已验证接口

### 必须实际读取并参考的现有文件

- `fetch_xtquant_data.py`
- `XtQuant.ipynb`
- `CSI/csiweb/`
- `nas_reader/`
- `songge/同花顺指数数据导入.ipynb`
- `songge/etf申赎清单爬虫.ipynb`
- `股指拟合流程增删.txt`

### CSI

```python
from CSI.csiweb import download_csi_constituent, read_csi_file
```

- 不重新实现 CSI 下载器。
- `csi_adapter.py` 只做薄封装。
- 同时下载权重文件和成分文件，并严格校验成分集合一致。

### NAS 日行情

```python
from nas_reader import read_daily_data
df_daily = read_daily_data(build_date)
```

已确认字段：

| 字段名称 | 字段含义 |
|---|---|
| xt_stock_code | 带市场后缀的股票代码 |
| trade_date | 交易日期 |
| tclose | 未复权收盘价 |
| topen | 开盘价 |
| high | 最高价 |
| low | 最低价 |
| volume | 成交量 |
| amount | 成交额 |

NAS 日行情只包含股票，不得用其中的六位代码替代指数代码。

### 同花顺 HTTP

- 凭证：环境变量 `IFIND_REFRESH_TOKEN`。
- 通过 `get_access_token` 换取 access token。
- 基址：`https://quantapi.51ifind.com/api/v1`。
- 使用 `get_trade_dates` 确定最近交易日和前一交易日，`marketcode=212001`。
- 使用 `smart_stock_picking`、`searchtype=fund` 获取跟踪目标指数的 ETF 产品清单。
- 使用 `date_sequence` 获取除权除息及构建日交易状态。
- 使用已验证行情接口获取指数和成分股构建日日行情。
- 同花顺不提供指数权重，也不提供 ETF PCF。

### 上交所主 ETF PCF

- 数据源：`https://query.sse.com.cn/commonQuery.do`。
- 请求前访问 `https://www.sse.com.cn/disclosure/fund/etflist/` 预热 Referer。
- 基本信息 SQL ID：`COMMON_SSE_CP_JJLB_ETFJJGK_GGSGSHQD_JBXX_C`。
- 成分信息 SQL ID：`COMMON_SSE_CP_JJLB_ETFJJGK_GGSGSHQD_COMPONENT_C`。
- 以上交所 PCF 为权威值，迅投 PCF 仅作对照。

## 八、详细 Task

### Task0：初始化运行环境

实现位置：`runtime.py`、`dataset_layout.py`

```python
initialize_runtime(index_code, debug=False)
```

任务目标：
初始化本次运行的唯一目录标识、时间信息和全部输出路径，为后续 Task 提供只读的运行上下文。

处理规则：
- 生成 `import_time` 与 `started_at`。
- 一次性创建本次运行的完整目录结构，并规划全部关键文件路径。
- 配置本次运行日志；路径已规划不代表对应文件已经生成。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| index_code | string | 目标指数代码 | 必填，六位代码 | 主函数输入 |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM | Task2、Task3 及所有需要运行标识的后续 Task |
| started_at | datetime | 主流程开始时间 | ISO 8601，含时区 | Task14 |
| output_paths | dict | 本次运行全部规划目录与关键文件路径；后续 Task 只读，不修改 | 完整字段结构见下表 | Task3 至 Task14、主函数返回值 |
##### `output_paths` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| run_dir | string | 本次运行根目录 | 绝对路径 |
| index_dir | string | 指数权重、成分及校验文件目录 | 绝对路径 |
| etf_dir | string | 相关 ETF 与主 ETF PCF 文件目录 | 绝对路径 |
| corporate_actions_dir | string | 公司行为文件目录 | 绝对路径 |
| market_dir | string | 构建日行情文件目录 | 绝对路径 |
| trading_status_dir | string | 构建日交易状态文件目录 | 绝对路径 |
| trading_rules_dir | string | 本次运行交易规则文件目录 | 绝对路径 |
| reports_dir | string | 最终报告目录 | 绝对路径 |
| logs_dir | string | 日志目录 | 绝对路径 |
| log_path | string | 本次运行日志文件 | 绝对路径，`.log` |
| index_weights_csv | string | 标准化指数权重文件 | 绝对路径，`.csv` |
| index_cons_csv | string | 标准化指数成分文件 | 绝对路径，`.csv` |
| index_constituents_check_csv | string | 指数双文件成分校验文件 | 绝对路径，`.csv` |
| related_etfs_csv | string | 目标指数相关 ETF 清单 | 绝对路径，`.csv` |
| sse_etf_components_csv | string | 上交所主 ETF PCF | 绝对路径，`.csv` |
| xt_etf_components_csv | string | 迅投主 ETF PCF | 绝对路径，`.csv` |
| etf_cross_check_csv | string | 主 ETF PCF 对照结果 | 绝对路径，`.csv` |
| corporate_actions_xt_csv | string | 迅投公司行为标准化表 | 绝对路径，`.csv` |
| corporate_actions_ifind_csv | string | 同花顺公司行为标准化表 | 绝对路径，`.csv` |
| corporate_actions_merged_csv | string | 合并公司行为表 | 绝对路径，`.csv` |
| corporate_action_failures_csv | string | 公司行为接口异常清单 | 绝对路径，`.csv` |
| trading_status_csv | string | 构建日交易状态表 | 绝对路径，`.csv` |
| security_rules_csv | string | 本次运行交易规则快照 | 绝对路径，`.csv` |
| rule_parser_candidates_csv | string | 交易规则解析候选表 | 绝对路径，`.csv` |
| component_rules_csv | string | 成分股板块与交易规则匹配表 | 绝对路径，`.csv` |
| market_snapshot_csv | string | 构建日行情标准表 | 绝对路径，`.csv` |
| report_xlsx | string | Excel 汇总报告 | 绝对路径，`.xlsx` |
| run_summary_csv | string | 运行摘要文件 | 绝对路径，`.csv` |
| validation_report_csv | string | 完整校验报告文件 | 绝对路径，`.csv` |
| theoretical_portfolio_csv | string | 理论股票篮子文件 | 绝对路径，`.csv` |
| target_portfolio_csv | string | 目标股票篮子文件 | 绝对路径，`.csv` |
| deviation_report_csv | string | 静态偏差报告文件 | 绝对路径，`.csv` |
| portfolio_summary_csv | string | 组合摘要文件 | 绝对路径，`.csv` |

### Task1：初始化外部数据源连接

实现位置：`xtquant_adapter.py`、`ifind_http.py`

主流程直接调用两个函数，因此分别定义契约。

任务目标：
初始化并验证迅投与同花顺 HTTP 两个外部数据源连接，确保后续 Task 可以访问强制数据源。

处理规则：
- 分别建立迅投连接和同花顺 HTTP 连接，并执行能够证明连接可用的最小检查。
- Token 只能从环境变量或已有 `.env` 读取，不得写入日志或返回结果。

校验与失败规则：
- 任一连接失败时，必须立即保存当前证据并抛出 `ManualReviewRequired`，不正常返回。

#### 1.1 迅投连接

```python
initialize_xtquant_connection(debug=False)
```

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| xtquant_connection_status | dict | 迅投连接状态 | 完整字段结构见下表 | 主函数仅用于确认连接成功 |
##### `xtquant_connection_status` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| connected | bool | 迅投连接是否成功 | true / false |
| data_dir | string | 迅投本地数据目录 | 绝对路径 |
| checked_at | datetime | 连接检查时间 | ISO 8601，含时区 |
| message | string | 连接状态说明 |  |
#### 1.2 同花顺 HTTP 连接

```python
initialize_ifind_connection(debug=False)
```

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| ifind_connection_status | dict | 同花顺 HTTP 连接状态 | 完整字段结构见下表 | 主函数仅用于确认连接成功 |
##### `ifind_connection_status` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| connected | bool | 同花顺 HTTP 连接是否成功 | true / false |
| access_token_obtained | bool | 是否成功换取 access token | true / false |
| checked_at | datetime | 连接检查时间 | ISO 8601，含时区 |
| message | string | 连接状态说明 |  |

### Task2：确定构建日与指数代码

实现位置：`ifind_http.py`，Notebook 负责编排。

```python
resolve_index_context(index_code, import_time, debug=False)
```

任务目标：
根据目标指数确定统一的指数代码表示、构建日与前一交易日，为后续数据查询建立日期和代码口径。

处理规则：
- 生成目标指数的迅投代码与同花顺代码。
- 使用同花顺交易日接口确定严格早于程序运行自然日的最近交易日作为 `build_date`，并确定其前一交易日。

校验与失败规则：
- 指数代码格式无效、无法唯一生成数据源代码或无法确定交易日期时阻断。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| index_code | string | 目标指数代码 | 六位代码 | 主函数输入 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM | Task0 输出 import_time |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| index_context | dict | 构建日及两数据源指数代码上下文 | 完整字段结构见下表 | Task4、Task5、Task6、Task9、Task11、Task14 |
##### `index_context` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| index_code | string | 六位目标指数代码 | 六位代码 |
| xt_index_code | string | 迅投指数代码 | 带市场后缀 |
| ifind_index_code | string | 同花顺指数代码 | 带市场后缀 |
| index_exchange | string | 指数所属交易所 | SH / SZ |
| build_date | date | 严格早于运行自然日的最近交易日 | YYYY-MM-DD |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task3：下载、读取并校验 CSI 权重与成分

实现位置：`csi_adapter.py`

```python
get_and_validate_csi_index_files(index_code, raw_csi_dir, import_time, debug=False)
```

任务目标：
获取程序运行时公开最新的 CSI 指数样本权重与成分文件，并确认两份文件描述同一组成集合。

处理规则：
- 每次运行均通过现有 `CSI.csiweb` 下载权重文件和成分文件，不使用未经确认的历史文件替代。
- 读取并标准化两份文件，将原始文件、标准化结果和校验结果写入本次运行目录。
- 权重使用公开文件中的原始值，不做归一化。

校验与失败规则：
- 下载失败、文件无法解析、代码不唯一或权重文件与成分文件集合不一致时阻断，不正常返回。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| index_code | string | 目标指数代码 | 六位代码 | Task2 输出 index_context.index_code |
| raw_csi_dir | string 或 Path | 指数权重、成分及校验文件目录 | 绝对路径 | Task0 输出 output_paths.index_dir |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM | Task0 输出 import_time |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| df_index_weights | DataFrame | 最新公开样本权重 | 完整字段结构见下表 | Task4、Task5、Task6、Task8、Task9、Task10、Task11 |
| df_index_cons | DataFrame | 最新公开成分股名单 | 完整字段结构见下表 | Task14 报告 |
| df_index_constituents_check | DataFrame | 双文件成分集合校验 | 完整字段结构见下表 | Task14 报告 |
| validation_records | DataFrame | 本 Task 正常完成时产生的校验记录 | 完整字段结构见下表 | 主函数追加至 validation_report |
##### `df_index_weights` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| index_code | string | 目标指数代码 | 六位代码 |
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| source_file | string | 来源文件 | 文件名 |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_index_cons` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| index_code | string | 目标指数代码 | 六位代码 |
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| source_file | string | 来源文件 | 文件名 |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_index_constituents_check` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| in_weights | bool | 是否存在于权重文件 | true / false |
| in_cons | bool | 是否存在于成分文件 | true / false |
| check_status | string | 集合校验结果 | MATCHED / ONLY_IN_WEIGHTS / ONLY_IN_CONS |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `validation_records` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| check_name | string | 稳定校验名称 | 英文 snake_case |
| level | string | 严重级别 | ERROR / WARNING / INFO |
| status | string | 校验状态 | PASS / FAIL |
| actual_value | string | 实际结果 | 标量或 JSON 文本 |
| expected_value | string | 期望结果 | 标量或 JSON 文本 |
| tolerance | string | 允许误差 | 无则为空 |
| message | string | 说明 |  |
| checked_at | datetime | 校验时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task4：获取目标指数相关 ETF 并验证主 ETF PCF

实现位置：`ifind_http.py`、`sse_etf_pcf.py`、`xtquant_adapter.py`

```python
get_and_validate_primary_etf(index_code, primary_etf_code, build_date, etf_dir, import_time, debug=False)
```

任务目标：
确认主 ETF 确实跟踪目标指数，并通过上交所与迅投 PCF 交叉验证其申赎成分。

处理规则：
- 通过同花顺获取目标指数相关 ETF 清单，并验证主 ETF 属于该清单。
- 主 ETF 未指定且 `index_code == "000016"` 时使用 `510050.SH`；其他指数不得自动猜测主 ETF。
- 分别获取上交所主 ETF PCF 与迅投主 ETF PCF，将原始及标准化结果写入本次运行目录。
- PCF 以 `etf_code + stock_code` 对齐，比较成分集合与数量。

校验与失败规则：
- 无法确定主 ETF、主 ETF 不在相关 ETF 清单、任一 PCF 获取失败，或两侧 PCF 存在任何成分或数量差异时阻断。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| index_code | string | 目标指数代码 | 六位代码 | Task2 输出 index_context.index_code |
| primary_etf_code | string 或 null | 用户指定的主 ETF；上证50为空时默认 510050.SH | 带市场后缀或 null | 主函数输入 |
| build_date | date | 构建日 | YYYY-MM-DD | Task2 输出 index_context.build_date |
| etf_dir | string 或 Path | 相关 ETF 与主 ETF PCF 文件目录 | 绝对路径 | Task0 输出 output_paths.etf_dir |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM | Task0 输出 import_time |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| resolved_primary_etf_code | string | 最终主 ETF 代码 | 带市场后缀 | Task14 |
| df_related_etfs | DataFrame | 同花顺返回的目标指数相关 ETF | 完整字段结构见下表 | Task14 报告 |
| df_sse_etf_components | DataFrame | 上交所主 ETF PCF | 完整字段结构见下表 | Task14 报告 |
| df_xt_etf_components | DataFrame | 迅投主 ETF PCF | 完整字段结构见下表 | Task14 报告 |
| df_etf_cross_check | DataFrame | 主 ETF PCF 对照结果 | 完整字段结构见下表 | Task14 报告 |
| validation_records | DataFrame | 本 Task 正常完成时产生的校验记录 | 完整字段结构见下表 | 主函数追加至 validation_report |
##### `df_related_etfs` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| etf_code | string | ETF 代码 | 带市场后缀 |
| etf_name | string | ETF 名称 |  |
| tracked_index_code | string | 跟踪指数代码 | 带市场后缀或六位代码 |
| tracked_index_name | string | 跟踪指数名称 |  |
| source | string | 数据来源 | ifind.smart_stock_picking |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_etf_components` 输出（`df_sse_etf_components` 与 `df_xt_etf_components` 共用）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| etf_code | string | 主 ETF 代码 | 带市场后缀 |
| stock_code | string | PCF 成分证券代码 | 带市场后缀 |
| stock_name | string 或 null | PCF 成分证券名称 |  |
| component_qty | float | PCF 成分数量 | 股 |
| source | string | 数据来源 | SSE / XTQUANT |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_etf_cross_check` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| etf_code | string | 主 ETF 代码 | 带市场后缀 |
| stock_code | string | PCF 成分证券代码 | 带市场后缀 |
| sse_component_qty | float 或 null | 上交所 PCF 数量 | 股 |
| xt_component_qty | float 或 null | 迅投 PCF 数量 | 股 |
| qty_match_status | string | 数量对照结果 | MATCHED / MISMATCH / ONLY_IN_SSE / ONLY_IN_XT |
| authoritative_qty | float 或 null | 裁决采用的上交所数量 | 股 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `validation_records` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| check_name | string | 稳定校验名称 | 英文 snake_case |
| level | string | 严重级别 | ERROR / WARNING / INFO |
| status | string | 校验状态 | PASS / FAIL |
| actual_value | string | 实际结果 | 标量或 JSON 文本 |
| expected_value | string | 期望结果 | 标量或 JSON 文本 |
| tolerance | string | 允许误差 | 无则为空 |
| message | string | 说明 |  |
| checked_at | datetime | 校验时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task5：获取并合并公司行为

实现位置：`xtquant_adapter.py`、`ifind_http.py`

```python
get_and_merge_corporate_actions(stock_codes, build_date, actions_dir, import_time, debug=False)
```

任务目标：
获取构建日前后需要关注的公司行为，并合并为可审计的统一事件表。

处理规则：
- 历史查询窗口为构建日前一个自然日至构建日；未来查询窗口为构建日后一个自然日。
- 分别查询迅投和同花顺，以 `stock_code + ex_date` 合并事件。
- 两侧均存在时优先采用迅投非空字段，并用同花顺补充空字段；单侧事件直接保留。
- 将两侧标准化结果、合并结果和接口失败清单写入本次运行目录。

校验与失败规则：
- 任一数据源接口异常或结果无法解析时阻断。
- 接口正常返回空结果、仅单侧存在事件或未来一日没有事件时允许继续。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| stock_codes | list[string] | 全部指数成分股代码 | 每项带市场后缀 | Task3 输出 df_index_weights.stock_code |
| build_date | date | 构建日 | YYYY-MM-DD | Task2 输出 index_context.build_date |
| actions_dir | string 或 Path | 公司行为文件目录 | 绝对路径 | Task0 输出 output_paths.corporate_actions_dir |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM | Task0 输出 import_time |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| df_corporate_actions_xt | DataFrame | 迅投公司行为标准化表 | 完整字段结构见下表 | Task14 报告 |
| df_corporate_actions_ifind | DataFrame | 同花顺公司行为标准化表 | 完整字段结构见下表 | Task14 报告 |
| df_corporate_actions_merged | DataFrame | 按 stock_code + ex_date 合并的公司行为表 | 完整字段结构见下表 | Task14 报告及后续扩展 |
| df_corporate_action_failures | DataFrame | 接口异常清单；成功运行时应为空 | 完整字段结构见下表 | Task14 报告 |
##### `df_corporate_actions` 输出（`df_corporate_actions_xt` 与 `df_corporate_actions_ifind` 共用）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| ex_date | date | 除权除息日 | YYYY-MM-DD |
| query_window | string | 查询窗口 | HISTORICAL_DAY / FUTURE_DAY |
| cash_dividend_ps | float 或 null | 每股现金分红 | 人民币元 |
| bonus_shares_ps | float 或 null | 每股送股 | 股 |
| raw_json | string | 数据源原始记录 | JSON |
| source | string | 数据来源 | XTQUANT / IFIND |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_corporate_actions_merged` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| ex_date | date | 除权除息日 | YYYY-MM-DD |
| query_window | string | 查询窗口 | HISTORICAL_DAY / FUTURE_DAY |
| cash_dividend_ps | float 或 null | 合并后的每股现金分红 | 人民币元 |
| bonus_shares_ps | float 或 null | 合并后的每股送股 | 股 |
| sources | string | 数据来源覆盖情况 | BOTH / XT_ONLY / IFIND_ONLY |
| xt_raw_json | string 或 null | 迅投原始记录 | JSON |
| ifind_raw_json | string 或 null | 同花顺原始记录 | JSON |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_corporate_action_failures` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string 或 null | 发生异常的成分股代码 | 带市场后缀 |
| source | string | 异常数据源 | XTQUANT / IFIND |
| query_window | string | 查询窗口 | HISTORICAL_DAY / FUTURE_DAY |
| reason | string | 接口异常或解析失败原因 |  |
| retrieved_at | datetime | 异常发生时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task6：获取构建日交易状态

实现位置：`ifind_http.py`

```python
get_build_date_trading_status(stock_codes, build_date, status_dir, import_time, debug=False)
```

任务目标：
获取并保存全部成分股在构建日的涨跌停与停牌状态，供后续复盘使用。

处理规则：
- 通过同花顺查询构建日交易状态并写入本次运行目录。
- 交易状态只保存和报告，不参与第一版目标数量计算，也不推算涨跌停价格。

校验与失败规则：
- 接口异常或结果无法解析时阻断。
- 单只股票正常返回空状态时允许继续。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| stock_codes | list[string] | 全部指数成分股代码 | 每项带市场后缀 | Task3 输出 df_index_weights.stock_code |
| build_date | date | 构建日 | YYYY-MM-DD | Task2 输出 index_context.build_date |
| status_dir | string 或 Path | 构建日交易状态文件目录 | 绝对路径 | Task0 输出 output_paths.trading_status_dir |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM | Task0 输出 import_time |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| df_trading_status | DataFrame | 构建日涨跌停与停牌状态 | 完整字段结构见下表 | Task14 报告及后续偏差复盘 |
##### `df_trading_status` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| build_date | date | 状态所属交易日 | YYYY-MM-DD |
| up_down_status | string 或 null | 涨跌停状态 | 同花顺标准化值或空 |
| suspension_status | string 或 null | 停牌状态 | 同花顺标准化值或空 |
| source | string | 数据来源 | ifind.date_sequence |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task7：加载或刷新官方交易数量规则

实现位置：`trading_rules_scraper.py`

```python
get_security_buy_rules(raw_trading_rules_dir, shared_rule_file_path, refresh_trade_rules=False, debug=False)
```

任务目标：
获得能够唯一确定各支持板块买入数量约束的官方交易规则快照。

处理规则：
- 本地共享规则文件存在且 `refresh_trade_rules=False` 时允许直接读取。
- 强制刷新或共享文件不存在时，必须实际抓取上交所、深交所官方原文并唯一解析，不得硬编码规则结果。
- 保存官方原文、解析候选、共享规则文件和本次运行规则快照。

校验与失败规则：
- 官方来源无法访问、规则无法唯一解析或标准化规则存在冲突时阻断。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| raw_trading_rules_dir | string 或 Path | 本次运行交易规则文件目录 | 绝对路径 | Task0 输出 output_paths.trading_rules_dir |
| shared_rule_file_path | string 或 Path | 跨运行复用的共享规则文件 | 绝对路径，固定为项目根目录 data/trading_rules/security_buy_rules.csv | 项目固定配置 |
| refresh_trade_rules | bool | 是否强制重新爬取官方规则 | 默认 false | 主函数输入 |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| df_security_rules | DataFrame | 唯一解析后的规则表 | 完整字段结构见下表 | Task8 |
| df_rule_parser_candidates | DataFrame | 解析候选与状态 | 完整字段结构见下表 | Task14 报告 |
| validation_records | DataFrame | 本 Task 正常完成时产生的校验记录 | 完整字段结构见下表 | 主函数追加至 validation_report |
##### `df_security_rules` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| exchange | string | 交易所 | SH / SZ |
| board | string | 规则适用板块 | MAIN / STAR / CHINEXT |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| rule_source_url | string | 官方规则来源 | URL |
| rule_effective_date | date 或 null | 规则生效日期 | YYYY-MM-DD 或空 |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
##### `df_rule_parser_candidates` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| exchange | string | 交易所 | SH / SZ |
| board | string | 规则适用板块 | MAIN / STAR / CHINEXT |
| rule_source_url | string | 官方规则来源 | URL |
| candidate_text | string | 官方原文候选片段 | 文本 |
| parsed_buy_min_qty | int 或 null | 解析出的最低首次买入数量 | 股 |
| parsed_buy_qty_step | int 或 null | 解析出的递增步长 | 股 |
| parse_status | string | 解析状态 | PARSED / AMBIGUOUS / FAILED |
| message | string | 解析说明 |  |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
##### `validation_records` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| check_name | string | 稳定校验名称 | 英文 snake_case |
| level | string | 严重级别 | ERROR / WARNING / INFO |
| status | string | 校验状态 | PASS / FAIL |
| actual_value | string | 实际结果 | 标量或 JSON 文本 |
| expected_value | string | 期望结果 | 标量或 JSON 文本 |
| tolerance | string | 允许误差 | 无则为空 |
| message | string | 说明 |  |
| checked_at | datetime | 校验时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task8：识别成分股板块并匹配交易规则

Notebook 实现：

```python
classify_boards_and_match_rules(df_index_weights, df_security_rules, import_time, debug=False)
```

任务目标：
识别每只指数成分股所属交易所和板块，并为其匹配唯一的买入数量规则。

处理规则：
- 根据成分股代码与交易所口径识别板块。
- 将每只成分股与 Task7 的规则表匹配，生成并落盘成分股规则表。

校验与失败规则：
- 每只成分股必须且只能匹配一条交易规则；无法识别板块、无匹配或多重匹配时阻断。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| df_index_weights | DataFrame | 最新公开样本权重 | 完整字段结构见下表 | Task3 输出 df_index_weights |
| df_security_rules | DataFrame | 官方交易数量规则 | 完整字段结构见下表 | Task7 输出 df_security_rules |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM | Task0 输出 import_time |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
##### `df_index_weights` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| index_code | string | 目标指数代码 | 六位代码 |
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| source_file | string | 来源文件 | 文件名 |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_security_rules` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| exchange | string | 交易所 | SH / SZ |
| board | string | 规则适用板块 | MAIN / STAR / CHINEXT |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| rule_source_url | string | 官方规则来源 | URL |
| rule_effective_date | date 或 null | 规则生效日期 | YYYY-MM-DD 或空 |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| df_component_rules | DataFrame | 成分股板块与交易规则匹配表 | 完整字段结构见下表 | Task10、Task11、Task14 |
| validation_records | DataFrame | 本 Task 正常完成时产生的校验记录 | 完整字段结构见下表 | 主函数追加至 validation_report |
##### `df_component_rules` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| board | string | 板块 | MAIN / STAR / CHINEXT |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| rule_source_url | string | 官方规则来源 | URL |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `validation_records` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| check_name | string | 稳定校验名称 | 英文 snake_case |
| level | string | 严重级别 | ERROR / WARNING / INFO |
| status | string | 校验状态 | PASS / FAIL |
| actual_value | string | 实际结果 | 标量或 JSON 文本 |
| expected_value | string | 期望结果 | 标量或 JSON 文本 |
| tolerance | string | 允许误差 | 无则为空 |
| message | string | 说明 |  |
| checked_at | datetime | 校验时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task9：获取并交叉验证构建日行情

实现位置：`xtquant_adapter.py`、`ifind_http.py`；Notebook 负责统一校验。

```python
get_and_validate_build_date_market(xt_index_code, ifind_index_code, stock_codes, build_date, market_dir, import_time, debug=False)
```

任务目标：
获取构建日指数与全部成分股的未复权收盘价，并通过独立数据源交叉验证后形成统一行情快照。

处理规则：
- 指数收盘价使用迅投与同花顺交叉验证，通过后采用迅投值。
- 成分股收盘价使用 NAS `tclose` 与同花顺交叉验证，通过后采用 NAS 值。
- 将标准化行情快照与逐项校验记录写入本次运行目录。

校验与失败规则：
- 双源相对偏差必须 `<= 0.01%`。
- 任一侧缺失、价格非正数、日期错位、代码不唯一或相对偏差超限时阻断。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| xt_index_code | string | 迅投指数代码 | 带市场后缀 | Task2 输出 index_context.xt_index_code |
| ifind_index_code | string | 同花顺指数代码 | 带市场后缀 | Task2 输出 index_context.ifind_index_code |
| stock_codes | list[string] | 全部指数成分股代码 | 每项带市场后缀 | Task3 输出 df_index_weights.stock_code |
| build_date | date | 构建日 | YYYY-MM-DD | Task2 输出 index_context.build_date |
| market_dir | string 或 Path | 构建日行情文件目录 | 绝对路径 | Task0 输出 output_paths.market_dir |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM | Task0 输出 import_time |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| df_market_snapshot | DataFrame | 指数与成分股构建日行情标准表 | 完整字段结构见下表 | Task10、Task11、Task14 |
| index_close | float | 通过双源验证后采用的迅投指数收盘点位 | 正数 | Task11、Task14 |
| validation_records | DataFrame | 本 Task 正常完成时产生的校验记录 | 完整字段结构见下表 | 主函数追加至 validation_report |
##### `df_market_snapshot` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| instrument_code | string | 指数或股票代码 | 带市场后缀 |
| instrument_type | string | 标的类型 | INDEX / STOCK |
| build_date | date | 构建日 | YYYY-MM-DD |
| authoritative_close | float | 拟合采用的未复权收盘价 | 正数 |
| primary_source | string | 权威来源 | XTQUANT / NAS |
| secondary_source | string | 对照来源 | IFIND |
| primary_close | float | 权威源收盘价 | 正数 |
| secondary_close | float | 对照源收盘价 | 正数 |
| close_diff_pct | float | 收盘价相对偏差 | 小数 |
| validation_status | string | 校验结果 | PASS / FAIL |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `validation_records` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| check_name | string | 稳定校验名称 | 英文 snake_case |
| level | string | 严重级别 | ERROR / WARNING / INFO |
| status | string | 校验状态 | PASS / FAIL |
| actual_value | string | 实际结果 | 标量或 JSON 文本 |
| expected_value | string | 期望结果 | 标量或 JSON 文本 |
| tolerance | string | 允许误差 | 无则为空 |
| message | string | 说明 |  |
| checked_at | datetime | 校验时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task10：拟合前完整性校验关卡

Notebook 实现：

```python
run_validation_gate(validation_report, df_index_weights, df_component_rules, df_market_snapshot, debug=False)
```

任务目标：
在开始拟合计算前执行统一完整性关卡，确认所有强制数据交付和累计校验均已通过。

校验与失败规则：
- 成分股权重、交易规则与价格的代码集合必须完全一致，且每只股票的关键值唯一有效。
- 累计 `validation_report` 中不得存在阻断级失败。
- 校验失败时保存当前报告与已有证据，并抛出 `ManualReviewRequired`；通过后返回最终完整校验报告。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| validation_report | DataFrame | 主函数累计的完整校验报告 | 完整字段结构见下表 | 主函数累计 Task3、4、7、8、9 的 validation_records |
| df_index_weights | DataFrame | 最新公开样本权重 | 完整字段结构见下表 | Task3 输出 df_index_weights |
| df_component_rules | DataFrame | 成分股板块与交易规则匹配表 | 完整字段结构见下表 | Task8 输出 df_component_rules |
| df_market_snapshot | DataFrame | 构建日行情标准表 | 完整字段结构见下表 | Task9 输出 df_market_snapshot |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
##### `validation_records` 输入（`validation_report`）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| check_name | string | 稳定校验名称 | 英文 snake_case |
| level | string | 严重级别 | ERROR / WARNING / INFO |
| status | string | 校验状态 | PASS / FAIL |
| actual_value | string | 实际结果 | 标量或 JSON 文本 |
| expected_value | string | 期望结果 | 标量或 JSON 文本 |
| tolerance | string | 允许误差 | 无则为空 |
| message | string | 说明 |  |
| checked_at | datetime | 校验时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_index_weights` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| index_code | string | 目标指数代码 | 六位代码 |
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| source_file | string | 来源文件 | 文件名 |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_component_rules` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| board | string | 板块 | MAIN / STAR / CHINEXT |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| rule_source_url | string | 官方规则来源 | URL |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_market_snapshot` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| instrument_code | string | 指数或股票代码 | 带市场后缀 |
| instrument_type | string | 标的类型 | INDEX / STOCK |
| build_date | date | 构建日 | YYYY-MM-DD |
| authoritative_close | float | 拟合采用的未复权收盘价 | 正数 |
| primary_source | string | 权威来源 | XTQUANT / NAS |
| secondary_source | string | 对照来源 | IFIND |
| primary_close | float | 权威源收盘价 | 正数 |
| secondary_close | float | 对照源收盘价 | 正数 |
| close_diff_pct | float | 收盘价相对偏差 | 小数 |
| validation_status | string | 校验结果 | PASS / FAIL |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| validation_report | DataFrame | 追加完整性检查记录后的完整校验报告 | 完整字段结构见下表 | Task14、主函数返回值 |
##### `validation_records` 输出（`validation_report`）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| check_name | string | 稳定校验名称 | 英文 snake_case |
| level | string | 严重级别 | ERROR / WARNING / INFO |
| status | string | 校验状态 | PASS / FAIL |
| actual_value | string | 实际结果 | 标量或 JSON 文本 |
| expected_value | string | 期望结果 | 标量或 JSON 文本 |
| tolerance | string | 允许误差 | 无则为空 |
| message | string | 说明 |  |
| checked_at | datetime | 校验时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task11：计算理论股票篮子

Notebook 实现：

```python
calculate_theoretical_portfolio(df_index_weights, df_component_rules, df_market_snapshot, index_close, index_units, contract_multiplier, build_date, import_time, debug=False)
```

任务目标：
按照指数原始样本权重和构建日收盘价，计算尚未应用交易数量规则的理论股票篮子。

处理规则：
- `target_stock_value = index_close × contract_multiplier × index_units`。
- `raw_weight = raw_weight_pct / 100`，且原始权重不得归一化。
- `theoretical_amount = target_stock_value × raw_weight`。
- `theoretical_qty = theoretical_amount / close_price`。
- 将交易数量规则字段并入理论篮子，并写入本次运行目录。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| df_index_weights | DataFrame | 最新公开样本权重 | 完整字段结构见下表 | Task3 输出 df_index_weights |
| df_component_rules | DataFrame | 成分股板块与交易规则匹配表 | 完整字段结构见下表 | Task8 输出 df_component_rules |
| df_market_snapshot | DataFrame | 构建日行情标准表 | 完整字段结构见下表 | Task9 输出 df_market_snapshot |
| index_close | float | 采用的指数收盘点位 | 正数 | Task9 输出 index_close |
| index_units | int | 指数单位数 | 正整数 | 主函数输入 |
| contract_multiplier | float | 合约乘数 | 正数 | 主函数输入 |
| build_date | date | 构建日 | YYYY-MM-DD | Task2 输出 index_context.build_date |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM | Task0 输出 import_time |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
##### `df_index_weights` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| index_code | string | 目标指数代码 | 六位代码 |
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| source_file | string | 来源文件 | 文件名 |
| retrieved_at | datetime | 获取时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_component_rules` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| board | string | 板块 | MAIN / STAR / CHINEXT |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| rule_source_url | string | 官方规则来源 | URL |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_market_snapshot` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| instrument_code | string | 指数或股票代码 | 带市场后缀 |
| instrument_type | string | 标的类型 | INDEX / STOCK |
| build_date | date | 构建日 | YYYY-MM-DD |
| authoritative_close | float | 拟合采用的未复权收盘价 | 正数 |
| primary_source | string | 权威来源 | XTQUANT / NAS |
| secondary_source | string | 对照来源 | IFIND |
| primary_close | float | 权威源收盘价 | 正数 |
| secondary_close | float | 对照源收盘价 | 正数 |
| close_diff_pct | float | 收盘价相对偏差 | 小数 |
| validation_status | string | 校验结果 | PASS / FAIL |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| df_theoretical_portfolio | DataFrame | 未考虑交易数量规则的理论股票篮子 | 完整字段结构见下表 | Task12、Task13、Task14 |
| target_stock_value | float | 股票篮子目标总金额 | 人民币元 | Task13、Task14 |
##### `df_theoretical_portfolio` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| board | string | 板块 | MAIN / STAR / CHINEXT |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| raw_weight | float | 用于计算的小数权重 | raw_weight_pct / 100 |
| close_price | float | NAS 权威未复权收盘价 | 正数 |
| target_stock_value | float | 股票篮子目标总金额 | 人民币元 |
| theoretical_amount | float | 单只股票理论目标金额 | 人民币元 |
| theoretical_qty | float | 未取整理论数量 | 股 |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| build_date | date | 构建日 | YYYY-MM-DD |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task12：使用两步贪心法生成目标股票篮子

Notebook 实现：

```python
build_target_portfolio_greedy(df_theoretical_portfolio, debug=False)
```

任务目标：
使用透明、确定且可复现的两步贪心算法，将理论数量转换为符合交易数量规则且不超预算的目标股票篮子。

处理规则：
- 单只股票合法数量为 `0`，或达到 `buy_min_qty` 后按 `buy_qty_step` 递增。
- 第一步对每只股票向下选择不超过理论数量的最大合法数量。
- 第二步使用剩余资金逐步补入合法数量；每次补入必须严格降低该股票金额误差。
- 固定候选排序和并列规则，确保相同输入产生相同结果。
- 将目标股票篮子写入本次运行目录。

校验与失败规则：
- 组合总市值不得超过 `target_stock_value`。
- 所有非零目标数量必须满足对应交易数量规则，且每次贪心补入必须严格改善金额误差；否则阻断。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| df_theoretical_portfolio | DataFrame | 理论股票篮子 | 完整字段结构见下表 | Task11 输出 df_theoretical_portfolio |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
##### `df_theoretical_portfolio` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| board | string | 板块 | MAIN / STAR / CHINEXT |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| raw_weight | float | 用于计算的小数权重 | raw_weight_pct / 100 |
| close_price | float | NAS 权威未复权收盘价 | 正数 |
| target_stock_value | float | 股票篮子目标总金额 | 人民币元 |
| theoretical_amount | float | 单只股票理论目标金额 | 人民币元 |
| theoretical_qty | float | 未取整理论数量 | 股 |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| build_date | date | 构建日 | YYYY-MM-DD |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| df_target_portfolio | DataFrame | 两步贪心法生成的目标股票篮子 | 完整字段结构见下表 | Task13、Task14 |
##### `df_target_portfolio` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| close_price | float | 权威未复权收盘价 | 正数 |
| theoretical_amount | float | 单只股票理论目标金额 | 人民币元 |
| theoretical_qty | float | 未取整理论数量 | 股 |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| initial_floor_qty | int | 第一步得到的最大合法数量 | 股 |
| greedy_added_qty | int | 第二步累计增加数量 | 股 |
| target_qty | int | 最终目标数量 | 股 |
| target_market_value | float | 最终目标市值 | 人民币元 |
| is_held | bool | 最终是否持有 | true / false |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task13：计算静态拟合偏差

Notebook 实现：

```python
calculate_static_deviation(df_theoretical_portfolio, df_target_portfolio, target_stock_value, import_time, debug=False)
```

任务目标：
量化理论股票篮子与目标股票篮子之间的数量、金额和权重偏差，并形成组合级拟合摘要。

处理规则：
- `qty_deviation = target_qty - theoretical_qty`。
- `amount_deviation = target_market_value - theoretical_amount`，`absolute_amount_error` 为其绝对值。
- `theoretical_weight = theoretical_amount / target_stock_value`。
- `target_funding_weight = target_market_value / target_stock_value`。
- `target_invested_weight = target_market_value / target_portfolio_market_value`；目标组合市值为零时返回空值。
- 汇总剩余现金、投资比例、金额误差和持股数量等组合级指标，并将偏差报告与组合摘要写入本次运行目录。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| df_theoretical_portfolio | DataFrame | 理论股票篮子 | 完整字段结构见下表 | Task11 输出 df_theoretical_portfolio |
| df_target_portfolio | DataFrame | 目标股票篮子 | 完整字段结构见下表 | Task12 输出 df_target_portfolio |
| target_stock_value | float | 股票篮子目标总金额 | 人民币元 | Task11 输出 target_stock_value |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM | Task0 输出 import_time |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
##### `df_theoretical_portfolio` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| board | string | 板块 | MAIN / STAR / CHINEXT |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| raw_weight | float | 用于计算的小数权重 | raw_weight_pct / 100 |
| close_price | float | NAS 权威未复权收盘价 | 正数 |
| target_stock_value | float | 股票篮子目标总金额 | 人民币元 |
| theoretical_amount | float | 单只股票理论目标金额 | 人民币元 |
| theoretical_qty | float | 未取整理论数量 | 股 |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| build_date | date | 构建日 | YYYY-MM-DD |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_target_portfolio` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| close_price | float | 权威未复权收盘价 | 正数 |
| theoretical_amount | float | 单只股票理论目标金额 | 人民币元 |
| theoretical_qty | float | 未取整理论数量 | 股 |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| initial_floor_qty | int | 第一步得到的最大合法数量 | 股 |
| greedy_added_qty | int | 第二步累计增加数量 | 股 |
| target_qty | int | 最终目标数量 | 股 |
| target_market_value | float | 最终目标市值 | 人民币元 |
| is_held | bool | 最终是否持有 | true / false |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| df_deviation_report | DataFrame | 每只股票的静态拟合偏差 | 完整字段结构见下表 | Task14、主函数返回值 |
| portfolio_summary | dict | 组合级拟合评价摘要 | 完整字段结构见下表 | Task14、主函数返回值 |
##### `df_deviation_report` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| theoretical_qty | float | 理论数量 | 股 |
| target_qty | int | 目标数量 | 股 |
| qty_deviation | float | 目标数量减理论数量 | 股 |
| theoretical_amount | float | 理论目标金额 | 人民币元 |
| target_market_value | float | 目标市值 | 人民币元 |
| amount_deviation | float | 目标市值减理论金额 | 人民币元 |
| absolute_amount_error | float | 金额偏差绝对值 | 人民币元 |
| theoretical_weight | float | 理论资金口径权重 | 小数 |
| target_funding_weight | float | 目标资金口径权重 | 小数 |
| funding_weight_deviation | float | 资金口径权重偏差 | 小数 |
| target_invested_weight | float 或 null | 已投资口径权重 | 小数 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `portfolio_summary` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| target_stock_value | float | 股票篮子目标总金额 | 人民币元 |
| target_portfolio_market_value | float | 目标篮子总市值 | 人民币元 |
| remaining_cash | float | 剩余现金 | 人民币元 |
| invested_ratio | float | 已投资金额占目标金额比例 | 小数 |
| total_absolute_amount_error | float | 逐股票金额误差绝对值总和 | 人民币元 |
| max_absolute_amount_error | float | 最大单股金额误差 | 人民币元 |
| held_stock_count | int | 最终持有股票数量 | 只 |
| zero_qty_stock_count | int | 最终数量为零的股票数量 | 只 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |

### Task14：生成输出报告

Notebook 实现：

```python
generate_reports(index_code, build_date, import_time, primary_etf_code, index_units, contract_multiplier, index_close, target_stock_value, validation_report, theoretical_portfolio, target_portfolio, deviation_report, portfolio_summary, output_paths, started_at, debug=False)
```

任务目标：
从已落盘中间产物和核心计算结果生成最终可审计报告，并返回本次运行摘要。

处理规则：
- 将核心报告写入 Task0 已规划的路径，生成规定的 CSV 文件与多工作表 Excel 报告。
- 不得修改 `output_paths` 字典；需要汇总中间产物时，只能从 `output_paths` 指向的已落盘文件读取。
- `report_xlsx` 的工作表和字段必须符合下方约定。

校验与失败规则：
- 报告生成后必须检查所有最终必需文件确实存在；缺失或写入失败时阻断。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| index_code | string | 目标指数代码 | 六位代码 | Task2 输出 index_context.index_code |
| build_date | date | 构建日 | YYYY-MM-DD | Task2 输出 index_context.build_date |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM | Task0 输出 import_time |
| primary_etf_code | string | 最终主 ETF 代码 | 带市场后缀 | Task4 输出 resolved_primary_etf_code |
| index_units | int | 指数单位数 | 正整数 | 主函数输入 |
| contract_multiplier | float | 合约乘数 | 正数 | 主函数输入 |
| index_close | float | 采用的指数收盘点位 | 正数 | Task9 输出 index_close |
| target_stock_value | float | 股票篮子目标总金额 | 人民币元 | Task11 输出 target_stock_value |
| validation_report | DataFrame | 完整校验报告 | 完整字段结构见下表 | Task10 输出 validation_report |
| theoretical_portfolio | DataFrame | 理论股票篮子 | 完整字段结构见下表 | Task11 输出 df_theoretical_portfolio |
| target_portfolio | DataFrame | 目标股票篮子 | 完整字段结构见下表 | Task12 输出 df_target_portfolio |
| deviation_report | DataFrame | 静态拟合偏差 | 完整字段结构见下表 | Task13 输出 df_deviation_report |
| portfolio_summary | dict | 组合级拟合评价摘要 | 完整字段结构见下表 | Task13 输出 portfolio_summary |
| output_paths | dict | Task0 预先规划且只读的全部路径 | 完整字段结构见下表 | Task0 输出 output_paths |
| started_at | datetime | 主流程开始时间 | ISO 8601，含时区 | Task0 输出 started_at |
| debug | bool | 是否启用调试模式 | 默认 false | 主函数输入 |
##### `validation_records` 输入（`validation_report`）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| check_name | string | 稳定校验名称 | 英文 snake_case |
| level | string | 严重级别 | ERROR / WARNING / INFO |
| status | string | 校验状态 | PASS / FAIL |
| actual_value | string | 实际结果 | 标量或 JSON 文本 |
| expected_value | string | 期望结果 | 标量或 JSON 文本 |
| tolerance | string | 允许误差 | 无则为空 |
| message | string | 说明 |  |
| checked_at | datetime | 校验时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_theoretical_portfolio` 输入（`theoretical_portfolio`）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| board | string | 板块 | MAIN / STAR / CHINEXT |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| raw_weight | float | 用于计算的小数权重 | raw_weight_pct / 100 |
| close_price | float | NAS 权威未复权收盘价 | 正数 |
| target_stock_value | float | 股票篮子目标总金额 | 人民币元 |
| theoretical_amount | float | 单只股票理论目标金额 | 人民币元 |
| theoretical_qty | float | 未取整理论数量 | 股 |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| build_date | date | 构建日 | YYYY-MM-DD |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_target_portfolio` 输入（`target_portfolio`）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| close_price | float | 权威未复权收盘价 | 正数 |
| theoretical_amount | float | 单只股票理论目标金额 | 人民币元 |
| theoretical_qty | float | 未取整理论数量 | 股 |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| initial_floor_qty | int | 第一步得到的最大合法数量 | 股 |
| greedy_added_qty | int | 第二步累计增加数量 | 股 |
| target_qty | int | 最终目标数量 | 股 |
| target_market_value | float | 最终目标市值 | 人民币元 |
| is_held | bool | 最终是否持有 | true / false |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_deviation_report` 输入（`deviation_report`）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| theoretical_qty | float | 理论数量 | 股 |
| target_qty | int | 目标数量 | 股 |
| qty_deviation | float | 目标数量减理论数量 | 股 |
| theoretical_amount | float | 理论目标金额 | 人民币元 |
| target_market_value | float | 目标市值 | 人民币元 |
| amount_deviation | float | 目标市值减理论金额 | 人民币元 |
| absolute_amount_error | float | 金额偏差绝对值 | 人民币元 |
| theoretical_weight | float | 理论资金口径权重 | 小数 |
| target_funding_weight | float | 目标资金口径权重 | 小数 |
| funding_weight_deviation | float | 资金口径权重偏差 | 小数 |
| target_invested_weight | float 或 null | 已投资口径权重 | 小数 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `portfolio_summary` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| target_stock_value | float | 股票篮子目标总金额 | 人民币元 |
| target_portfolio_market_value | float | 目标篮子总市值 | 人民币元 |
| remaining_cash | float | 剩余现金 | 人民币元 |
| invested_ratio | float | 已投资金额占目标金额比例 | 小数 |
| total_absolute_amount_error | float | 逐股票金额误差绝对值总和 | 人民币元 |
| max_absolute_amount_error | float | 最大单股金额误差 | 人民币元 |
| held_stock_count | int | 最终持有股票数量 | 只 |
| zero_qty_stock_count | int | 最终数量为零的股票数量 | 只 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `output_paths` 输入字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| run_dir | string | 本次运行根目录 | 绝对路径 |
| index_dir | string | 指数权重、成分及校验文件目录 | 绝对路径 |
| etf_dir | string | 相关 ETF 与主 ETF PCF 文件目录 | 绝对路径 |
| corporate_actions_dir | string | 公司行为文件目录 | 绝对路径 |
| market_dir | string | 构建日行情文件目录 | 绝对路径 |
| trading_status_dir | string | 构建日交易状态文件目录 | 绝对路径 |
| trading_rules_dir | string | 本次运行交易规则文件目录 | 绝对路径 |
| reports_dir | string | 最终报告目录 | 绝对路径 |
| logs_dir | string | 日志目录 | 绝对路径 |
| log_path | string | 本次运行日志文件 | 绝对路径，`.log` |
| index_weights_csv | string | 标准化指数权重文件 | 绝对路径，`.csv` |
| index_cons_csv | string | 标准化指数成分文件 | 绝对路径，`.csv` |
| index_constituents_check_csv | string | 指数双文件成分校验文件 | 绝对路径，`.csv` |
| related_etfs_csv | string | 目标指数相关 ETF 清单 | 绝对路径，`.csv` |
| sse_etf_components_csv | string | 上交所主 ETF PCF | 绝对路径，`.csv` |
| xt_etf_components_csv | string | 迅投主 ETF PCF | 绝对路径，`.csv` |
| etf_cross_check_csv | string | 主 ETF PCF 对照结果 | 绝对路径，`.csv` |
| corporate_actions_xt_csv | string | 迅投公司行为标准化表 | 绝对路径，`.csv` |
| corporate_actions_ifind_csv | string | 同花顺公司行为标准化表 | 绝对路径，`.csv` |
| corporate_actions_merged_csv | string | 合并公司行为表 | 绝对路径，`.csv` |
| corporate_action_failures_csv | string | 公司行为接口异常清单 | 绝对路径，`.csv` |
| trading_status_csv | string | 构建日交易状态表 | 绝对路径，`.csv` |
| security_rules_csv | string | 本次运行交易规则快照 | 绝对路径，`.csv` |
| rule_parser_candidates_csv | string | 交易规则解析候选表 | 绝对路径，`.csv` |
| component_rules_csv | string | 成分股板块与交易规则匹配表 | 绝对路径，`.csv` |
| market_snapshot_csv | string | 构建日行情标准表 | 绝对路径，`.csv` |
| report_xlsx | string | Excel 汇总报告 | 绝对路径，`.xlsx` |
| run_summary_csv | string | 运行摘要文件 | 绝对路径，`.csv` |
| validation_report_csv | string | 完整校验报告文件 | 绝对路径，`.csv` |
| theoretical_portfolio_csv | string | 理论股票篮子文件 | 绝对路径，`.csv` |
| target_portfolio_csv | string | 目标股票篮子文件 | 绝对路径，`.csv` |
| deviation_report_csv | string | 静态偏差报告文件 | 绝对路径，`.csv` |
| portfolio_summary_csv | string | 组合摘要文件 | 绝对路径，`.csv` |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| run_summary | DataFrame | 本次运行参数、状态与结果摘要 | 完整字段结构见下表 | 主函数返回值 |
| finished_at | datetime | 报告生成完成时间 | ISO 8601，含时区 | 仅用于 run_summary 与日志 |
##### `run_summary` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| index_code | string | 目标指数代码 | 六位代码 |
| build_date | date | 构建日 | YYYY-MM-DD |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
| primary_etf_code | string | 主 ETF 代码 | 带市场后缀 |
| index_units | int | 指数单位数 | 正整数 |
| contract_multiplier | float | 合约乘数 | 正数 |
| index_close | float | 采用的指数收盘点位 | 正数 |
| target_stock_value | float | 目标股票金额 | 人民币元 |
| target_portfolio_market_value | float | 目标篮子总市值 | 人民币元 |
| remaining_cash | float | 剩余现金 | 人民币元 |
| held_stock_count | int | 最终持有股票数量 | 只 |
| validation_error_count | int | 失败校验数量 | 非负整数 |
| started_at | datetime | 主流程开始时间 | ISO 8601，含时区 |
| finished_at | datetime | 主流程结束时间 | ISO 8601，含时区 |
| status | string | 运行状态 | SUCCESS / REVIEW_REQUIRED |
| debug | bool | 是否启用调试模式 | true / false |

`report_xlsx` 必须包含以下工作表，工作表字段与本 Task 输入或输出中同名结构的字段表完全一致：

| 工作表名称 | 内容结构 | 来源 |
|---|---|---|
| run_summary | `run_summary` | Task14 输出 |
| validation_report | `validation_records` | Task10 输出 validation_report |
| theoretical_portfolio | `df_theoretical_portfolio` | Task11 输出 |
| target_portfolio | `df_target_portfolio` | Task12 输出 |
| deviation_report | `df_deviation_report` | Task13 输出 |
| portfolio_summary | `portfolio_summary` | Task13 输出，按单行表写入 |

### Task15：主函数串联完整流程

Notebook 实现：

```python
run_index_fitting(index_code, index_units, contract_multiplier, primary_etf_code=None, refresh_trade_rules=False, debug=False)
```

任务目标：
按照固定顺序串联 Task0 至 Task14，统一管理校验记录、阻断失败和最终返回结果。

处理规则：
- 严格按照 Task0 至 Task14 的整数序号顺序调用，并显式传递上一步输出。
- 维护累计 `validation_report`；仅追加 Task3、Task4、Task7、Task8、Task9 正常返回的 `validation_records`，Task10 返回最终完整校验报告。
- 仅在全部强制数据交付和校验通过后计算理论篮子与目标篮子。

校验与失败规则：
- 任一 Task 阻断失败时，将失败记录追加到累计报告，保存当前报告与证据，并抛出 `ManualReviewRequired`，不正常返回。

#### 输入标准表

| 参数名称 | 参数类型 | 参数含义 | 输入格式 | 来源 |
|---|---|---|---|---|
| index_code | string | 目标指数代码 | 必填，六位代码 | 用户输入 |
| index_units | int | 用于确定拟合规模的指数单位数 | 必填，正整数 | 用户输入 |
| contract_multiplier | float | 用户指定的标准合约乘数 | 必填，正数 | 用户输入 |
| primary_etf_code | string 或 null | 用于 PCF 严格验证的主 ETF | 带市场后缀；上证50为空时默认 510050.SH | 用户输入 |
| refresh_trade_rules | bool | 是否强制重新爬取官方交易规则 | 默认 false | 用户输入 |
| debug | bool | 是否启用调试模式 | 默认 false | 用户输入 |
#### 输出标准表

| 输出名称 | 输出类型 | 输出含义 | 输出格式 | 使用方 |
|---|---|---|---|---|
| run_summary | DataFrame | 本次运行摘要 | 完整字段结构见下表 | 用户 |
| validation_report | DataFrame | 完整校验报告 | 完整字段结构见下表 | 用户 |
| theoretical_portfolio | DataFrame | 理论股票篮子 | 完整字段结构见下表 | 用户 |
| target_portfolio | DataFrame | 目标股票篮子 | 完整字段结构见下表 | 用户 |
| deviation_report | DataFrame | 静态拟合偏差 | 完整字段结构见下表 | 用户 |
| portfolio_summary | dict | 组合级拟合评价摘要 | 完整字段结构见下表 | 用户 |
| output_paths | dict | Task0 预先规划的目录与关键文件路径 | 完整字段结构见下表 | 用户 |
##### `run_summary` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| index_code | string | 目标指数代码 | 六位代码 |
| build_date | date | 构建日 | YYYY-MM-DD |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
| primary_etf_code | string | 主 ETF 代码 | 带市场后缀 |
| index_units | int | 指数单位数 | 正整数 |
| contract_multiplier | float | 合约乘数 | 正数 |
| index_close | float | 采用的指数收盘点位 | 正数 |
| target_stock_value | float | 目标股票金额 | 人民币元 |
| target_portfolio_market_value | float | 目标篮子总市值 | 人民币元 |
| remaining_cash | float | 剩余现金 | 人民币元 |
| held_stock_count | int | 最终持有股票数量 | 只 |
| validation_error_count | int | 失败校验数量 | 非负整数 |
| started_at | datetime | 主流程开始时间 | ISO 8601，含时区 |
| finished_at | datetime | 主流程结束时间 | ISO 8601，含时区 |
| status | string | 运行状态 | SUCCESS / REVIEW_REQUIRED |
| debug | bool | 是否启用调试模式 | true / false |
##### `validation_records` 输出（`validation_report`）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| check_name | string | 稳定校验名称 | 英文 snake_case |
| level | string | 严重级别 | ERROR / WARNING / INFO |
| status | string | 校验状态 | PASS / FAIL |
| actual_value | string | 实际结果 | 标量或 JSON 文本 |
| expected_value | string | 期望结果 | 标量或 JSON 文本 |
| tolerance | string | 允许误差 | 无则为空 |
| message | string | 说明 |  |
| checked_at | datetime | 校验时间 | ISO 8601，含时区 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_theoretical_portfolio` 输出（`theoretical_portfolio`）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| exchange | string | 交易所 | SH / SZ |
| board | string | 板块 | MAIN / STAR / CHINEXT |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| raw_weight | float | 用于计算的小数权重 | raw_weight_pct / 100 |
| close_price | float | NAS 权威未复权收盘价 | 正数 |
| target_stock_value | float | 股票篮子目标总金额 | 人民币元 |
| theoretical_amount | float | 单只股票理论目标金额 | 人民币元 |
| theoretical_qty | float | 未取整理论数量 | 股 |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| build_date | date | 构建日 | YYYY-MM-DD |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_target_portfolio` 输出（`target_portfolio`）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| stock_name | string | 成分股名称 |  |
| raw_weight_pct | float | 原始百分数权重 | 不归一化 |
| close_price | float | 权威未复权收盘价 | 正数 |
| theoretical_amount | float | 单只股票理论目标金额 | 人民币元 |
| theoretical_qty | float | 未取整理论数量 | 股 |
| buy_min_qty | int | 最低首次买入数量 | 股 |
| buy_qty_step | int | 后续递增步长 | 股 |
| initial_floor_qty | int | 第一步得到的最大合法数量 | 股 |
| greedy_added_qty | int | 第二步累计增加数量 | 股 |
| target_qty | int | 最终目标数量 | 股 |
| target_market_value | float | 最终目标市值 | 人民币元 |
| is_held | bool | 最终是否持有 | true / false |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `df_deviation_report` 输出（`deviation_report`）字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| stock_code | string | 成分股代码 | 带市场后缀 |
| theoretical_qty | float | 理论数量 | 股 |
| target_qty | int | 目标数量 | 股 |
| qty_deviation | float | 目标数量减理论数量 | 股 |
| theoretical_amount | float | 理论目标金额 | 人民币元 |
| target_market_value | float | 目标市值 | 人民币元 |
| amount_deviation | float | 目标市值减理论金额 | 人民币元 |
| absolute_amount_error | float | 金额偏差绝对值 | 人民币元 |
| theoretical_weight | float | 理论资金口径权重 | 小数 |
| target_funding_weight | float | 目标资金口径权重 | 小数 |
| funding_weight_deviation | float | 资金口径权重偏差 | 小数 |
| target_invested_weight | float 或 null | 已投资口径权重 | 小数 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `portfolio_summary` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| target_stock_value | float | 股票篮子目标总金额 | 人民币元 |
| target_portfolio_market_value | float | 目标篮子总市值 | 人民币元 |
| remaining_cash | float | 剩余现金 | 人民币元 |
| invested_ratio | float | 已投资金额占目标金额比例 | 小数 |
| total_absolute_amount_error | float | 逐股票金额误差绝对值总和 | 人民币元 |
| max_absolute_amount_error | float | 最大单股金额误差 | 人民币元 |
| held_stock_count | int | 最终持有股票数量 | 只 |
| zero_qty_stock_count | int | 最终数量为零的股票数量 | 只 |
| import_time | string | 本次运行目录键 | YYYYMMDD-HHMM |
##### `output_paths` 输出字段结构

| 字段名称 | 字段类型 | 字段含义 | 格式 |
|---|---|---|---|
| run_dir | string | 本次运行根目录 | 绝对路径 |
| index_dir | string | 指数权重、成分及校验文件目录 | 绝对路径 |
| etf_dir | string | 相关 ETF 与主 ETF PCF 文件目录 | 绝对路径 |
| corporate_actions_dir | string | 公司行为文件目录 | 绝对路径 |
| market_dir | string | 构建日行情文件目录 | 绝对路径 |
| trading_status_dir | string | 构建日交易状态文件目录 | 绝对路径 |
| trading_rules_dir | string | 本次运行交易规则文件目录 | 绝对路径 |
| reports_dir | string | 最终报告目录 | 绝对路径 |
| logs_dir | string | 日志目录 | 绝对路径 |
| log_path | string | 本次运行日志文件 | 绝对路径，`.log` |
| index_weights_csv | string | 标准化指数权重文件 | 绝对路径，`.csv` |
| index_cons_csv | string | 标准化指数成分文件 | 绝对路径，`.csv` |
| index_constituents_check_csv | string | 指数双文件成分校验文件 | 绝对路径，`.csv` |
| related_etfs_csv | string | 目标指数相关 ETF 清单 | 绝对路径，`.csv` |
| sse_etf_components_csv | string | 上交所主 ETF PCF | 绝对路径，`.csv` |
| xt_etf_components_csv | string | 迅投主 ETF PCF | 绝对路径，`.csv` |
| etf_cross_check_csv | string | 主 ETF PCF 对照结果 | 绝对路径，`.csv` |
| corporate_actions_xt_csv | string | 迅投公司行为标准化表 | 绝对路径，`.csv` |
| corporate_actions_ifind_csv | string | 同花顺公司行为标准化表 | 绝对路径，`.csv` |
| corporate_actions_merged_csv | string | 合并公司行为表 | 绝对路径，`.csv` |
| corporate_action_failures_csv | string | 公司行为接口异常清单 | 绝对路径，`.csv` |
| trading_status_csv | string | 构建日交易状态表 | 绝对路径，`.csv` |
| security_rules_csv | string | 本次运行交易规则快照 | 绝对路径，`.csv` |
| rule_parser_candidates_csv | string | 交易规则解析候选表 | 绝对路径，`.csv` |
| component_rules_csv | string | 成分股板块与交易规则匹配表 | 绝对路径，`.csv` |
| market_snapshot_csv | string | 构建日行情标准表 | 绝对路径，`.csv` |
| report_xlsx | string | Excel 汇总报告 | 绝对路径，`.xlsx` |
| run_summary_csv | string | 运行摘要文件 | 绝对路径，`.csv` |
| validation_report_csv | string | 完整校验报告文件 | 绝对路径，`.csv` |
| theoretical_portfolio_csv | string | 理论股票篮子文件 | 绝对路径，`.csv` |
| target_portfolio_csv | string | 目标股票篮子文件 | 绝对路径，`.csv` |
| deviation_report_csv | string | 静态偏差报告文件 | 绝对路径，`.csv` |
| portfolio_summary_csv | string | 组合摘要文件 | 绝对路径，`.csv` |

## 九、Notebook 组织顺序

1. 标题、目标与边界。
2. 导入标准库、第三方库和 `index_fitting` 模块。
3. 参数配置。
4. Notebook 内部纯业务函数：统一校验、板块匹配、理论篮子、贪心算法、偏差和报告。
5. 主函数 `run_index_fitting`。
6. 默认调用主函数的最后一个单元。
7. 展示核心返回结果，不展示大型 PCF 或原始数据全表。

## 十、验收标准

- 实际生成 `股指拟合.ipynb` 和固定的 `index_fitting/` 模块。
- 所有 Task 为连续整数编号 `Task0` 至 `Task15`。
- 不存在小数序号或未编号 Task，也不得另设运行编号字段。
- 外部数据获取和爬虫均位于独立 `.py` 文件。
- Notebook JSON 有效，可从上到下顺序运行。
- 全部 Python 文件与 Notebook 代码通过静态语法检查。
- 主函数参数和返回值严格符合本文。
- 所有 Token 均从环境读取，且不写入日志。
- 主 ETF PCF、指数价格、成分股价格和交易规则均完成规定的阻断校验。
- 公司行为按规定合并，交易状态按规定保存。
- 两步贪心法满足合法数量、预算上限和可复现性要求。
- 关键失败保存证据并进入人工审核。
