# 股指单日静态拟合原型：AI程序生成说明

你是一名熟悉 Python、Jupyter Notebook、XtQuant/迅投接口、A股交易规则、公开数据爬取、指数复制和组合优化的高级量化开发工程师。

请根据本说明，在当前项目根目录中实际创建一个完整、可逐步运行、可独立调试的 Jupyter Notebook：

```text
股指拟合.ipynb
```

不要只提供设计方案、伪代码或独立 `.py` 文件。


## 一、第一版目标与边界

### 1. 核心目标

第一版只完成“单日静态拟合”：

在程序运行时，自动确定目标指数所属交易所的上一交易日，使用该日指数与成分股的未复权收盘价，以及程序运行时取得的交易所公开最新样本权重，构建满足交易数量规则的目标股票篮子，并分析理论股票篮子与目标股票篮子之间的静态拟合偏差。

目标股票金额计算公式：

```text
目标股票金额 = 上一交易日现货指数收盘点位 × contract_multiplier × index_units
```

其中：

- `index_units`：用户指定的指数单位数，仅用于确定拟合规模，不代表实际期货持仓。
- `contract_multiplier`：用户显式输入的标准合约乘数，程序不自动推断或交叉校验。
- 第一版固定拟合完整名义规模，不设置 `hedge_ratio`。

### 2. 第一版明确不做

第一版不实现以下内容：

- 不输入或处理具体股指期货合约代码。
- 不进行期货损益、基差变化或保证金计算。
- 不进行多日持有、历史回测或收益归因。
- 不模拟实际交易或实际成交。
- 不计算交易成本、滑点或市场冲击。
- 不考虑涨停、跌停和停牌状态。
- 不自动下单。
- 不使用 ETF 参与当前拟合。
- 不使用除权除息信息调整当前拟合。
- 不实现样本外股票中性化或复杂整数规划。
- 不生成图表。

### 3. 第一版必须采集但暂不参与拟合的数据

以下数据在第一版必须获取并保存，用于验证数据链路和后续扩展，但不参与目标股票数量计算：

- 全市场 ETF 申赎清单。
- 当前指数成分股在构建日前一个月内的除权除息因子。
- 当前指数成分股在构建日后一个月内可查询到的未来除权除息事件。


## 二、核心术语与业务口径

### 1. 构建日

构建日是严格早于程序运行自然日的最近交易日。

- 根据目标指数所属交易所的交易日历确定。
- 无论程序在盘前、盘中还是盘后运行，均使用上一交易日。
- 第一版不允许用户指定任意历史构建日。

### 2. 公开最新样本权重

- 程序运行时通过现有 `CSI.csiweb` 功能，从中证指数官网公开渠道下载最新样本权重文件。
- 样本权重是第一版唯一使用的指数权重来源。
- 不使用迅投 `get_index_weight()`。
- 权重文件名中的日期是下载日期，不是权重生效日期。
- 权重直接使用公开文件中的原始值，不做归一化。

### 3. 理论股票篮子

完全按照目标股票金额、原始样本权重和构建日未复权收盘价计算，尚未考虑交易数量规则的股票及数量集合。

### 4. 目标股票篮子

经过交易数量规则处理和第一版两步贪心算法后，计划持有的股票及数量集合。

- 目标数量允许为 `0`。
- 不要求持有全部指数成分股。
- 组合总市值不得超过目标股票金额。

### 5. 静态拟合偏差

理论股票篮子与目标股票篮子之间的数量、金额和权重差异。

第一版不包含实际执行偏差和交易成本。


## 三、主函数接口

主函数名称：

```python
run_index_fitting(
    index_code: str,
    index_units: int,
    contract_multiplier: float,
    refresh_trade_rules: bool = False,
    debug: bool = False,
)
```

### 1. 主函数入参

- `index_code`：
    - 用户输入的六位指数代码，例如 `"000016"`。
    - 程序根据公开指数资料识别指数所属交易所，并生成迅投行情代码，例如 `"000016.SH"`。
    - 无法唯一确定指数所属交易所时，停止程序并进入人工审核。
- `index_units`：
    - 正整数。
    - 用于确定目标股票金额的指数单位数。
- `contract_multiplier`：
    - 用户显式输入的正数。
    - 程序不自动映射或推断。
- `refresh_trade_rules`：
    - `False`：优先使用本地交易规则快照。
    - `True`：重新从官方来源爬取并生成交易规则快照。
- `debug`：
    - 控制详细日志和调试中间产物。
    - 每一个 Task 函数都必须保留 `debug=False` 参数。

### 2. 主函数成功返回值

主函数成功时仅返回以下核心结果，不返回所有中间产物：

```python
{
    "run_summary": DataFrame,
    "validation_report": DataFrame,
    "theoretical_portfolio": DataFrame,
    "target_portfolio": DataFrame,
    "deviation_report": DataFrame,
    "portfolio_summary": dict,
    "output_paths": dict,
}
```

以下业务中间产物必须落盘保存，但不应作为主函数返回值：

- 指数样本权重。
- 指数成分校验结果。
- 行情快照。
- 交易规则。
- 全市场 ETF 汇总、成分和失败清单。
- 除权除息数据和失败清单。
- 原始下载文件。


## 四、程序总体实现原则

- 每一个执行 Task 都写成独立函数。
- 主函数按顺序调用 Task，并将输出显式传递给下一步。
- 禁止依赖执行顺序不明确的隐藏全局变量。
- 每个 Task 必须：
    - 具有完整中文 docstring。
    - 明确说明参数、返回值、数据口径、异常和限制。
    - 可以独立调用和调试。
    - 保留 `debug=False` 参数。
- 所有 DataFrame 字段名保持稳定。
- 所有金额使用人民币元，权重需明确百分数或小数口径。
- 所有日期统一使用 `YYYYMMDD` 或明确的日期类型。
- 禁止静默忽略关键异常。
- 禁止用旧价格、删除缺价股票或其他替代数据继续拟合。
- 禁止在 Notebook 中硬编码 Token、账号、密码或签名 URL。
- XtQuant Token 必须从环境变量读取。
- Notebook 必须可以从上到下顺序运行。
- Notebook 最后一个单元格默认调用完整主函数。


## 五、运行目录、日志与人工审核

### 1. 运行编号

每次运行生成：

```text
run_id = YYYYMMDD_HHMMSS
```

### 2. 运行目录

一次运行的全部产物必须收拢到：

```text
outputs/<run_id>/
├── report.xlsx
├── csv/
├── raw/
│   ├── csi/
│   ├── etf/
│   ├── corporate_actions/
│   └── trading_rules/
├── debug/                 # 仅 debug=True 时产生
└── run.log
```

### 3. 业务中间产物与调试中间产物

- 业务中间产物无论 `debug` 是否开启都保存。
- 调试中间产物仅在 `debug=True` 时保存到 `debug/`。
- `run.log` 始终生成；`debug=True` 时记录更详细内容。

### 4. 人工审核机制

定义自定义异常：

```python
class ManualReviewRequired(Exception):
    ...
```

关键异常发生时：

1. 保存已经取得的原始数据、业务中间产物、日志和校验报告。
2. 将运行状态设为 `REVIEW_REQUIRED`。
3. 抛出 `ManualReviewRequired`。
4. 异常对象至少携带：
    - `run_id`
    - `validation_report`
    - `output_paths`
5. 不允许吞掉异常继续执行。
6. 不支持从中间强制继续；人工修复问题后完整重跑。


## 六、可参考的现有文件

编写 Notebook 前，必须实际读取并参考：

- `fetch_xtquant_data.py`
- `XtQuant.ipynb`
- `CSI/main.py`
- `CSI/csiweb/downloader.py`
- `CSI/csiweb/reader.py`
- `CSI/README.md`

已知可复用能力：

- 可以直接导入：

```python
from CSI.csiweb import download_csi_constituent, read_csi_file
```

- 不运行 `CSI/main.py`。
- 新编写的其他核心逻辑全部放在 `股指拟合.ipynb` 中，不额外创建新的 Python 模块。

已验证的迅投思路：

- 使用 `xtdata.download_etf_info()` 和 `xtdata.get_etf_info()` 获取全市场 ETF 申赎清单。
- 使用 `xtdata.get_divid_factors()` 获取除权除息因子。
- 使用迅投历史行情下载与查询接口获取指数和股票行情。
- 使用迅投交易日历接口确定上一交易日。
- 使用迅投证券详情辅助交叉检查股票所属交易所和板块。

对无法确认的 XtQuant 接口或返回字段，不得凭空编造；必须通过现有文件、实际环境或最小探测代码确认。


## 七、详细 Task 拆分

### Task0：初始化运行环境

- 函数名：

```python
initialize_runtime(output_root="outputs", debug=False)
```

- 输出：
    - `runtime_context` 字典，至少包含：
        - `run_id`
        - `run_status`
        - `run_dir`
        - `csv_dir`
        - `raw_dir`
        - `debug_dir`
        - `log_path`
        - `started_at`
- 要求：
    - 创建本次运行完整目录结构。
    - 配置日志。
    - 不覆盖历史运行结果。


### Task1：执行最小自测

- 函数名：

```python
run_minimum_self_tests(debug=False)
```

- 描述：
    - 使用虚构的小型数据测试纯计算逻辑。
    - 不调用迅投、网络或真实文件。
    - Notebook 从上到下运行时自动执行。
    - 任一测试失败则停止程序。
- 至少测试：
    - 合法目标数量判断。
    - 向下选择最大合法数量。
    - 理论数量低于 `buy_min_qty` 时初始数量为 `0`。
    - 首次补入使用 `buy_min_qty`。
    - 已有数量后按 `buy_qty_step` 补入。
    - 补入不能扩大逐股票金额误差。
    - 组合总市值不得超过目标股票金额。
    - 固定并列规则能产生可复现结果。
    - 权重合计误差超过 1 个百分点时产生 `ERROR`。
- 不要求为交易规则网页爬虫编写自动测试。


### Task2：初始化迅投连接

- 函数名：

```python
initialize_xtquant_connection(data_home="data", address_list=None, debug=False)
```

- 输出：
    - 连接状态字典。
- 要求：
    - 从环境变量读取 Token。
    - Token 不存在时抛出清晰异常。
    - 不记录 Token。
    - 默认关闭需要额外权限的 K 线全推功能。
    - 使用交易日期或证券详情接口执行最小连接检查。
    - 连接失败时触发人工审核暂停。


### Task3：下载并读取 CSI 样本权重和成分股文件

- 函数名：

```python
get_csi_index_files(index_code, raw_csi_dir, debug=False)
```

- 输入：
    - 六位 `index_code`。
    - 本次运行的 `raw/csi/` 目录。
- 输出：
    - `df_index_weights`
    - `df_index_cons`
    - 原始文件路径字典。
- 要求：
    - 每次运行都下载：
        - `download_type="closeweight"`
        - `download_type="cons"`
    - 显式将 `save_path` 指向本次运行的 `raw/csi/`。
    - 不使用共享 `CSI/csi_constituent/` 缓存。
    - 下载失败时不使用历史文件，立即暂停人工审核。
    - 使用 `read_csi_file()` 读取文件。
    - 保存下载来源、下载时间和实际文件路径。


### Task4：校验 CSI 权重与成分

- 函数名：

```python
validate_csi_index_data(df_index_weights, df_index_cons, debug=False)
```

- 输出：
    - `df_index_constituents_check`
    - 部分 `validation_report` 记录。
- 要求：
    - 比较 `closeweight` 和 `cons` 的成分代码集合。
    - 任何不一致均产生 `ERROR`：
        - 输出 `only_in_closeweight`
        - 输出 `only_in_cons`
        - 输出两份文件中的数据日期
        - 暂停并进入人工审核
    - 检查代码重复、空代码、权重空值和负权重。
    - 权重直接使用原始值，不归一化。
    - 权重合计与 100% 的绝对偏差：
        - 不超过 1 个百分点：记录 `WARNING`，允许继续。
        - 超过 1 个百分点：记录 `ERROR`，暂停人工审核。


### Task5：识别指数交易所并确定构建日

- 函数名：

```python
resolve_index_market_and_build_date(index_code, df_index_weights, debug=False)
```

- 输出：
    - `index_market_context` 字典，至少包含：
        - 六位指数代码
        - 迅投指数行情代码
        - 指数所属交易所
        - 程序运行日
        - 构建日
- 要求：
    - 根据公开指数资料识别指数所属交易所。
    - 无法唯一确定时暂停人工审核。
    - 根据目标指数所属交易所的迅投交易日历，确定严格早于运行自然日的最近交易日。


### Task6：刷新并保存全市场 ETF 申赎清单

- 函数名：

```python
get_all_etf_redemption_lists(raw_etf_dir, debug=False)
```

- 输出业务中间产物：
    - `df_etf_summary`
    - `df_etf_components`
    - `df_etf_failures`
    - 原始 JSON 文件路径。
- 要求：
    - 每次运行必须调用 `xtdata.download_etf_info()` 强制刷新。
    - 使用 `xtdata.get_etf_info()` 获取全市场 ETF 数据。
    - 不要求用户指定 ETF 代码。
    - 完整保存原始返回数据。
    - 接口整体失败、总表为空或原始数据无法保存：
        - 产生 `ERROR`
        - 暂停人工审核
    - 个别 ETF 的空清单、字段缺失或解析异常：
        - 写入 `df_etf_failures`
        - 不阻断当前拟合
    - 第一版 ETF 数据不参与拟合。


### Task7：查询并保存除权除息信息

- 函数名：

```python
get_component_corporate_actions(stock_codes, build_date, raw_actions_dir, debug=False)
```

- 输出业务中间产物：
    - `df_corporate_actions`
    - `df_corporate_action_failures`
- 查询范围：
    - 历史窗口：构建日前一个月至构建日。
    - 未来窗口：构建日后一个月。
- 要求：
    - 每次运行对全部当前指数成分股重新查询，不复用旧结果。
    - 复用 `fetch_xtquant_data.py` 中已验证的 `xtdata.get_divid_factors()` 查询方式。
    - 接口正常返回空表表示该股票在窗口内无事件，属于成功。
    - 未来窗口返回空表允许继续，不能据此声称未来无事件。
    - 任一股票查询出现接口异常或无法解析：
        - 写入失败清单
        - 产生 `ERROR`
        - 暂停人工审核
    - 第一版公司行为数据不参与拟合。


### Task8：加载或刷新官方交易数量规则

- 函数名：

```python
get_security_buy_rules(refresh_trade_rules=False, debug=False)
```

- 共享规则文件：

```text
data/trading_rules/security_buy_rules.csv
```

- 固定字段至少包括：

```text
exchange
board
buy_min_qty
buy_qty_step
rule_source_url
rule_effective_date
retrieved_at
```

- 第一版支持板块：
    - 上交所主板
    - 科创板
    - 深交所主板
    - 创业板
- 要求：
    - 不设置自动失效期限。
    - 本地文件存在且 `refresh_trade_rules=False` 时直接读取。
    - 本地文件不存在，或 `refresh_trade_rules=True` 时，实际执行官方规则网页爬虫。
    - 仅允许上交所、深交所等官方来源。
    - 爬虫保存官方原文和候选片段。
    - 仅当能唯一解析 `buy_min_qty` 和 `buy_qty_step` 时写入规则快照。
    - 无法唯一解析时暂停人工审核。
    - 不允许运行时临时手填规则绕过。
    - 将本次实际使用的规则快照复制到 `raw/trading_rules/`。


### Task9：识别成分股板块并匹配交易规则

- 函数名：

```python
classify_boards_and_match_rules(df_index_weights, df_security_rules, debug=False)
```

- 输出：
    - `df_component_rules`
- 要求：
    - 使用 CSI 文件中的交易所信息和证券代码规则识别板块。
    - 与迅投证券详情交叉检查。
    - 不允许只按代码前缀直接得出最终板块结论。
    - 任何冲突或无法识别产生 `ERROR` 并暂停人工审核。
    - 每只成分股必须唯一匹配交易规则。
    - 出现北交所或其他第一版不支持板块时暂停人工审核。


### Task10：刷新并获取构建日未复权收盘行情

- 函数名：

```python
get_build_date_close_snapshot(
    xt_index_code,
    stock_codes,
    build_date,
    debug=False,
)
```

- 输出：
    - `df_market_snapshot`
    - 指数收盘点位。
- 要求：
    - 每次运行先刷新目标指数和全部成分股的历史行情。
    - 使用构建日未复权收盘价。
    - 指数和全部成分股必须具有同一构建日的有效正数收盘数据。
    - 任一数据下载失败、缺失、为零、为负数或日期不一致时：
        - 产生 `ERROR`
        - 不使用旧价格填补
        - 不删除缺价股票
        - 不使用替代数据
        - 暂停人工审核


### Task11：统一输入校验关卡

- 函数名：

```python
validate_inputs(..., debug=False)
```

- 输出：

```text
validation_report[
    check_name,
    level,
    status,
    actual_value,
    expected_value,
    tolerance,
    message
]
```

- 校验级别：
    - `ERROR`：停止流程并进入人工审核。
    - `WARNING`：允许继续，但必须在运行摘要中突出展示。
    - `INFO`：仅记录。
- 至少汇总检查：
    - CSI 两份文件是否成功下载并读取。
    - `closeweight` 与 `cons` 成分集合是否一致。
    - 权重合计误差是否在 1 个百分点以内。
    - 指数代码、交易所和行情代码是否唯一明确。
    - 构建日是否成功确定。
    - 全市场 ETF 整体采集是否成功。
    - 除权除息查询是否对全部成分股成功完成。
    - 板块识别是否无冲突。
    - 每只成分股是否唯一匹配交易规则。
    - 指数和全部成分股是否具有完整构建日未复权收盘价。
    - `index_units` 是否为正整数。
    - `contract_multiplier` 是否为正数。
- 存在任一 `ERROR` 时，不得生成理论股票篮子或目标股票篮子。


### Task12：计算理论股票篮子

- 函数名：

```python
calculate_theoretical_portfolio(
    df_index_weights,
    df_market_snapshot,
    index_close,
    index_units,
    contract_multiplier,
    debug=False,
)
```

- 输出：
    - `df_theoretical_portfolio`
    - 理论组合摘要。
- 核心公式：

```text
target_stock_value = index_close × contract_multiplier × index_units
theoretical_amount_i = target_stock_value × raw_weight_i
theoretical_qty_i = theoretical_amount_i ÷ stock_close_i
```

- `df_theoretical_portfolio` 至少包含：

```text
stock_code
stock_name
exchange
board
raw_weight
close_price
target_stock_value
theoretical_amount
theoretical_qty
buy_min_qty
buy_qty_step
```

- 要求：
    - 使用原始样本权重，不归一化。
    - 明确权重字段是百分数还是小数，并统一转换。
    - 保留源字段和审计信息。


### Task13：使用两步贪心法生成目标股票篮子

- 函数名：

```python
build_target_portfolio_greedy(df_theoretical_portfolio, debug=False)
```

- 描述：
    - 拟合算法是整个程序的核心策略模块。
    - 第一版使用简单、透明、可解释的两步贪心法。
    - 函数边界必须清晰，未来可以替换为更精密的拟合算法，而不重写其他数据流程。

#### 1. 合法目标数量定义

单只股票合法目标数量为：

```text
qty = 0
```

或：

```text
qty >= buy_min_qty
且 qty 按 buy_qty_step 递增
```

#### 2. 第一步：向下选择最大合法数量

- 对每只股票，选择不超过 `theoretical_qty` 的最大合法数量。
- 如果理论数量低于 `buy_min_qty`，初始目标数量为 `0`。

#### 3. 第二步：逐步分配剩余资金

- 逐股票金额误差定义：

```text
total_absolute_amount_error
= Σ |target_market_value_i - theoretical_amount_i|
```

- 每轮尝试为一个股票增加下一合法增量：
    - 当前数量为 `0` 时，首次增加 `buy_min_qty`。
    - 当前数量大于 `0` 时，增加 `buy_qty_step`。
- 仅允许满足以下两个条件的补入：
    1. 补入后组合总市值不超过目标股票金额。
    2. 补入后严格降低逐股票金额误差绝对值总和。
- 当没有任何补入能够严格降低误差时停止。
- 剩余资金保留为现金。
- 不为了用尽现金而继续买入。

#### 4. 固定并列规则

多个补入候选改善相同时，依次按照以下规则选择：

1. 补入后逐股票金额误差总和更低。
2. 当前单票金额缺口更大。
3. 本次新增金额更小。
4. 股票代码字典序更小。

同一输入必须得到完全相同的结果。

#### 5. 调试输出

`debug=True` 时保存 `debug/greedy_steps.csv`，仅记录：

- 每次实际选中的补入操作。
- 补入前后误差。
- 新增数量与金额。
- 剩余现金。
- 最终停止原因。

不要保存每轮全部候选。


### Task14：计算静态拟合偏差

- 函数名：

```python
calculate_static_deviation(
    df_theoretical_portfolio,
    df_target_portfolio,
    target_stock_value,
    debug=False,
)
```

- 输出：
    - `df_deviation_report`
    - `portfolio_summary`

#### 1. 单股票层面至少包含

```text
stock_code
stock_name
raw_weight
close_price
theoretical_amount
theoretical_qty
target_qty
target_market_value
qty_deviation
amount_deviation
absolute_amount_deviation
fund_weight
invested_weight
weight_deviation
is_held
```

#### 2. 权重口径

```text
fund_weight = target_market_value / target_stock_value
```

- 与原始指数权重比较时使用 `fund_weight`。
- 现金余额需要体现为拟合偏差。

```text
invested_weight = target_market_value / target_portfolio_market_value
```

- `invested_weight` 仅用于观察已买股票内部结构。
- 不作为主要拟合偏差口径。

#### 3. 组合层面至少包含

```text
target_stock_value
target_portfolio_market_value
cash_balance
capital_utilization
total_absolute_amount_error
max_single_absolute_amount_error
active_share
held_stock_count
unheld_constituent_count
```

主动权重总量：

```text
active_share = 0.5 × Σ |fund_weight_i - raw_weight_i|
```


### Task15：生成输出报告

- 函数名：

```python
generate_index_fitting_report(results, runtime_context, debug=False)
```

- 输出：
    - `output_paths` 字典。
- 必须生成：
    - 一个多工作表 Excel 汇总文件 `report.xlsx`。
    - 各 DataFrame 的 CSV 明细文件。
    - 原始数据文件。
    - `run.log`。
- CSV 使用 `utf-8-sig` 编码，便于 Excel 打开。
- Excel 至少包含以下工作表：

```text
运行摘要
目标股票篮子
偏差报告
校验报告
指数样本权重
指数成分校验
行情快照
交易规则
ETF汇总
ETF成分
ETF失败清单
除权除息
除权除息失败清单
```

- Excel 工作表名称需要满足 Excel 长度限制。
- 不生成图表。


### Task16：主函数串联完整流程

- 函数名：

```python
run_index_fitting(
    index_code,
    index_units,
    contract_multiplier,
    refresh_trade_rules=False,
    debug=False,
)
```

- 固定执行顺序：

```text
初始化运行环境
→ 执行最小自测
→ 初始化迅投连接
→ 下载并读取 CSI closeweight 与 cons
→ 校验 CSI 权重和成分
→ 识别指数交易所并确定构建日
→ 强制刷新并保存全市场 ETF
→ 查询并保存全部成分股除权除息信息
→ 加载或刷新官方交易数量规则
→ 识别成分股板块并匹配交易规则
→ 刷新并读取构建日未复权收盘行情
→ 执行统一输入校验
→ 计算理论股票篮子
→ 执行两步贪心拟合
→ 计算静态拟合偏差
→ 生成报告
→ 返回核心结果
```

- 只有全部强制数据交付项和校验通过后，才允许生成理论股票篮子和目标股票篮子。


## 八、Notebook 结构要求

Notebook 必须按以下顺序组织：

1. Markdown：项目目标、业务口径、第一版范围和明确限制。
2. 环境配置说明：
    - Python 依赖。
    - XtQuant 连接要求。
    - 环境变量设置说明，但不展示真实 Token。
3. 导入依赖。
4. 通用类型、异常、日志、保存和 debug 辅助函数。
5. Task0 至 Task16 的函数定义；每个 Task 前有 Markdown 说明。
6. 最小自测单元格，并默认执行。
7. 集中参数配置单元格，仅要求用户填写：

```python
INDEX_CODE = "000016"
INDEX_UNITS = 14
CONTRACT_MULTIPLIER = 300
REFRESH_TRADE_RULES = False
DEBUG = False
```

8. 最后一个单元格默认调用：

```python
results = run_index_fitting(
    index_code=INDEX_CODE,
    index_units=INDEX_UNITS,
    contract_multiplier=CONTRACT_MULTIPLIER,
    refresh_trade_rules=REFRESH_TRADE_RULES,
    debug=DEBUG,
)
```

9. 展示主函数返回的核心结果表和摘要，不展示完整大型 ETF 明细。


## 九、关键校验与失败规则汇总

以下情况必须产生 `ERROR`、保存证据并抛出 `ManualReviewRequired`：

- XtQuant 连接失败。
- CSI `closeweight` 或 `cons` 当次下载或读取失败。
- `closeweight` 与 `cons` 成分代码集合存在任何不一致。
- 权重合计与 100% 的绝对偏差超过 1 个百分点。
- 指数所属交易所或迅投行情代码无法唯一确定。
- 无法确定上一交易日。
- 全市场 ETF 接口整体失败、总表为空或原始数据无法保存。
- 任一成分股除权除息查询出现接口异常或无法解析。
- 交易规则网页无法唯一解析。
- 成分股板块识别与迅投证券详情冲突。
- 任一成分股无法唯一匹配交易规则。
- 出现第一版不支持的板块。
- 指数或任一成分股行情刷新失败。
- 指数或任一成分股缺少构建日未复权收盘价。
- 任一关键价格为零、负数或日期错位。
- `index_units` 不是正整数。
- `contract_multiplier` 不是正数。
- 最小自测失败。

以下情况不应阻断流程：

- 权重合计与 100% 的绝对偏差不超过 1 个百分点：记录 `WARNING`。
- 单只 ETF 申赎清单为空或字段解析异常：记录 ETF 失败清单。
- 除权除息接口正常返回空表：记录无事件，不属于失败。
- 未来一个月除权除息事件为空：允许继续，不得声称未来无事件。
- 某成分股经拟合后目标数量为 `0`：属于合法结果。
- 贪心停止后存在现金余额：属于合法结果。


## 十、编码质量要求

- 使用 Python 类型注解。
- 使用清晰的中文 docstring 和必要注释。
- 优先使用：
    - `pandas`
    - `numpy`
    - `pathlib`
    - `logging`
    - `requests`
    - `json`
    - `datetime`
- 不需要使用 `matplotlib`。
- 避免把大段业务逻辑直接写在 Notebook 单元格顶层。
- 不使用无法解释的魔法数字。
- 对外部接口调用增加明确异常处理，不得使用空 `except`。
- 对失败证券和失败数据保留失败清单。
- 对关键计算增加断言或显式校验。
- 同一输入应得到可复现结果。
- 所有原始文件、来源 URL、抓取时间和使用版本可追溯。
- 业务中间产物必须保存，但不污染主函数返回值。
- 拟合算法必须封装为独立策略函数，便于未来替换。


## 十一、第一版必须回答的业务问题

程序最终结果必须能够回答：

1. 本次运行的构建日是哪一天？
2. 本次使用了哪两份 CSI 官方文件？
3. 样本权重合计是多少，是否存在允许范围内的误差？
4. 上一交易日指数收盘点位是多少？
5. 指定 `index_units` 和 `contract_multiplier` 对应多少目标股票金额？
6. 每只股票理论目标金额和理论数量是多少？
7. 每只股票适用的最低买入数量和买入数量步长是什么？
8. 两步贪心法生成的每只股票目标数量和目标市值是多少？
9. 哪些成分股目标数量为 `0`？
10. 目标股票篮子总市值、现金余额和资金使用率是多少？
11. 哪些股票造成最大的绝对金额偏差？
12. 逐股票金额误差绝对值总和是多少？
13. 主动权重总量是多少？
14. 全市场 ETF 和成分股除权除息数据是否成功完成采集？
15. 是否存在需要人工审核的异常或警告？


## 十二、验收标准

- 实际生成完整的 `股指拟合.ipynb`。
- Notebook JSON 格式有效。
- Notebook 可以从上到下顺序运行。
- 所有 Task 函数均包含 `debug=False` 参数。
- 每个 Task 可以独立调用。
- 最小自测默认执行并通过。
- 最后一个单元格默认执行完整主流程。
- 主函数仅暴露已确认的五个参数。
- 主函数仅返回已确认的七类核心结果。
- CSI 每次运行下载 `closeweight` 和 `cons`，并严格校验成分一致性。
- 不使用迅投指数权重。
- 使用目标指数所属交易所的上一交易日。
- 使用指数与全部成分股的构建日未复权收盘价。
- 全市场 ETF 每次运行强制刷新并保存。
- 除权除息每次运行对全部当前成分股重新查询并保存。
- 交易规则爬虫实际实现，且仅使用官方来源。
- 交易规则不能唯一解析或不能唯一匹配时暂停人工审核。
- 两步贪心算法严格遵守合法目标数量、误差改善、金额上限、停止条件和并列规则。
- 不生成模拟交易、交易成本、涨跌停、期货损益、回测或图表。
- 所有业务中间产物可审计。
- 关键失败会保存证据并抛出 `ManualReviewRequired`。
- 程序不包含任何真实 Token、账号、密码或签名 URL。


## 十三、生成程序时的执行要求

- 编写前先检查当前根目录中参考文件的真实内容和字段。
- 对无法确认的 XtQuant 接口，不得凭空编造。
- 对官方交易规则来源，必须实现实际爬取、保存原文和唯一解析检查。
- 完成后检查 Notebook JSON 格式。
- 对 Notebook 中全部 Python 代码做静态语法检查。
- 检查所有 Task 是否包含 `debug=False`。
- 检查主函数参数与返回值是否严格符合本文。
- 检查已删除功能是否被错误重新加入。
- 总结已实现功能、数据限制和需要人工审核的边界。
