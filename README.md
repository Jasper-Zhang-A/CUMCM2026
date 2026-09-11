# CUMCM 2026 C题 — Q2 Forecasting Baselines

当前阶段：采用用户冻结的区间结束标签和事件顺序协议，只使用附件2进行Load/PV单变量B0–B4与DLinear逐日预测。

项目现已迁移到 **D:/数学建模/CUMCM2026C_TS_Baseline**。最新DLinear阶段见
`DLINEAR_PROTOCOL.md`、`MODEL_SELECTION.md`、`MODEL_COMPARISON.md`，状态在
`artifacts/dlinear/run_status.json`。入口为 `python run_dlinear.py`；下文保留Seasonal阶段说明。

DLinear完整运行（包含测试、模型选择、月度重训、低数据实验、指标和图）：

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe run_dlinear.py
```

安装依赖及训练协议见 `DLINEAR_PROTOCOL.md`。只重建已通过审计的DLinear报告：
`.\.venv\Scripts\python.exe -c "from report_dlinear import generate; generate()"`。
项目、虚拟环境与训练产物位于D盘；原始压缩包仍保留在用户提供的桌面位置。

## 阅读入口

- BASELINE_REPORT.md：全部结果、低历史量表现、周期性和后续对照建议。
- TIME_PROTOCOL.md：当前生效的canonical区间及EventTime协议。
- configs/baselines.json：固定的日期、模型、事件rank和评价配置。
- artifacts/baselines/run_status.json：最近运行是否完整成功；failed/running时不要将旧结果当作本次成功结果。

## 一条命令完整复现

在此目录打开PowerShell：

```powershell
.\.venv\Scripts\python.exe run_baselines.py
```

顺序固定为：因果性预检 → 334天walk-forward → 所有输出的独立核验 → 指标/图 → 报告。任一测试失败立即停止。不会运行深度模型、附件3插值或优化，也不会读取或修改result工作簿。

迁移到另一台机器，复制项目（不需复制.venv）并执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-baselines.txt
.\.venv\Scripts\python.exe run_baselines.py
```

原始输入路径由configs/baselines.json指定，默认使用项目内inputs/C题/附件/附件2.xlsx。

只复核运行结果：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_event_causality.py tests/test_prediction_artifacts.py -q
```

在完整结果已通过审计后重新计算指标/图：

```powershell
.\.venv\Scripts\python.exe evaluate_all.py
```

## 输出

- artifacts/baselines/predictions.csv：按series、model、forecast_date、target_slot唯一，共480,960行。target_power_kw为事后真值，pred_power_kw为冻结预测，单位均kW。
- artifacts/baselines/canonical_actuals.csv：保留source_day/slot_id/source_label/interval_start/end与available_at/rank。
- artifacts/baselines/lineage.csv：每个预测日实际使用的历史日期、输入量及最晚可用事件。
- artifacts/metrics/：global/monthly/horizon/history_size/daily/exact_31_days指标。
- artifacts/figures/：12张PNG评价图。
- artifacts/leakage_audit/：两阶段测试日志与XML报告、完整causality审计。
- artifacts/baselines/environment.json：版本、输入/代码/config哈希与运行信息。

history_start/history_end表示全部可用历史范围，并非模型实际使用范围；后者在lineage.csv。history_end可以等于decision_time，合法性由rank0<rank2决定。详见TIME_PROTOCOL.md。

## 审计历史

DATA_AUDIT.md、TIME_ALIGNMENT_ISSUES.md和artifacts/data_audit保留上一阶段的原始审计记录。其中附件2的时间待定状态已由TIME_PROTOCOL.md覆盖；result模板问题仍未用于生成submission mapping。

审计本身仍可通过`.\.venv\Scripts\python.exe run_audit.py`复现，使用独立configs/audit.json。本轮baseline不调用审计入口，不读取附件3/4或模板。
