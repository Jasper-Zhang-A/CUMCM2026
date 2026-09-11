# TiRex Zero-Shot Forecasting Baseline

## 结果

本流程以纯推理方式运行 TiRex，本地训练步数和微调步数均为 0。TiRex historical-only baseline 生成 577,152 条十分钟预测，覆盖负载、光伏实际功率、波动电价，以及每日 00:00、06:00、12:00、18:00 四个决策时刻。

| 范围 | 模型 | 目标 | 分辨率 | n | MAE | RMSE | WAPE (%) |
|---|---|---|---|---:|---:|---:|---:|
| Q2_day_ahead_24h | TiRex_historical_only | A2_LOAD | 10min | 48096 | 669.2722 | 993.7321 | 14.4965 |
| Q2_day_ahead_24h | TiRex_historical_only | A2_PV_ACTUAL | 10min | 48096 | 196.8471 | 355.2581 | 8.2749 |
| Q3_TiRex_historical_only_latest_path | TiRex_historical_only | A2_LOAD | 10min | 48096 | 288.9375 | 418.3866 | 6.2584 |
| Q3_TiRex_historical_only_latest_path | TiRex_historical_only | A2_PV_ACTUAL | 10min | 48096 | 195.3240 | 357.6949 | 8.2108 |
| Q4-2_day_ahead_24h | TiRex_historical_only | A2_LOAD | 10min | 48096 | 669.2722 | 993.7321 | 14.4965 |
| Q4-2_day_ahead_24h | TiRex_historical_only | A2_PV_ACTUAL | 10min | 48096 | 196.8471 | 355.2581 | 8.2749 |
| Q4-2_day_ahead_24h | TiRex_historical_only | A4_PRICE | 10min | 48096 | 0.0913 | 0.1203 | 12.0513 |
| Q4-3_TiRex_latest_path | TiRex_historical_only | A2_LOAD | 10min | 48096 | 288.9375 | 418.3866 | 6.2584 |
| Q4-3_TiRex_latest_path | TiRex_historical_only | A2_PV_ACTUAL | 10min | 48096 | 195.3240 | 357.6949 | 8.2108 |
| Q4-3_TiRex_latest_path | TiRex_historical_only | A4_PRICE | 10min | 48096 | 0.0632 | 0.0840 | 8.3484 |
| Q3_official_vs_TiRex_latest_path | TiRex_historical_only | A2_PV_ACTUAL | hourly_endpoint_proxy | 8016 | 204.4583 | 367.3847 | 8.5984 |
| Q3_official_vs_TiRex_latest_path | Official_Attachment3 | A2_PV_ACTUAL | hourly_endpoint_proxy | 8016 | 57.5078 | 132.3267 | 2.4185 |

`Q3_official_vs_TiRex_latest_path` 在相同发布时刻、相同整点目标和相同代理真值上比较两个模型。附件 3 的整点光伏预报构成 Official Attachment3 baseline；历史实际光伏输入 TiRex 构成 TiRex historical-only baseline。

## 口径

- 日前路径：00:00 发布的 144 个十分钟预测覆盖当天 24 小时，每个目标全年计 48,096 条。
- 最新可用路径：00/06/12/18 各版本只负责紧随其后的 6 小时，每个十分钟目标全年计 48,096 条。
- 全修订预测池：`metrics.csv` 保留每次发布后完整 24 小时的分段预测能力，用于模型诊断。
- 负载在线假设：06/12/18 预测读取决策时刻之前已经观测的实际负载。保守口径继续采用 00:00 负载预测，其分数等于 Q2 日前负载行。
- 价格信息假设：未来实时价格在决策时刻未知，TiRex 生成 `decision_price_forecast`；附件 4 实际价格作为 `realized_settlement_price` 参与评价。
- 分位数处理：点预测采用 TiRex 原始中位数并投影到非负域；q10—q90 经非负投影和单调重排后用于概率指标。
- 电价投影依据：附件 4 的本题价格序列全部位于非负域。

## 复现

```bash
python3 -m venv --system-site-packages .venv-tirex
.venv-tirex/bin/pip install -r requirements-tirex.txt
.venv-tirex/bin/python run_tirex_zero_shot.py --device cuda:0
```

## 交付文件

- `forecasts.csv.gz`：TiRex historical-only 完整预测、实际值、决策假设和单调分位数。
- `question_metrics.csv`：按题目使用口径汇总的点预测及概率指标。
- `latest_available_path_metrics.csv`：48,096 条十分钟最新可用路径，以及 TiRex/附件 3 的公平整点对比。
- `metrics.csv`：所有滚动修订预测按决策时刻和 6 小时跨度分组的诊断指标。
- `a3_hourly_comparison.csv`：TiRex 与附件 3 在全部整点发布版本上的分组对比。
- `run_metadata.json`：运行参数、软件版本与语义设置。

本轮交付范围为预测层。购电、储能、紧急购电和费用结算属于后续调度优化层。
