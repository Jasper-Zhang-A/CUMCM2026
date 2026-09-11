#!/usr/bin/env python3
"""TiRex historical-only and official-PV forecasting baselines for problem C."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tirex import TiRexZero


CONTEXT_LENGTH = 2048
PREDICTION_LENGTH = 144
QUANTILES = np.arange(0.1, 1.0, 0.1)
Q_COLUMNS = [f"q{int(q * 100):02d}" for q in QUANTILES]
TARGETS = {
    "A2_LOAD": ("load_kw", "kW", "realized_load"),
    "A2_PV_ACTUAL": ("pv_actual_kw", "kW", "realized_pv"),
    "A4_PRICE": ("electricity_price_cny_per_kwh", "CNY/kWh", "realized_settlement_price"),
}
PREDICTION_ROLES = {
    "A2_LOAD": "decision_load_forecast",
    "A2_PV_ACTUAL": "decision_pv_forecast",
    "A4_PRICE": "decision_price_forecast",
}
ASSUMPTIONS = {
    "A2_LOAD": "online_actual_load_available_through_issue_time",
    "A2_PV_ACTUAL": "historical_actual_pv_available_through_issue_time",
    "A4_PRICE": "future_price_unknown; actual_price_used_only_for_settlement_evaluation",
}


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=["dataset_id", "interval_end", "issue_time", "valid_time", "lead_hours", "value"],
        low_memory=False,
    )


def prepare_series(data: pd.DataFrame, dataset_id: str) -> pd.Series:
    rows = data.loc[data["dataset_id"].eq(dataset_id), ["interval_end", "value"]].copy()
    rows["interval_end"] = pd.to_datetime(rows["interval_end"])
    rows["value"] = pd.to_numeric(rows["value"])
    return rows.sort_values("interval_end").set_index("interval_end")["value"]


def make_contexts(series: pd.Series, issues: pd.DatetimeIndex) -> np.ndarray:
    times = series.index.to_numpy()
    values = series.to_numpy(dtype=np.float32)
    return np.stack(
        [
            values[
                times.searchsorted(issue.to_datetime64(), side="right") - CONTEXT_LENGTH :
                times.searchsorted(issue.to_datetime64(), side="right")
            ]
            for issue in issues
        ]
    )


def forecast_all(model: TiRexZero, data: pd.DataFrame, issues: pd.DatetimeIndex, batch_size: int) -> pd.DataFrame:
    series = {dataset_id: prepare_series(data, dataset_id) for dataset_id in TARGETS}
    contexts = np.concatenate([make_contexts(series[dataset_id], issues) for dataset_id in TARGETS])
    raw_quantiles, _ = model.forecast(
        contexts,
        prediction_length=PREDICTION_LENGTH,
        batch_size=batch_size,
        output_type="numpy",
        output_device="cpu",
    )

    # All attachment targets are nonnegative. Keep the TiRex median for point scoring,
    # then monotonically rearrange quantiles for probabilistic scoring.
    raw_quantiles = np.maximum(raw_quantiles, 0.0)
    point_predictions = raw_quantiles[:, :, 4].copy()
    repaired_quantiles = np.sort(raw_quantiles, axis=2)

    steps = np.tile(np.arange(1, PREDICTION_LENGTH + 1), len(issues))
    repeated_issues = np.repeat(issues.to_numpy(), PREDICTION_LENGTH)
    valid_end = repeated_issues + steps.astype("timedelta64[m]") * 10
    frames = []
    for target_index, (dataset_id, (metric, unit, actual_role)) in enumerate(TARGETS.items()):
        start = target_index * len(issues)
        stop = start + len(issues)
        quantiles = repaired_quantiles[start:stop].reshape(-1, len(QUANTILES))
        frame = pd.DataFrame(
            {
                "baseline_model": "TiRex_historical_only",
                "target_dataset": dataset_id,
                "target_metric": metric,
                "unit": unit,
                "decision_data_assumption": ASSUMPTIONS[dataset_id],
                "issue_time": repeated_issues,
                "decision_minute": pd.DatetimeIndex(repeated_issues).hour * 60,
                "horizon_step": steps,
                "lead_minutes": steps * 10,
                "valid_interval_start": valid_end - np.timedelta64(10, "m"),
                "valid_interval_end": valid_end,
                "prediction_role": PREDICTION_ROLES[dataset_id],
                "actual_role": actual_role,
                "actual": series[dataset_id].reindex(pd.DatetimeIndex(valid_end)).to_numpy(),
                "prediction": point_predictions[start:stop].reshape(-1),
            }
        )
        frame[Q_COLUMNS] = quantiles
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def point_metrics(actual: pd.Series, prediction: pd.Series) -> dict[str, float | int]:
    actual = actual.to_numpy(dtype=float)
    prediction = prediction.to_numpy(dtype=float)
    error = prediction - actual
    denominator = np.abs(actual) + np.abs(prediction)
    smape = np.divide(2 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0)
    return {
        "n": len(actual),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape_pct": float(100 * np.sum(np.abs(error)) / np.sum(np.abs(actual))),
        "smape_pct": float(100 * np.mean(smape)),
        "bias": float(np.mean(error)),
    }


def probabilistic_metrics(group: pd.DataFrame) -> dict[str, float]:
    actual = group["actual"].to_numpy(dtype=float)
    forecasts = group[Q_COLUMNS].to_numpy(dtype=float)
    error = actual[:, None] - forecasts
    pinball = np.maximum(QUANTILES * error, (QUANTILES - 1) * error)
    return {
        "mean_pinball_loss": float(pinball.mean()),
        "q10_q90_coverage_pct": float(100 * np.mean((actual >= forecasts[:, 0]) & (actual <= forecasts[:, -1]))),
        "q10_q90_mean_width": float(np.mean(forecasts[:, -1] - forecasts[:, 0])),
    }


def score_row(group: pd.DataFrame, **labels: object) -> dict[str, object]:
    row = dict(labels)
    row.update(point_metrics(group["actual"], group["prediction"]))
    if set(Q_COLUMNS).issubset(group.columns):
        row.update(probabilistic_metrics(group))
    return row


def revision_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    evaluated = forecasts.dropna(subset=["actual"]).copy()
    evaluated["horizon_bucket"] = pd.cut(
        evaluated["lead_minutes"],
        [0, 360, 720, 1080, 1440],
        labels=["0-6h", "6-12h", "12-18h", "18-24h"],
        include_lowest=True,
    )
    rows = []
    for (target, decision, bucket), group in evaluated.groupby(
        ["target_dataset", "decision_minute", "horizon_bucket"], observed=True
    ):
        rows.append(
            score_row(
                group,
                metric_scope="all_revision_forecasts_pooled",
                baseline_model="TiRex_historical_only",
                target_dataset=target,
                target_metric=TARGETS[target][0],
                unit=TARGETS[target][1],
                decision_minute=int(decision),
                horizon_bucket=str(bucket),
                issue_count=group["issue_time"].nunique(),
            )
        )
    return pd.DataFrame(rows)


def align_a3(data: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    official = data.loc[
        data["dataset_id"].eq("A3_PV_FORECAST"), ["issue_time", "valid_time", "lead_hours", "value"]
    ].copy()
    official["issue_time"] = pd.to_datetime(official["issue_time"])
    official["valid_time"] = pd.to_datetime(official["valid_time"])
    official = official[official["issue_time"].ge("2025-02-01")].rename(columns={"value": "official_prediction"})
    tirex = forecasts.loc[
        forecasts["target_dataset"].eq("A2_PV_ACTUAL") & forecasts["horizon_step"].mod(6).eq(0),
        ["issue_time", "valid_interval_end", "actual", "prediction"],
    ].rename(columns={"valid_interval_end": "valid_time", "prediction": "tirex_prediction"})
    aligned = official.merge(tirex, on=["issue_time", "valid_time"]).dropna(subset=["actual"])
    aligned["decision_minute"] = aligned["issue_time"].dt.hour * 60
    aligned["lead_bucket"] = pd.cut(
        aligned["lead_hours"], [0, 6, 12, 18, 24], labels=["1-6h", "7-12h", "13-18h", "19-24h"]
    )
    return aligned


def pv_comparison(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (decision, bucket), group in aligned.groupby(["decision_minute", "lead_bucket"], observed=True):
        for model_name, column in (
            ("TiRex_historical_only", "tirex_prediction"),
            ("Official_Attachment3", "official_prediction"),
        ):
            renamed = group.rename(columns={column: "prediction"})
            rows.append(
                score_row(
                    renamed,
                    metric_scope="all_hourly_revision_forecasts_pooled",
                    baseline_model=model_name,
                    target_dataset="A2_PV_ACTUAL",
                    target_metric="pv_actual_kw",
                    unit="kW",
                    decision_minute=int(decision),
                    lead_bucket=str(bucket),
                    issue_count=group["issue_time"].nunique(),
                )
            )
    return pd.DataFrame(rows)


def latest_path_metrics(forecasts: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    latest = forecasts.dropna(subset=["actual"])
    latest = latest[latest["lead_minutes"].le(360)]
    rows = []
    for target, group in latest.groupby("target_dataset"):
        rows.append(
            score_row(
                group,
                path_scope="latest_available_6h_path",
                baseline_model="TiRex_historical_only",
                target_dataset=target,
                target_metric=TARGETS[target][0],
                unit=TARGETS[target][1],
                resolution="10min",
                assumption=ASSUMPTIONS[target],
                issue_count=group["issue_time"].nunique(),
            )
        )

    hourly = aligned[aligned["lead_hours"].le(6)]
    for model_name, column in (
        ("TiRex_historical_only", "tirex_prediction"),
        ("Official_Attachment3", "official_prediction"),
    ):
        group = hourly.rename(columns={column: "prediction"})
        rows.append(
            score_row(
                group,
                path_scope="latest_available_6h_path",
                baseline_model=model_name,
                target_dataset="A2_PV_ACTUAL",
                target_metric="pv_actual_kw",
                unit="kW",
                resolution="hourly_endpoint_proxy",
                assumption="Attachment3 forecast point matched to same-time ten-minute actual interval",
                issue_count=group["issue_time"].nunique(),
            )
        )
    return pd.DataFrame(rows)


def question_metrics(forecasts: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    evaluated = forecasts.dropna(subset=["actual"])
    latest = evaluated["lead_minutes"].le(360)
    specs = [
        ("Q2_day_ahead_24h", ["A2_LOAD", "A2_PV_ACTUAL"], evaluated["decision_minute"].eq(0)),
        ("Q3_TiRex_historical_only_latest_path", ["A2_LOAD", "A2_PV_ACTUAL"], latest),
        ("Q4-2_day_ahead_24h", list(TARGETS), evaluated["decision_minute"].eq(0)),
        ("Q4-3_TiRex_latest_path", list(TARGETS), latest),
    ]
    rows = []
    for scope, targets, mask in specs:
        selected = evaluated.loc[mask & evaluated["target_dataset"].isin(targets)]
        for target, group in selected.groupby("target_dataset"):
            rows.append(
                score_row(
                    group,
                    question_scope=scope,
                    baseline_model="TiRex_historical_only",
                    target_dataset=target,
                    target_metric=TARGETS[target][0],
                    unit=TARGETS[target][1],
                    resolution="10min",
                    assumption=ASSUMPTIONS[target],
                    issue_count=group["issue_time"].nunique(),
                )
            )

    hourly = aligned[aligned["lead_hours"].le(6)]
    for model_name, column in (
        ("TiRex_historical_only", "tirex_prediction"),
        ("Official_Attachment3", "official_prediction"),
    ):
        group = hourly.rename(columns={column: "prediction"})
        rows.append(
            score_row(
                group,
                question_scope="Q3_official_vs_TiRex_latest_path",
                baseline_model=model_name,
                target_dataset="A2_PV_ACTUAL",
                target_metric="pv_actual_kw",
                unit="kW",
                resolution="hourly_endpoint_proxy",
                assumption="Attachment3 forecast point matched to same-time ten-minute actual interval",
                issue_count=group["issue_time"].nunique(),
            )
        )
    return pd.DataFrame(rows)


def markdown_table(metrics: pd.DataFrame) -> str:
    return "\n".join(
        [
            "| 范围 | 模型 | 目标 | 分辨率 | n | MAE | RMSE | WAPE (%) |",
            "|---|---|---|---|---:|---:|---:|---:|",
            *[
                f"| {row.question_scope} | {row.baseline_model} | {row.target_dataset} | "
                f"{row.resolution} | {row.n} | {row.mae:.4f} | {row.rmse:.4f} | {row.wape_pct:.4f} |"
                for row in metrics.itertuples()
            ],
        ]
    )


def write_report(path: Path, forecasts: pd.DataFrame, metrics: pd.DataFrame) -> None:
    path.write_text(
        f"""# TiRex Zero-Shot Forecasting Baseline

## 结果

本流程以纯推理方式运行 TiRex，本地训练步数和微调步数均为 0。TiRex historical-only baseline 生成 {len(forecasts):,} 条十分钟预测，覆盖负载、光伏实际功率、波动电价，以及每日 00:00、06:00、12:00、18:00 四个决策时刻。

{markdown_table(metrics)}

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
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Data/data_clean.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("HuggingFace/model.ckpt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tirex_zero_shot"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(root / args.data)
    issues = pd.date_range("2025-02-01", "2025-12-31 18:00", freq="6h")
    model = TiRexZero.from_pretrained(str(root / args.checkpoint), device=args.device, backend="torch").eval()
    forecasts = forecast_all(model, data, issues, args.batch_size)
    aligned = align_a3(data, forecasts)
    revision = revision_metrics(forecasts)
    latest = latest_path_metrics(forecasts, aligned)
    questions = question_metrics(forecasts, aligned)
    comparison = pv_comparison(aligned)

    csv_options = {"index": False, "float_format": "%.8f"}
    forecasts.to_csv(
        output_dir / "forecasts.csv.gz",
        compression="gzip",
        date_format="%Y-%m-%dT%H:%M:%S",
        **csv_options,
    )
    revision.to_csv(output_dir / "metrics.csv", **csv_options)
    questions.to_csv(output_dir / "question_metrics.csv", **csv_options)
    latest.to_csv(output_dir / "latest_available_path_metrics.csv", **csv_options)
    comparison.to_csv(output_dir / "a3_hourly_comparison.csv", **csv_options)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "TiRex zero-shot inference",
        "training_steps": 0,
        "finetuning_steps": 0,
        "input": str(args.data),
        "checkpoint": str(args.checkpoint),
        "parameters": {
            "device": args.device,
            "batch_size": args.batch_size,
            "context_length": CONTEXT_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "issue_minutes": [0, 360, 720, 1080],
            "quantile_postprocess": "nonnegative_projection_then_monotonic_rearrangement",
        },
        "semantics": {
            "tirex_baseline": "historical_only",
            "q3_official_baseline": "Attachment3",
            "rolling_execution_path": "latest forecast supplies next 6 hours",
            "load_revision_assumption": ASSUMPTIONS["A2_LOAD"],
            "price_prediction_role": "decision_price_forecast",
            "price_actual_role": "realized_settlement_price",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "output": {
            "forecast_rows": len(forecasts),
            "evaluated_rows": int(forecasts["actual"].notna().sum()),
            "latest_path_rows_per_ten_minute_target": 48096,
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(root / "TIREX_ZERO_SHOT_REPORT.md", forecasts, questions)
    print(f"wrote {len(forecasts):,} forecasts with {args.device}")


if __name__ == "__main__":
    main()
