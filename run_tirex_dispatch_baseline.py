#!/home/jasper/miniconda3/envs/tslib/bin/python
"""Replace the main method's forecast centers with cached TiRex predictions.

The reference simulation, scenario construction, controller, and settlement are
imported read-only from the main project. Outputs are comparison artifacts only.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CASES = ("result2", "result3", "result4-2", "result4-3")
TARGETS = ("A2_LOAD", "A2_PV_ACTUAL", "A4_PRICE")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_bank(path):
    columns = ["baseline_model", "target_dataset", "issue_time", "horizon_step",
               "valid_interval_end", "prediction", "actual"]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame.loc[frame.baseline_model.eq("TiRex_historical_only")].copy()
    assert len(frame) == 334 * 4 * 144 * 3
    frame["issue_time"] = pd.to_datetime(frame.issue_time)
    frame["valid_interval_end"] = pd.to_datetime(frame.valid_interval_end)
    assert not frame.duplicated(["target_dataset", "issue_time", "horizon_step"]).any()
    assert (frame.valid_interval_end == frame.issue_time +
            pd.to_timedelta(frame.horizon_step * 10, unit="min")).all()
    assert np.isfinite(frame.prediction).all() and (frame.prediction >= 0).all()
    bank = {}
    for issue, group in frame.groupby("issue_time", sort=True):
        pivot = group.pivot(index="horizon_step", columns="target_dataset", values="prediction")
        assert list(pivot.index) == list(range(1, 145))
        values = pivot[list(TARGETS)].to_numpy(float)
        values[:, :2] /= 6.0
        bank[issue.isoformat()] = values
    expected = pd.date_range("2025-02-01", "2025-12-31T18:00:00", freq="6h")
    assert set(bank) == {x.isoformat() for x in expected}
    return bank, frame


def make_forecaster(base_class, bank, pv_mode):
    class TiRexForecaster(base_class):
        def _tirex_center(self, d, start, issue_cap=None):
            cap = start if issue_cap is None else min(start, int(issue_cap))
            issue = pd.Timestamp(self.data.dates[d]) + pd.Timedelta(minutes=10 * cap)
            horizon = 144 - start
            offset = start - cap
            out = bank[issue.isoformat()][offset:offset + horizon].copy()
            assert out.shape == (horizon, 3)
            if not self.use_price:
                out[:, 2] = self.data.tariff[start:]
            out[:, 2] = np.maximum(out[:, 2], 1e-4)
            return out

        def _base_center(self, d):
            if d < 31:
                return super()._base_center(d)
            return self._tirex_center(d, 0)

        def predict(self, d, start=0, issue_cap=None):
            if d < 31:
                return super().predict(d, start, issue_cap)
            out = self._tirex_center(d, start, issue_cap)
            if self.use_a3 and pv_mode == "official":
                out[:, 1], _ = self.data.forecast(d, start, issue_cap=issue_cap)
            return out

        def scenarios(self, d, start=0, **kwargs):
            sc = super().scenarios(d, start, **kwargs)
            if d >= 31:
                # For the historical-only variant, the parent might have queried
                # A3 metadata; no A3 numerical value enters predict or scenarios.
                cap = kwargs.get("issue_cap")
                cap = start if cap is None else min(start, cap)
                issue = pd.Timestamp(self.data.dates[d]) + pd.Timedelta(minutes=10 * cap)
                sc["forecast_issue_time"] = issue.isoformat()
            return sc

    return TiRexForecaster


def independent_totals(trace, end_date="2025-12-31"):
    """Rebuild every revision and all charges directly from executed arrays."""
    dates = np.asarray(trace["dates"])
    selected = (dates >= "2025-02-01") & (dates <= end_date)
    price = trace["price"][selected]
    effective = trace["q0"][selected].copy()
    adjustment = np.zeros_like(effective)
    for version in trace["versions"][selected].transpose(1, 0, 2):
        adopted = np.isfinite(version)
        delta = np.where(adopted, version - effective, 0.0)
        adjustment += 0.5 * price * np.abs(delta)
        effective[adopted] = version[adopted]
    np.testing.assert_allclose(effective, trace["q"][selected], atol=1e-7, rtol=0)
    emergency = trace["u"][selected]
    normal = float(np.sum(effective * price))
    fee = float(adjustment.sum())
    emergency_cost = float(np.sum(5 * price * emergency))
    return dict(total_cost_cny=normal + fee + emergency_cost,
                emergency_kwh=float(emergency.sum()), emergency_cost_cny=emergency_cost,
                effective_purchase_cost_cny=normal, adjustment_fee_cny=fee,
                days=int(selected.sum()), initial_energy_kwh=float(trace["E"][selected][0, 0]),
                final_energy_kwh=float(trace["E"][selected][-1, -1]))


def run_case(job):
    main_root, forecasts, output, case, pv_mode, days = job
    sys.path.insert(0, main_root)
    from src.forecasting import Data, Forecaster
    from src import simulation
    from src.verification import verify_trace, summarize

    bank, frame = read_bank(forecasts)
    data = Data(main_root)
    # Verify forecast target truth against the exact source used for execution.
    for i, target in enumerate(TARGETS):
        actual = pd.Series(data.actual[:, :, i].ravel(),
                           index=pd.date_range("2025-01-01T00:10", periods=365 * 144, freq="10min"))
        target_rows = frame.loc[frame.target_dataset.eq(target)]
        expected = actual.reindex(target_rows.valid_interval_end).to_numpy()
        if i < 2:
            expected *= 6.0
        np.testing.assert_allclose(target_rows.actual.to_numpy(), expected, atol=1e-8, rtol=1e-10)
    root = Path(main_root)
    original = simulation.load_trace(root / "cache" / case)
    config = original["metadata"]["config"]
    variant = dict(original["metadata"]["variant"])
    variant["name"] = f"tirex_{pv_mode}_{case}"
    simulation.Forecaster = make_forecaster(Forecaster, bank, pv_mode)
    trace = simulation.simulate(data, config, variant, days=days, progress=lambda msg: print(msg, flush=True))
    # All January controls and inventory must exactly match the reference warmup.
    for key in simulation.ARRAY_KEYS:
        np.testing.assert_allclose(trace[key][:31], original[key][:31], atol=1e-7, rtol=0,
                                   equal_nan=True, err_msg=f"warmup changed: {case}/{key}")
    verification = verify_trace(trace, case)
    totals = independent_totals(trace)
    daily = summarize(trace)
    np.testing.assert_allclose(totals["total_cost_cny"],
        sum(x["total_cost"] for x in daily if x["day_id"] >= "2025-02-01"), atol=1e-6, rtol=0)
    if case in ("result2", "result3"):
        np.testing.assert_array_equal(trace["price"], np.broadcast_to(data.tariff, trace["price"].shape))
    verification["january_matches_reference"] = True
    verification["forecast_truth_alignment_passed"] = True
    trace["metadata"]["baseline"] = {
        "model": "TiRex_historical_only", "pv_mode": pv_mode,
        "role": "forecast-center replacement with unchanged empirical scenario and dispatch methods",
        "january": "reference forecaster warmup and reference residuals before TiRex forecasts begin",
        "residual_window_days": config["history_window_days"],
        "forecast_file_sha256": sha256(forecasts),
    }
    folder = Path(output) / pv_mode / case
    simulation.save_trace(trace, folder)
    write_json(folder / "verification.json", verification)
    write_json(folder / "totals.json", totals)
    write_json(folder / "daily.json", daily)
    main_totals = independent_totals(original, end_date=trace["dates"][-1])
    totals.update(case=case, pv_mode=pv_mode, main_total_cost_cny=main_totals["total_cost_cny"],
                  baseline_minus_main_cny=totals["total_cost_cny"] - main_totals["total_cost_cny"],
                  main_saving_pct=100 * (1 - main_totals["total_cost_cny"] / totals["total_cost_cny"]))
    print(json.dumps(totals, ensure_ascii=False), flush=True)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=ROOT.parent / "CUMCM")
    parser.add_argument("--forecasts", type=Path, default=ROOT / "artifacts/tirex_zero_shot/forecasts.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tirex_dispatch_baseline")
    parser.add_argument("--pv-mode", choices=("official", "historical"), default="official")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()
    assert 32 <= args.days <= 365
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert sha256(ROOT / "Data/data_clean.csv") == sha256(args.main_root / "data/data_clean.csv")
    manifest = dict(main_root=str(args.main_root.resolve()), forecast_model="TiRex_historical_only",
                    pv_mode=args.pv_mode, days=args.days, forecast_sha256=sha256(args.forecasts),
                    input_sha256=sha256(ROOT / "Data/data_clean.csv"),
                    reference_source_sha256={str(p.relative_to(args.main_root)): sha256(p)
                        for p in sorted((args.main_root / "src").glob("*.py"))},
                    runner_sha256=sha256(__file__))
    jobs = [(str(args.main_root.resolve()), str(args.forecasts.resolve()), str(output),
             case, args.pv_mode, args.days) for case in args.cases]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        totals = list(pool.map(run_case, jobs))
    write_json(output / f"{args.pv_mode}_summary.json", totals)
    pd.DataFrame(totals).to_csv(output / f"{args.pv_mode}_summary.csv", index=False)
    write_json(output / f"{args.pv_mode}_run_manifest.json", manifest)


if __name__ == "__main__":
    main()
