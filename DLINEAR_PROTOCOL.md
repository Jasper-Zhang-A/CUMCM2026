# DLinear Stage 1 protocol

Project: D:/数学建模/CUMCM2026C_TS_Baseline. TIME_PROTOCOL.md still defines all intervals and EventTime ordering.

Only univariate DLinear for Load/PV. Pinned TSLib commit: 4e938a1767106324dd753b2a44832bf870a0252e. Original model, decomposition module, LICENSE and hashes are under vendor/tslib.

January 1-24 is fit history; January 25-31 is chronological validation with daily midnight origins and latest causal context. Only seq_len=144/432/1008 are compared. Each candidate's scaler uses only fit history. Validation chooses length and best epoch. These validation scores are selection scores, not unbiased test results.

configs/dlinear.json was written before evaluation: Adam, MSE, population z-score, stride1, chronological batches, seed2026. No clipping, true-daylight mask, attachment3/4, templates, other deep models or optimization.

Monthly refits use all preceding history with the frozen January epoch count. Checkpoints stay fixed within month, while context advances daily.

Low-data targets are March20, June21, September23 and December21. Budgets are the preceding7/14/30/60/90/180 days. B0-B4 are recomputed within the same budget. Missing history or too few points for seq_len+144 is explicitly skipped, never repaired by changing the selected length. January architecture calibration is separately disclosed: N days denotes the parameter-fitting budget, not all data used in research design. One target per season cannot establish season-wide performance.

Training inputs and targets must be available before the cutoff event. Scaler hashes match the exact training slice. Each checkpoint records weight, data, configuration and implementation hashes, fit cutoff, selection-information cutoff and epoch history. A January-selected checkpoint cannot be used before its validation information is available.

This is offline simulation of information availability. Training wall time is measured, but deployment latency is not incorporated into decision timing.

Only a series beating its fixed strong baseline at seed2026 gets2027/2028 repeats. All repeats use the same length/epochs, with no best-seed selection. Stop after DLinear.

Run: python run_dlinear.py. Original59 tests and learned-model tests gate all fitting/reporting. Matching checkpoints resume work; no existing checkpoints means fresh fitting.

Environment: Python 3.12, NVIDIA CUDA 12.8 PyTorch wheel, RTX 4060 Laptop GPU. Install PyTorch with `python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128`, then `python -m pip install -r requirements-dlinear.txt`. The verified wheel SHA256 is 2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30. This run stores its virtual environment, wheel, temporary installation files, and pip cache on D:.
