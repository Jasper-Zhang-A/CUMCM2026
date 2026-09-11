# MODEL SELECTION — January only

只使用1月；1月1–24日训练，25–31日每日00:00验证。验证期权重固定，但每天使用当时已可见的最新context。scaler仅fit前24天。三种候选的训练滑窗目标均不越过01-25 rank2。

| series | seq_len | validation_mae_kw | best_epoch | epochs_executed | train_points | validation_points | window_count | checkpoint |
|---|---|---|---|---|---|---|---|---|
| Load | 144.0000 | 689.6668 | 50.0000 | 50.0000 | 3456.0000 | 1008.0000 | 3169.0000 | artifacts\dlinear\checkpoints\selection_Load_L144.pt |
| Load | 432.0000 | 671.9859 | 50.0000 | 50.0000 | 3456.0000 | 1008.0000 | 2881.0000 | artifacts\dlinear\checkpoints\selection_Load_L432.pt |
| Load | 1008.0000 | 146.7398 | 49.0000 | 50.0000 | 3456.0000 | 1008.0000 | 2305.0000 | artifacts\dlinear\checkpoints\selection_Load_L1008.pt |
| PV | 144.0000 | 213.2884 | 50.0000 | 50.0000 | 3456.0000 | 1008.0000 | 3169.0000 | artifacts\dlinear\checkpoints\selection_PV_L144.pt |
| PV | 432.0000 | 162.7873 | 46.0000 | 50.0000 | 3456.0000 | 1008.0000 | 2881.0000 | artifacts\dlinear\checkpoints\selection_PV_L432.pt |
| PV | 1008.0000 | 164.8420 | 7.0000 | 15.0000 | 3456.0000 | 1008.0000 | 2305.0000 | artifacts\dlinear\checkpoints\selection_PV_L1008.pt |

按7个验证日合并的MAE最小选seq_len，各候选用验证MAE早停；选择对应best epoch作为未来所有重训固定轮数。选定后不再根据2–12月修改结构、轮数、学习率或数据处理。

{
  "Load": {
    "seq_len": 1008,
    "epochs": 49,
    "validation_mae_kw": 146.73977463508024
  },
  "PV": {
    "seq_len": 432,
    "epochs": 46,
    "validation_mae_kw": 162.78731363250176
  },
  "selection_cutoff": "2025-02-01",
  "config_sha256": "e23528230b1e3ad9a0ce0e397f46fc4e05d928547fb45c0b499816cc3a4234cc",
  "selection_source_hash": "e986351555b31810106ffa02950acffac2d5d33b519ed3c2c84b97f91136c2b2"
}

固定：TSLib DLinear原实现、moving_avg=25、Adam lr=0.0005、batch=256、MSE、stride=1、shuffle=False、max_epochs=50、patience=8、seed=2026。不裁剪负功率，不用真实日照掩码。

月度重训使用该月之前全部历史，包括当时已经发生的原1月验证段；这是允许的refit。Controlled Low-Data仅fit目标前N天，但结构来自共同January设计阶段；N不是包含模型设计数据在内的整个研究数据预算。后续No-FT对照需同样披露设计/校准数据。

7天预算若小于seq_len+pred_len所需长度，DLinear组合明确跳过；不能临时改seq_len来完成该组合。时间事件按TIME_PROTOCOL.md，00:00 rank0先于rank2，未添加epsilon。