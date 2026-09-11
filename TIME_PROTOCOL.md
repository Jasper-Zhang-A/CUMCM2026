# Frozen time and availability protocol

本协议由用户在DATA AUDIT之后明确冻结，取代TIME_ALIGNMENT_ISSUES.md中针对附件2的待定状态。原审计保留作为历史证据。

## 区间

附件1/2/4的源标签作为interval_end解释，这是建模约定，并非题面明确规定。本轮代码只读取附件2。

source_day是原始日期列；slot_id由源时间标签解析出的分钟偏移/10-1得到，不能用结果模板或字符串列位置决定。interval_start=source_day+slot_id*10min；interval_end=interval_start+10min。源标签00:10对应[00:00,00:10)，0:00+1对应[23:50,24:00)。每个source_day必须完整覆盖[00:00,24:00)。保留source_label和source_timestamp=interval_end。功率保留kW，不作kWh转换。

## 事件顺序

EventTime(timestamp, rank)按二元组字典序排序：rank0=actual完成可读取，rank1=同刻预报发布，rank2=决策。实际值available_event=(interval_end,0)，每日decision_event=(source_day 00:00,2)。任何输入均须available_event<decision_event且input.source_day<target.source_day。

2025-02-01 00:00 rank0完成的1月31日最后一个区间可以在rank2决策中使用。其墙钟可用时间等于decision_time，不需要伪造epsilon或回拨timestamp。

history_start/history_end记录**已允许历史的最早interval_start和最晚interval_end**，因此history_end可等于decision_time；另存history_end_rank=0、decision_rank=2作为可用性证明。模型实际取用的日期、样本数和事件范围单独保存到lineage.csv，避免把可用历史量与方法使用样本量混为一谈。

## 预测与揭晓

预测器只接收已过滤且经过完整性验证的CausalHistory，不接收全年actual、目标真值、价格、附件3或模板。B0–B4均先生成并固定144步预测，之后runner才将对应真值加入评价记录。daylight-only的y_true>0掩码仅在评价函数中使用。所有均值仅在明确选中的合法历史行中计算，无scaler，无全年拟合，无超参数搜索。

全年指标只用于事后比较；本轮不根据全年test结果训练、调参、挑选例图或形成上线模型。例图日期在运行前固定为2025-02-01。历史量与月份共同变化，因此误差随历史增加的变化不能单独解释为学习收益。

result模板不决定本时间轴，本轮不读写result文件，不建立submission mapping。附件3/4、深度模型和优化均不进入本轮pipeline。
