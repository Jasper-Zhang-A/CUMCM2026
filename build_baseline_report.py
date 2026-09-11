"""Generate the report directly from frozen prediction metrics; no hand-copied CSV results."""
from pathlib import Path
import json, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from seasonal import ROOT, config

def md(frame):
    frame=frame.copy()
    for column in frame.select_dtypes(include='number'):
        if column in ['n','horizon','tests','failed','skipped']:
            frame[column]=frame[column].map(lambda number:f'{int(number):,}' if pd.notna(number) else 'N/A')
        elif column=='nmae':
            frame[column]=frame[column].map(lambda number:f'{number:.2%}' if pd.notna(number) else 'N/A')
        else:
            frame[column]=frame[column].map(lambda number:f'{number:.4f}' if pd.notna(number) else 'N/A')
    frame=frame.fillna('N/A')
    return '| '+' | '.join(frame.columns)+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(str(cell) for cell in row)+' |' for row in frame.itertuples(index=False,name=None))

def change_word(fraction):
    return f'降低{fraction:.2%}' if fraction>=0 else f'升高{-fraction:.2%}'

def interval(horizon):
    slot=int(horizon)-1
    start=slot*10; end=start+10
    return f'{start//60:02d}:{start%60:02d}–{end//60:02d}:{end%60:02d}'

def build_report():
    settings=config(); metrics_dir=ROOT/'artifacts/metrics'
    global_metrics=pd.read_csv(metrics_dir/'metrics_global.csv')
    monthly=pd.read_csv(metrics_dir/'metrics_monthly.csv')
    history=pd.read_csv(metrics_dir/'metrics_history_size.csv')
    daily=pd.read_csv(metrics_dir/'metrics_daily.csv',parse_dates=['forecast_date'])
    horizons=pd.read_csv(metrics_dir/'metrics_horizon.csv')
    low=pd.read_csv(metrics_dir/'metrics_exact_31_days.csv')
    environment=json.loads((ROOT/'artifacts/baselines/environment.json').read_text(encoding='utf-8'))
    runtime=pd.read_csv(ROOT/'artifacts/baselines/runtime.csv')
    tests=[]
    for phase in ['preflight_tests','postflight_tests']:
        suite=ET.parse(ROOT/f'artifacts/leakage_audit/{phase}.xml').getroot().find('testsuite')
        assert int(suite.get('failures'))==0 and int(suite.get('errors'))==0 and int(suite.get('skipped'))==0
        tests.append({'phase':phase,'tests':int(suite.get('tests')),'failed':0,'skipped':0})
    winners={series:global_metrics.loc[global_metrics.series==series].sort_values('mae_kw').iloc[0] for series in ['Load','PV']}
    summary=global_metrics[['series','model','n','mae_kw','rmse_kw','nmae','daylight_mae_kw']].copy()
    summary['nmae']=summary.nmae.map(lambda fraction:f'{fraction:.2%}')
    names={'B0':'Persistence','B1':'Yesterday Same Slot','B2':'Last Week Same Slot','B3':'Previous 7-day Same-Slot Mean','B4':'Historical Same-Weekday Mean'}
    lines=['# Seasonal Baseline Report — Q2 / B0–B4',
        '**范围：只使用附件2，2025-02-01至2025-12-31每日00:00预测当天144个区间。没有使用附件3/4、result模板、深度模型或LP/MPC。**',
        f'本轮得到480,960条预测记录（334天×144步×2条序列×5种方法）；{sum(item["tests"] for item in tests)}项测试全部通过。下文“最强”指固定方法在本次Feb–Dec评估上的事后MAE排名，不是使用test集选择上线模型或超参数。',
        '## 全局结果',md(summary),
        'MAE/RMSE单位kW；nMAE=MAE/同组mean(|actual|)。PV daylight-only严格以事后actual>0筛选；夜间0值不进入该子集，未传给预测器。全日MAE即表中mae_kw。',
        '## 1–2. Load和PV最强简单baseline']
    for series in ['Load','PV']:
        best=winners[series]
        ranking=global_metrics.loc[global_metrics.series==series].sort_values('mae_kw')
        lines.append(f'{series}：**{best.model} ({names[best.model]})**，MAE={best.mae_kw:.4f} kW，RMSE={best.rmse_kw:.4f} kW，nMAE={best.nmae:.2%}。MAE排名：'+ ' < '.join(ranking.model)+ '。')
    lines.append('## 3. Weekly seasonality是否非常强？')
    for series in ['Load','PV']:
        group=global_metrics.loc[global_metrics.series==series].set_index('model')
        reduction=1-group.loc['B2','mae_kw']/group.loc['B1','mae_kw']
        daily_pairs=daily.loc[daily.series==series].pivot(index='forecast_date',columns='model',values='mae_kw')
        wins=(daily_pairs.B2<daily_pairs.B1).mean()
        interpretation='支持较强的周周期信号' if reduction>0.2 and wins>0.6 else '不足以称为非常强的周周期信号'
        lines.append(f'{series}：B2相比B1的MAE{change_word(reduction)}，在{wins:.2%}的评估日胜出；{interpretation}。这是预测对照证据，不是独立样本下的显著性检验。')
    lines.append('## 4. Moving-average是否对PV更强？')
    pv=global_metrics.loc[global_metrics.series=='PV'].set_index('model')
    lines.append(f'PV的B3 MAE={pv.loc["B3","mae_kw"]:.4f} kW；相比B1降低{1-pv.loc["B3","mae_kw"]/pv.loc["B1","mae_kw"]:.2%}，相比B2降低{1-pv.loc["B3","mae_kw"]/pv.loc["B2","mae_kw"]:.2%}。'+('B3同时优于这两个单日季节基线。' if pv.loc['B3','mae_kw']<min(pv.loc['B1','mae_kw'],pv.loc['B2','mae_kw']) else 'B3并没有同时优于这两个单日季节基线。')+'是否为所有方法中最佳，以全局表为准，不能预设均值法一定适合PV。')
    lines.extend(['## 5. 只有31天历史时的性能',
        '2025-02-01恰有31个完整历史日、4,464个可见区间。最后一格于02-01 00:00 rank0可读，先于同刻rank2决策。B0实际只用1点，B1/B2用1天，B3用7天，B4用1月份4个同weekday日；available_history_days并不等于每个方法的实际输入量。',
        md(low[['series','model','mae_kw','rmse_kw','nmae','daylight_mae_kw']]),
        '31天精确条件只对应一个预测日，不能用这一天代表整个低数据分布。2月份整体另报如下（历史量31–58天）：',
        md(monthly.loc[monthly.month=='2025-02',['series','model','n','mae_kw','rmse_kw','nmae']]),
        '## 6. 随历史积累是否改善？'])
    bin_table=history.pivot(index=['series','model'],columns='history_size_group',values='mae_kw').reindex(columns=settings['history_bin_labels']).reset_index()
    lines.append(md(bin_table))
    for series in ['Load','PV']:
        model=winners[series].model
        subset=history.loc[(history.series==series)&(history.model==model)].set_index('history_size_group').reindex(settings['history_bin_labels'])
        trend=1-subset.mae_kw.iloc[-1]/subset.mae_kw.iloc[0]
        monotone=bool(np.all(np.diff(subset.mae_kw)<=0))
        lines.append(f'{series}的全局最佳{model}从31–60天组到>240天组，MAE{change_word(trend)}；四组'+('单调不增。' if monotone else '并非单调不增。'))
    lines.append('**不能把上述变化直接归因于数据量增长。** B0/B1/B2/B3只使用固定长度的近期历史，不因累计历史增加而学习更多；只有B4的平均样本数扩展。历史天数与季节、月份完全联动，当前设计不能分离这些因素，也未人为缩减历史做固定季节对照。')
    lines.append('## 7. 哪些horizon最难？')
    hardest=[]
    for series in ['Load','PV']:
        for model in settings['models']:
            row=horizons.loc[(horizons.series==series)&(horizons.model==model)].sort_values('mae_kw',ascending=False).iloc[0]
            hardest.append({'series':series,'model':model,'horizon':int(row.horizon_step),'interval':interval(row.horizon_step),'mae_kw':row.mae_kw})
        model=winners[series].model
        peaks=horizons.loc[(horizons.series==series)&(horizons.model==model)].nlargest(3,'mae_kw')
        lines.append(f'{series}以{model}为参照的最高误差3个区间：'+ '；'.join(f'h={int(row.horizon_step)} / {interval(row.horizon_step)} / MAE={row.mae_kw:.2f} kW' for row in peaks.itertuples())+'。')
    lines.extend([md(pd.DataFrame(hardest)),'每天均从00:00预测，因此horizon同时等于日内时刻。不能仅凭曲线认为远期步长天然更难；负载峰谷和PV日照强度同样影响绝对误差。PV夜间实际均为0的horizon分母为0时，nMAE记为NaN，不伪装为0。',
        '## 8. Causality检查结果',md(pd.DataFrame(tests)),
        '未发现因果性违规。预检覆盖事件字典序、1月末样本边界、日期/slot唯一性和连续性、B0–B4精确输入集合、拒绝未来/迟到/乱序/缺失输入、未来值污染不变性、无scaler以及模型保存/加载。回测后逐行检查480,960条输出和3,340条输入来源记录，并用独立数组实现重算全部B0–B4预测与真值对齐。',
        '测试不是“无风险”的逻辑证明；它们覆盖当前输入、实现和冻结协议。真实数据若存在未建模的发布延迟，需要修改available_event；当前按用户授权的零延迟rank0约定执行。',
        '## 9–10. 后续DLinear和PatchTST至少要超过谁？',
        f'DLinear至少应在同样事件协议、单变量任务和目标日上超过Load的**{winners["Load"].model}**与PV的**{winners["PV"].model}**，同时完整保留B1/B2/B3/B4及月度、低历史量对照。这里只给出本次事后最强简单基线，不能按全年成本或test MAE选择深度模型超参数。',
        'PatchTST至少达到相同的简单基线门槛，并在DLinear正式完成后继续比较DLinear。DLinear/PatchTST均未运行，所以当前不能声称它们优于任何基线；将来调参必须只用每个forecast origin之前的chronological validation。',
        '## 时间和输出定义',
        '依据TIME_PROTOCOL.md：源00:10解释为[00:00,00:10)的结束标签，是建模约定。source_day永远保留原始日期；slot_id由解析后的结束分钟计算，全年每条序列52,560个区间、365个完整日。source_timestamp=interval_end；一个source_day严格覆盖[00:00,24:00)。',
        'predictions.csv中history_start/history_end为可用历史区间起止，history_end等于decision_time是合法的，因为rank0<rank2。input_available_at/input_available_rank记录本模型实际使用输入的最晚可用事件。lineage.csv列出全部使用日期及样本数。target_power_kw是预测冻结后加入的结算真值，不能将含真值的整表作为后续优化器输入。',
        '本轮不填result模板，也不把其错误边界反向用于canonical时间轴。后续submission mapping单独处理。',
        '## 复现与交付',
        '在项目目录执行：` .\\.venv\\Scripts\\python.exe run_baselines.py `。入口先运行因果性测试，失败立即停止；生成预测后再次审计全部结果，再生成指标、图片和本报告。随机种子2026；无随机划分、无深度训练、无scaler。所有指标由预测CSV程序生成，未手工填数。',
        '- artifacts/baselines/predictions.csv：480,960条逐slot评估记录。\n- artifacts/baselines/canonical_actuals.csv：Load/PV各52,560条源标签及区间。\n- artifacts/baselines/lineage.csv：3,340条逐预测的历史来源。\n- artifacts/metrics/metrics_global.csv、metrics_monthly.csv、metrics_horizon.csv、metrics_history_size.csv：要求的四类指标；另附daily与exact_31_days指标。\n- artifacts/leakage_audit/：预检/回测后测试报告和causality_summary.json。\n- artifacts/figures/：Load/PV各3张actual对照、月度MAE和horizon MAE，另有历史量曲线和全局概览，共12张图。',
        f'Python {environment["python"].split()[0]}；NumPy {environment["packages"]["numpy"]}；Pandas {environment["packages"]["pandas"]}。PyTorch/TSLib/CUDA未使用，神经网络参数数为0。各方法fit/predict累计时间如下（含输入选择与校验，非深度训练）：',
        md(runtime.groupby(['series','model']).fit_predict_seconds.sum().reset_index()),
        '环境、输入/代码/config的SHA256、各日耗时已保存。审计用全年数值描述不进入任何基线统计。',
        '## 图示',
        '![全局MAE](artifacts/figures/baseline_overview.png)',
        '![历史量与误差](artifacts/figures/mae_vs_available_history_days.png)',
        '## 当前结论',
        f'已建立事件因果性受约束且结果逐项复核的Seasonal Baseline。当前Load的主要简单对照为{winners["Load"].model}，PV为{winners["PV"].model}。数据量阶段变化须结合季节解释。任务按要求停止在B0–B4，不启动深度模型或储能优化。'])
    (ROOT/'BASELINE_REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('BASELINE_REPORT.md generated from metrics.',flush=True)

if __name__=='__main__': build_report()

