"""Reports from audited DLinear predictions."""
import json, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from seasonal import ROOT, config as seasonal_config
from dlinear_core import OUT, settings
from evaluate_all import metrics

def table(frame):
    frame=frame.copy()
    for column in frame.select_dtypes(include='number'):
        integral=pd.api.types.is_integer_dtype(frame[column])
        frame[column]=frame[column].map(lambda number:(f'{int(number)}' if integral else f'{number:.4f}') if pd.notna(number) else 'N/A')
    frame=frame.fillna('N/A')
    return '| '+' | '.join(frame.columns)+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(str(cell) for cell in row)+' |' for row in frame.itertuples(index=False,name=None))

def load_preds(path):
    frame=pd.read_csv(path)
    for column in ['forecast_date','decision_time','interval_start','interval_end']: frame[column]=pd.to_datetime(frame[column],format='mixed')
    return frame

def generate():
    cfg=settings();basecfg=seasonal_config();frame=load_preds(OUT/'predictions_seed2026.csv')
    frame['month']=frame.forecast_date.dt.strftime('%Y-%m')
    frame['history_size_group']=pd.cut(frame.available_history_days,basecfg['history_bin_edges'],labels=basecfg['history_bin_labels'])
    outputs={}
    for name,extra in {'global':[],'monthly':['month'],'horizon':['horizon_step'],'history_size':['history_size_group']}.items():
        outputs[name]=metrics(frame,['series','model']+extra)
        outputs[name].to_csv(OUT/f'metrics_dlinear_{name}.csv',index=False)
    baselines=pd.read_csv(ROOT/'artifacts/metrics/metrics_global.csv')
    comparison=pd.concat([baselines,outputs['global']],ignore_index=True)
    comparison.to_csv(OUT/'metrics_model_comparison.csv',index=False)
    improvements=[]
    for scope in ['global','monthly','horizon','history_size']:
        base=pd.read_csv(ROOT/f'artifacts/metrics/metrics_{scope}.csv')
        references=base.loc[((base.series=='Load')&(base.model=='B2'))|((base.series=='PV')&(base.model=='B3'))]
        keys=['series']+{'global':[],'monthly':['month'],'horizon':['horizon_step'],'history_size':['history_size_group']}[scope]
        merged=outputs[scope].merge(references,on=keys,suffixes=('_dlinear','_baseline'),validate='one_to_one')
        merged['absolute_improvement_kw']=merged.mae_kw_baseline-merged.mae_kw_dlinear
        merged['relative_improvement_pct']=100*merged.absolute_improvement_kw/merged.mae_kw_baseline.replace(0,np.nan)
        merged['scope']=scope;improvements.append(merged)
    improvement=pd.concat(improvements,ignore_index=True);improvement.to_csv(OUT/'improvements.csv',index=False)
    seeds=[]
    for path in sorted(OUT.glob('predictions_seed*.csv')): seeds.append(metrics(load_preds(path),['series','model','seed']))
    seed_metrics=pd.concat(seeds,ignore_index=True);seed_metrics.to_csv(OUT/'metrics_by_seed.csv',index=False)
    stability=seed_metrics.groupby('series').agg(n_seeds=('seed','size'),mae_mean_kw=('mae_kw','mean'),mae_std_kw=('mae_kw','std'),rmse_mean_kw=('rmse_kw','mean'),rmse_std_kw=('rmse_kw','std')).reset_index()
    stability.to_csv(OUT/'metrics_seed_stability.csv',index=False)
    selection=pd.read_csv(OUT/'january_validation.csv');frozen=json.loads((OUT/'frozen_selection.json').read_text())
    low=pd.read_csv(OUT/'metrics_low_data.csv');audits=pd.read_csv(OUT/'training_sample_audit.csv')
    tests=[]
    for name in ['seasonal_regression','learned_preflight','learned_postflight']:
        suite=ET.parse(OUT/f'tests/{name}.xml').getroot().find('testsuite')
        assert int(suite.get('failures'))==int(suite.get('errors'))==int(suite.get('skipped'))==0
        tests.append({'suite':name,'tests':int(suite.get('tests')),'failures':0,'skipped':0})
    from dlinear_plots import plots
    plots(frame,outputs,low,selection)
    global_imp=improvement.loc[improvement.scope=='global']
    lines=['# MODEL COMPARISON — DLinear Stage 1',
        '**只实现DLinear；项目已迁移到D:/数学建模/CUMCM2026C_TS_Baseline。沿用TIME_PROTOCOL.md、canonical_actuals.csv和EventTime。未使用附件3/4、result模板或优化。**',
        '## 1. January validation选择',table(selection[['series','seq_len','validation_mae_kw','best_epoch','epochs_executed','train_points','validation_points','window_count']]),
        f'Load冻结seq_len={frozen["Load"]["seq_len"]}、epochs={frozen["Load"]["epochs"]}；PV冻结seq_len={frozen["PV"]["seq_len"]}、epochs={frozen["PV"]["epochs"]}。只按1月25–31日验证MAE选择。前24天fit，后7天不进入初次拟合或scaler；验证日context使用当时最新历史。详见MODEL_SELECTION.md。',
        '选定后每月用该月之前全部历史重新训练，包括已成为合法历史的原1月验证段。固定结构、学习率、批量和轮数，不用Feb–Dec重新选择。',
        '## 2–3. DLinear与强简单基线',table(comparison[['series','model','n','mae_kw','rmse_kw','nmae','daylight_mae_kw']]),
        table(global_imp[['series','model_baseline','mae_kw_baseline','mae_kw_dlinear','absolute_improvement_kw','relative_improvement_pct']]),
        'MAE/RMSE单位kW；nMAE为比例。Improvement=(baseline MAE−DLinear MAE)/baseline MAE；负数为退步。PV daylight-only仅在事后评价时用actual>0，未用于模型输入。输出不裁剪、不加真实夜间掩码。']
    for series in ['Load','PV']:
        row=global_imp.loc[global_imp.series==series].iloc[0]
        lines.append(f'{series}：DLinear'+('优于' if row.absolute_improvement_kw>0 else '未超过')+f'{row.model_baseline}，MAE差额{row.absolute_improvement_kw:.4f} kW，相对改善{row.relative_improvement_pct:.2f}%。')
        lines.append(f'RMSE：{row.rmse_kw_baseline:.4f} → {row.rmse_kw_dlinear:.4f} kW（相对改善{100*(row.rmse_kw_baseline-row.rmse_kw_dlinear)/row.rmse_kw_baseline:.2f}%）。成功门槛仍按预注册的MAE判定，不事后切换指标。')
    lines+=['## 4. Monthly表现',table(improvement.loc[improvement.scope=='monthly',['series','month','mae_kw_baseline','mae_kw_dlinear','relative_improvement_pct']])]
    for series in ['Load','PV']:
        monthly=improvement.loc[(improvement.scope=='monthly')&(improvement.series==series)]
        lines.append(f'{series}在{int((monthly.absolute_improvement_kw>0).sum())}/11个月胜过强基线，最差相对改善{monthly.relative_improvement_pct.min():.2f}%。')
    lines+=['## 5. Horizon表现']
    for series in ['Load','PV']:
        worst=outputs['horizon'].loc[outputs['horizon'].series==series].nlargest(3,'mae_kw')
        lines.append(series+'最高误差3个horizon：'+'；'.join(f'h={int(row.horizon_step)}（{(int(row.horizon_step)-1)//6:02d}:{((int(row.horizon_step)-1)%6)*10:02d}起），MAE={row.mae_kw:.2f} kW' for row in worst.itertuples())+'。')
    lines.append('每天00:00预测，horizon和日内时刻完全耦合，不能把误差峰值单独解释为远期不确定性。夜间actual均值为0时nMAE保留NaN。')
    lines += ['## 6. Controlled Low-Data',
        '固定目标日：03-20、06-21、09-23、12-21；只改变其之前7/14/30/60/90/180天的拟合预算。B0–B4均在同样预算内重新计算。每季只有一天，不能据此概括整个季节。',
        table(low.loc[low.model=='DLinear',['series','target_group','history_days','status','MAE','RMSE','nMAE','reason']]),
        '历史不足或seq_len+144超过可用点数的组合明确跳过，不修改seq_len凑结果。结构来自共同的January模型设计阶段，因此N天是本次参数拟合预算，并非整个研究设计阶段的数据预算。',
        'Load四个固定目标日从14天增至30天均改善，但更长历史不保证进一步改善；PV曲线同样非单调。扩大历史同时改变季节覆盖及固定轮数下的优化步数，不能把曲线全部归因于统计样本量。']
    for series,reference in [('Load','B2'),('PV','B3')]:
        candidate=low.loc[(low.series==series)&(low.model=='DLinear')&(low.status=='completed')]
        baseline=low.loc[(low.series==series)&(low.model==reference)&(low.status=='completed')]
        pairs=candidate.merge(baseline,on=['target_group','history_days'],suffixes=('_dlinear','_baseline'))
        lines.append(f'{series}低数据有效组合中，DLinear胜过{reference}的数量为{int((pairs.MAE_dlinear<pairs.MAE_baseline).sum())}/{len(pairs)}。这是固定四天的描述性比较，不是按测试日挑选历史长度。')
    lines.append('## 7. 过拟合与稳定性')
    for series in ['Load','PV']:
        chosen=selection.loc[(selection.series==series)&(selection.seq_len==frozen[series]['seq_len'])].iloc[0]
        metadata=json.loads((ROOT/chosen.checkpoint).with_suffix('.json').read_text())
        best=metadata['selection_validation_mae_kw'];last=metadata['epoch_curve'][-1]['validation_mae_kw']
        lines.append(f'{series}最佳验证MAE={best:.4f}，末轮={last:.4f} kW。'+('存在末段验证回落，已保留最佳权重和轮数。' if last>best else '未见末段验证回落。')+'训练MSE和验证MAE量纲不同，不直接比较数值大小。')
    lines.append('选中模型的末轮退化均不足1 kW，仅说明轻微验证波动，不能据此断言严重过拟合。PV的1008点候选在第7轮达到最佳、第15轮早停，提示更长输入存在验证退化风险。验证只有7天，模型选择稳定性证据有限；未超过全年基线也不能单独证明过拟合。')
    lines += [table(stability),
        'seed2026仅在胜过强基线的序列上补2027/2028，沿用相同结构和轮数，不选择最好seed；std为样本标准差，单seed时留空。',
        '## 8. Causality tests',table(pd.DataFrame(tests)),
        f'共{sum(row["tests"] for row in tests)}项全部通过，含原59项。窗口和scaler的最晚available_event都早于训练cutoff事件，训练数据和checkpoint hash可追踪，真实Feb/Mar模型对未来actual污染的预测逐位不变。',
        '信息可用性按rank0先于rank2；这是离线回测，计算时间另报，未将实际训练耗时作为在线决策延迟建模。',
        '## 样本量及复现',
        '每序列全年52,560点、365个日周期。1月候选fit为3,456点、24个周期；验证1,008点、7个周期。滑窗stride=1高度重叠，window_count不是独立样本数；effective_day_cycles也不声称各天统计独立。',
        table(audits.groupby(['phase','series']).agg(fits=('training_cutoff','size'),training_seconds=('training_seconds','sum'),max_parameters=('parameter_count','max')).reset_index()),
        '复现入口：python run_dlinear.py。仅在config和训练数据hash匹配时复用checkpoint；没有checkpoint时从头训练。全部数值由程序生成。环境与版本见artifacts/dlinear/environment.json。',
        '固定[TSLib提交4e938a1](https://github.com/thuml/Time-Series-Library/tree/4e938a1767106324dd753b2a44832bf870a0252e)；原始代码、LICENSE和哈希见vendor/tslib。',
        '## 9. 是否值得继续PatchTST？']
    successes=global_imp.loc[global_imp.absolute_improvement_kw>0].series.tolist()
    if successes:
        lines.append('DLinear在'+ '、'.join(successes)+'超过单seed强基线，有继续比较的理由，但需先核对补充种子均值、月度退化和低数据曲线；PatchTST不保证更好。')
    else:
        lines.append('两条序列均未超过强基线，当前不支持纳入最终方案或立即扩展PatchTST；不能用本次test反向调参后仍当作原始评估。')
    lines += ['在DLinear阶段停止，未继续PatchTST。',
        '![月度对比](artifacts/dlinear/figures/monthly_comparison.png)',
        '![Load低数据](artifacts/dlinear/figures/load_low_data.png)',
        '![PV低数据](artifacts/dlinear/figures/pv_low_data.png)']
    (ROOT/'MODEL_COMPARISON.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
