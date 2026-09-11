"""Metrics and figures, executed only after predictions are frozen and audited."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from seasonal import ROOT, config

DATE_COLUMNS=['forecast_date','decision_time','interval_start','interval_end','history_start','history_end','input_available_at']

def read_predictions():
    frame=pd.read_csv(ROOT/'artifacts/baselines/predictions.csv')
    for column in DATE_COLUMNS: frame[column]=pd.to_datetime(frame[column],format='mixed')
    return frame

def metrics(frame, grouping):
    records=frame.copy()
    errors=records.pred_power_kw-records.target_power_kw
    records['absolute_error_kw']=errors.abs()
    records['squared_error_kw2']=errors**2
    records['absolute_actual_kw']=records.target_power_kw.abs()
    # Settlement-only mask. This module is never imported by a forecaster.
    records['daylight_absolute_error_kw']=records.absolute_error_kw.where((records.series=='PV')&(records.target_power_kw>0))
    result=records.groupby(grouping,observed=True,sort=True).agg(
        n=('target_power_kw','size'),mae_kw=('absolute_error_kw','mean'),
        mse_kw2=('squared_error_kw2','mean'),mean_abs_actual_kw=('absolute_actual_kw','mean'),
        daylight_n=('daylight_absolute_error_kw','count'),daylight_mae_kw=('daylight_absolute_error_kw','mean')).reset_index()
    result['rmse_kw']=np.sqrt(result.pop('mse_kw2'))
    result['nmae']=result.mae_kw/result.mean_abs_actual_kw.replace(0,np.nan)
    result['all_day_mae_kw']=result.mae_kw.where(result.series=='PV')
    result['nmae_defined']=result.mean_abs_actual_kw>0
    return result

def evaluate():
    settings=config(); frame=read_predictions()
    frame['month']=frame.forecast_date.dt.strftime('%Y-%m')
    frame['history_size_group']=pd.cut(frame.available_history_days,settings['history_bin_edges'],labels=settings['history_bin_labels'],right=True)
    assert frame.history_size_group.notna().all()
    folder=ROOT/'artifacts/metrics'; folder.mkdir(exist_ok=True,parents=True)
    definitions={'global':[], 'monthly':['month'], 'horizon':['horizon_step'],
        'history_size':['history_size_group'], 'daily':['forecast_date','available_history_days']}
    outputs={}
    for name,extra in definitions.items():
        outputs[name]=metrics(frame,['series','model']+extra)
        outputs[name].to_csv(folder/f'metrics_{name}.csv',index=False,float_format='%.12g')
    # Exact low-data point: forecast on Feb 1, with 31 available days.
    outputs['31_days']=metrics(frame.loc[frame.available_history_days==31],['series','model'])
    outputs['31_days'].to_csv(folder/'metrics_exact_31_days.csv',index=False,float_format='%.12g')
    specification={'MAE':'mean(abs(pred_power_kw-target_power_kw))',
        'RMSE':'sqrt(mean((pred_power_kw-target_power_kw)^2))',
        'nMAE':'MAE / mean(abs(target_power_kw)) within the same group; NaN if denominator=0',
        'daylight':'PV target_power_kw>0, settlement only, not available to models',
        'aggregation':'All slots pooled, no unweighted average of monthly metrics',
        'horizon_step':'target_slot+1; h1=[00:00,00:10), h144=[23:50,24:00)',
        'history_days':'complete available days, not necessarily days used by a specific model',
        'test_selection':'No parameter or model selected by test metrics; descriptive comparison only'}
    (folder/'metric_definitions.json').write_text(json.dumps(specification,ensure_ascii=False,indent=2),encoding='utf-8')
    figures(frame,outputs,settings)
    return outputs

def figures(frame,outputs,settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.18,'savefig.dpi':180})
    folder=ROOT/'artifacts/figures'; folder.mkdir(exist_ok=True,parents=True)
    colors={'B0':'#88939e','B1':'#d28a25','B2':'#2671af','B3':'#bd536c','B4':'#398770'}
    example_day=pd.Timestamp(settings['example_forecast_day'])
    for series in ['Load','PV']:
        example=frame.loc[(frame.series==series)&(frame.forecast_date==example_day)]
        for model in ['B1','B2','B3']:
            subset=example.loc[example.model==model].sort_values('target_slot')
            fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
            hour=subset.target_slot/6
            ax.plot(hour,subset.target_power_kw,color='#202d3a',label='Actual (revealed after forecast)',linewidth=1.5)
            ax.plot(hour,subset.pred_power_kw,color=colors[model],label=model,linewidth=1.5)
            ax.set(xlim=(0,24),xticks=range(0,25,3),xlabel='Interval start hour',ylabel='Power (kW)',title=f'{series}: actual vs {model} | {example_day.date()} | 31 history days')
            ax.legend(fontsize=9); fig.savefig(folder/f'{series.lower()}_actual_vs_{model.lower()}.png'); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
        for model in settings['models']:
            subset=outputs['monthly'].loc[(outputs['monthly'].series==series)&(outputs['monthly'].model==model)]
            ax.plot(subset.month.str[-2:].astype(int),subset.mae_kw,label=model,color=colors[model],marker='o',markersize=3)
        ax.set(xlabel='Forecast month (2025)',ylabel='MAE (kW)',title=f'{series}: monthly MAE',xticks=range(2,13));ax.legend(ncol=5)
        fig.savefig(folder/f'{series.lower()}_monthly_mae.png');plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
        for model in settings['models']:
            subset=outputs['horizon'].loc[(outputs['horizon'].series==series)&(outputs['horizon'].model==model)]
            ax.plot(subset.horizon_step,subset.mae_kw,label=model,color=colors[model])
        ax.set(xlabel='Forecast horizon h (h=1 starts at 00:00)',ylabel='MAE (kW)',title=f'{series}: error by horizon',xlim=(1,144));ax.legend(ncol=5)
        fig.savefig(folder/f'{series.lower()}_horizon_mae.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained',sharex='col')
    for row,series in enumerate(['Load','PV']):
        for model in settings['models']:
            daily=outputs['daily'].loc[(outputs['daily'].series==series)&(outputs['daily'].model==model)].sort_values('available_history_days')
            # Only trailing smoothing for the display; does not enter the predictor.
            axes[row,0].plot(daily.available_history_days,daily.mae_kw.rolling(7,min_periods=1).mean(),color=colors[model],label=model)
            history=outputs['history_size'].loc[(outputs['history_size'].series==series)&(outputs['history_size'].model==model)].set_index('history_size_group').reindex(settings['history_bin_labels'])
            axes[row,1].plot(np.arange(4),history.mae_kw,color=colors[model],label=model,marker='o')
        axes[row,0].set(title=f'{series}: trailing 7-day evaluation MAE',ylabel='MAE (kW)',xlabel='Available complete history days')
        axes[row,1].set(title=f'{series}: pooled MAE by history size',ylabel='MAE (kW)',xlabel='Available history days',xticks=np.arange(4),xticklabels=['31-60','61-120','121-240','>240'])
    axes[0,0].legend(ncol=5,fontsize=8);fig.savefig(folder/'mae_vs_available_history_days.png');plt.close(fig)
    # A compact inspectable overview for the report.
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for axis,series in zip(axes,['Load','PV']):
        summary=outputs['global'].loc[outputs['global'].series==series].set_index('model').reindex(settings['models'])
        bars=axis.bar(summary.index,summary.mae_kw,color=[colors[model] for model in summary.index])
        axis.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        axis.set(title=f'{series}: Feb-Dec MAE',ylabel='MAE (kW)',ylim=(0,summary.mae_kw.max()*1.18))
    fig.savefig(folder/'baseline_overview.png');plt.close(fig)

if __name__=='__main__':
    status=json.loads((ROOT/'artifacts/leakage_audit/causality_summary.json').read_text())
    assert status['violations']==0
    evaluate()
