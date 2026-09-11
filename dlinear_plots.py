import json
import numpy as np
import pandas as pd
from seasonal import ROOT
from dlinear_core import OUT, settings

def plots(frame,outputs,low,selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.2,'savefig.dpi':160})
    base=pd.read_csv(ROOT/'artifacts/baselines/predictions.csv');base['forecast_date']=pd.to_datetime(base.forecast_date)
    references={'Load':'B2','PV':'B3'};folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for series,reference in references.items():
        day=pd.Timestamp('2025-02-01')
        subset=frame.loc[(frame.series==series)&(frame.forecast_date==day)].sort_values('target_slot')
        seasonal=base.loc[(base.series==series)&(base.model==reference)&(base.forecast_date==day)].sort_values('target_slot')
        fig,axis=plt.subplots(figsize=(10,4),layout='constrained')
        axis.plot(subset.target_slot/6,subset.target_power_kw,label='Actual (post forecast)',color='#263547')
        axis.plot(subset.target_slot/6,seasonal.pred_power_kw,label=reference,color='#3b987d')
        axis.plot(subset.target_slot/6,subset.pred_power_kw,label='DLinear',color='#c15656')
        axis.set(title=f'{series}: fixed example 2025-02-01',xlabel='Interval start hour',ylabel='Power (kW)',xticks=range(0,25,3));axis.legend()
        fig.savefig(folder/f'{series.lower()}_vs_dlinear.png');plt.close(fig)
    for scope in ['monthly','horizon']:
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        source=pd.read_csv(ROOT/f'artifacts/metrics/metrics_{scope}.csv')
        for axis,series in zip(axes,['Load','PV']):
            reference=references[series]
            for label,color,subset in [('DLinear','#c15656',outputs[scope].loc[outputs[scope].series==series]),(reference,'#3b987d',source.loc[(source.series==series)&(source.model==reference)])]:
                coordinate=subset.month.str[-2:].astype(int) if scope=='monthly' else subset.horizon_step
                axis.plot(coordinate,subset.mae_kw,label=label,color=color)
            axis.set(title=f'{series}: {scope} MAE',xlabel='Month' if scope=='monthly' else 'Horizon (10 min)',ylabel='MAE (kW)');axis.legend()
        fig.savefig(folder/f'{scope}_comparison.png');plt.close(fig)
    cfg=settings()
    for series,reference in references.items():
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        for axis,(group,day) in zip(axes.flat,cfg['low_data_dates'].items()):
            for model,color in [('DLinear','#c15656'),(reference,'#3b987d')]:
                subset=low.loc[(low.series==series)&(low.model==model)&(low.target_group==group)].set_index('history_days').reindex(cfg['low_data_history_days'])
                axis.plot(np.arange(6),subset.MAE,marker='o',label=model,color=color)
            axis.set(title=f'{group}: {day}',xlabel='Training history days',ylabel='MAE (kW)',xticks=np.arange(6),xticklabels=cfg['low_data_history_days']);axis.legend()
        fig.suptitle(f'{series}: fixed targets, fixed January-selected structure',fontsize=13)
        fig.savefig(folder/f'{series.lower()}_low_data.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,series in enumerate(['Load','PV']):
        for selection_row in selection.loc[selection.series==series].itertuples():
            meta=json.loads((ROOT/selection_row.checkpoint).with_suffix('.json').read_text());curve=pd.DataFrame(meta['epoch_curve'])
            axes[row,0].plot(curve.epoch,curve.training_mse_scaled,label=f'L={selection_row.seq_len}')
            axes[row,1].plot(curve.epoch,curve.validation_mae_kw,label=f'L={selection_row.seq_len}')
        axes[row,0].set(title=f'{series}: training loss',xlabel='Epoch',ylabel='MSE (scaled)');axes[row,0].legend()
        axes[row,1].set(title=f'{series}: January validation',xlabel='Epoch',ylabel='MAE (kW)');axes[row,1].legend()
    fig.savefig(folder/'january_learning_curves.png');plt.close(fig)
