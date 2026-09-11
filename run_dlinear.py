"""January-only selection, monthly retraining and fixed-target low-data experiment."""
import json, subprocess, sys, time, platform, os
import numpy as np
import pandas as pd
import torch
from seasonal import ROOT, EventTime, SeasonalForecaster
from dlinear_core import OUT, settings, canonical, allowed_history, train, frame_hash, sha
from evaluate_all import metrics


def tests(path,name):
    (OUT/'test_tmp').mkdir(parents=True,exist_ok=True)
    (OUT/'tests').mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pytest',*path,'-q',f'--junitxml={OUT}/tests/{name}.xml',f'--basetemp={OUT}/test_tmp/{name}']
    child_env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8'}
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env=child_env)
    (OUT/f'tests/{name}.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    print(result.stdout,flush=True)
    if result.returncode: raise RuntimeError('Causality tests failed; stop: '+name)


def select_models(actuals):
    cfg=settings();rows=[];frozen={}
    for series in ['Load','PV']:
        for length in cfg['seq_len_candidates']:
            history=allowed_history(actuals,series,cfg['selection_train_cutoff'])
            validation=[]
            for day in pd.date_range(cfg['selection_validation_start'],cfg['selection_validation_end']):
                context=allowed_history(actuals,series,day)
                truth=actuals.loc[(actuals.series==series)&(actuals.source_day==day)].power_kw.to_numpy()
                validation.append((context,truth))
            learner=train(history,length,cfg['max_selection_epochs'],cfg['seed'],OUT/f'checkpoints/selection_{series}_L{length}.pt','january_selection',validation)
            rows.append({'series':series,'seq_len':length,'validation_mae_kw':learner.metadata['selection_validation_mae_kw'],'best_epoch':learner.metadata['best_epoch'],'epochs_executed':learner.metadata['epochs_executed'],'train_points':len(history.frame),'validation_points':7*144,'window_count':learner.metadata['window_count'],'checkpoint':learner.metadata['checkpoint_path']})
        winner=min([row for row in rows if row['series']==series],key=lambda row:(row['validation_mae_kw'],row['seq_len']))
        frozen[series]={'seq_len':winner['seq_len'],'epochs':winner['best_epoch'],'validation_mae_kw':winner['validation_mae_kw']}
    frozen['selection_cutoff']=cfg['selection_final_cutoff'];frozen['config_sha256']=sha(ROOT/'configs/dlinear.json')
    frozen['selection_source_hash']=frame_hash(actuals.loc[actuals.source_day<pd.Timestamp(cfg['selection_final_cutoff'])])
    pd.DataFrame(rows).to_csv(OUT/'january_validation.csv',index=False)
    (OUT/'frozen_selection.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    from report_dlinear import table
    text=['# MODEL SELECTION — January only','只使用1月；1月1–24日训练，25–31日每日00:00验证。验证期权重固定，但每天使用当时已可见的最新context。scaler仅fit前24天。三种候选的训练滑窗目标均不越过01-25 rank2。',table(pd.DataFrame(rows)),
        '按7个验证日合并的MAE最小选seq_len，各候选用验证MAE早停；选择对应best epoch作为未来所有重训固定轮数。选定后不再根据2–12月修改结构、轮数、学习率或数据处理。',json.dumps(frozen,ensure_ascii=False,indent=2),
        '固定：TSLib DLinear原实现、moving_avg=25、Adam lr=0.0005、batch=256、MSE、stride=1、shuffle=False、max_epochs=50、patience=8、seed=2026。不裁剪负功率，不用真实日照掩码。',
        '月度重训使用该月之前全部历史，包括当时已经发生的原1月验证段；这是允许的refit。Controlled Low-Data仅fit目标前N天，但结构来自共同January设计阶段；N不是包含模型设计数据在内的整个研究数据预算。后续No-FT对照需同样披露设计/校准数据。',
        '7天预算若小于seq_len+pred_len所需长度，DLinear组合明确跳过；不能临时改seq_len来完成该组合。时间事件按TIME_PROTOCOL.md，00:00 rank0先于rank2，未添加epsilon。']
    (ROOT/'MODEL_SELECTION.md').write_text('\n\n'.join(text),encoding='utf-8')
    return frozen


def record_forecast(series,model,day,history,prediction,actuals,extra):
    assert len(prediction)==144 and np.isfinite(prediction).all()
    # Truth is joined only after prediction has been returned.
    truth=actuals.loc[(actuals.series==series)&(actuals.source_day==day)].sort_values('slot_id')
    assert len(truth)==144 and (truth.interval_start>=day).all()
    return pd.DataFrame({'series':series,'model':model,'forecast_date':day,'decision_time':day,'decision_rank':2,
        'target_slot':truth.slot_id.to_numpy(),'horizon_step':truth.slot_id.to_numpy()+1,
        'interval_start':truth.interval_start.to_numpy(),'interval_end':truth.interval_end.to_numpy(),
        'target_power_kw':truth.power_kw.to_numpy(),'pred_power_kw':prediction,
        'history_start':history.frame.interval_start.min(),'history_end':history.frame.interval_end.max(),
        'available_history_days':history.frame.source_day.nunique(),'context_available_at':history.frame.available_at.max(),'context_available_rank':0,**extra})


def operational(actuals,frozen,seed,series_list):
    cfg=settings();blocks=[]
    for series in series_list:
        chosen=frozen[series]
        for month in pd.date_range(cfg['operational_start'],cfg['operational_end'],freq='MS'):
            history=allowed_history(actuals,series,month)
            checkpoint=OUT/f'checkpoints/operational_{series}_{month:%Y%m}_seed{seed}.pt'
            learner=train(history,chosen['seq_len'],chosen['epochs'],seed,checkpoint,'operational')
            month_end=min(month+pd.offsets.MonthEnd(0),pd.Timestamp(cfg['operational_end']))
            for day in pd.date_range(month,month_end):
                context=allowed_history(actuals,series,day)
                prediction=learner.predict(context)
                blocks.append(record_forecast(series,'DLinear',day,context,prediction,actuals,{
                    'seed':seed,'seq_len':chosen['seq_len'],'model_training_cutoff':month,'model_training_cutoff_rank':2,
                    'scaler_fit_end':learner.scaler.fit_event.timestamp,'scaler_fit_rank':0,'training_data_hash':learner.training_hash,
                    'context_data_hash':frame_hash(context.frame.tail(chosen['seq_len'])),
                    'checkpoint_path':learner.metadata['checkpoint_path'],'checkpoint_sha256':learner.metadata['checkpoint_sha256']}))
    result=pd.concat(blocks,ignore_index=True)
    result.to_csv(OUT/f'predictions_seed{seed}.csv',index=False,float_format='%.12g')
    return result


def low_data(actuals,frozen):
    cfg=settings();blocks=[];status=[]
    for group,day_string in cfg['low_data_dates'].items():
        day=pd.Timestamp(day_string)
        for series in ['Load','PV']:
            for days in cfg['low_data_history_days']:
                for name in ['B0','B1','B2','B3','B4','DLinear']:
                    identity={'series':series,'model':name,'target_group':group,'target_date':day_string,'history_days':days}
                    try: history=allowed_history(actuals,series,day,days)
                    except ValueError as error:
                        status.append({**identity,'status':'skipped','reason':str(error)});continue
                    if name=='DLinear':
                        chosen=frozen[series]
                        if len(history.frame)<chosen['seq_len']+144:
                            status.append({**identity,'status':'skipped','reason':'insufficient_points_for_context_plus_target'});continue
                        learner=train(history,chosen['seq_len'],chosen['epochs'],cfg['seed'],OUT/f'checkpoints/low_{series}_{group}_N{days}.pt','controlled_low_data')
                        prediction=learner.predict(history)
                        extra={'checkpoint_path':learner.metadata['checkpoint_path'],'scaler_fit_end':learner.scaler.fit_event.timestamp,'scaler_fit_rank':0,'training_data_hash':learner.training_hash}
                    else:
                        learner=SeasonalForecaster(name).fit(history);prediction=learner.predict(history.decision,144)
                        extra={'checkpoint_path':'','scaler_fit_end':pd.NaT,'scaler_fit_rank':np.nan,'training_data_hash':frame_hash(learner.used)}
                    blocks.append(record_forecast(series,name,day,history,prediction,actuals,{**extra,'target_group':group,'history_days':days}))
                    status.append({**identity,'status':'completed','reason':''})
    predictions=pd.concat(blocks,ignore_index=True)
    predictions.to_csv(OUT/'predictions_low_data.csv',index=False,float_format='%.12g')
    statuses=pd.DataFrame(status);statuses.to_csv(OUT/'low_data_status.csv',index=False)
    scored=metrics(predictions,['series','model','target_group','history_days'])
    scored=scored.rename(columns={'mae_kw':'MAE','rmse_kw':'RMSE','nmae':'nMAE'})
    result=statuses.merge(scored,on=['series','model','target_group','history_days'],how='left',validate='one_to_one')
    result.to_csv(OUT/'metrics_low_data.csv',index=False)


def write_sample_audit():
    rows=[]
    for path in sorted((OUT/'checkpoints').glob('*.json')):
        meta=json.loads(path.read_text(encoding='utf-8'))
        rows.append({key:meta.get(key) for key in ['phase','series','seed','seq_len','training_cutoff','raw_points','available_days','window_count','effective_day_cycles','training_start','training_end','last_window_target_end','last_window_target_available_rank','scaler_fit_end','scaler_fit_rank','training_data_hash','scaler_data_hash','checkpoint_path','checkpoint_sha256','parameter_count','best_epoch','training_seconds']})
    pd.DataFrame(rows).to_csv(OUT/'training_sample_audit.csv',index=False)


def main():
    started=time.perf_counter();cfg=settings();status_path=OUT/'run_status.json'
    status_path.write_text(json.dumps({'status':'running','stage':'DLinear only'}),encoding='utf-8')
    tests(['tests/test_event_causality.py','tests/test_prediction_artifacts.py'],'seasonal_regression')
    tests(['tests/test_dlinear_causality.py'],'learned_preflight')
    actuals=canonical()
    frozen=select_models(actuals)
    forecasts=operational(actuals,frozen,cfg['seed'],['Load','PV'])
    summary=metrics(forecasts,['series','model']).set_index('series')
    baselines=pd.read_csv(ROOT/'artifacts/metrics/metrics_global.csv')
    improved=[]
    for series,reference in [('Load','B2'),('PV','B3')]:
        baseline=baselines.loc[(baselines.series==series)&(baselines.model==reference)].mae_kw.iloc[0]
        if summary.loc[series,'mae_kw']<baseline: improved.append(series)
    # This trigger adds stability evaluation; it never changes architecture or seed-2026 predictions.
    (OUT/'multiseed_trigger.json').write_text(json.dumps({'series_improved_seed2026':improved,'rule':'global all-day MAE strictly better than fixed strong baseline; structure remains January frozen'},indent=2),encoding='utf-8')
    for seed in cfg['additional_seeds_if_improved']:
        if improved: operational(actuals,frozen,seed,improved)
    low_data(actuals,frozen)
    write_sample_audit()
    tests(['tests/test_dlinear_artifacts.py'],'learned_postflight')
    from report_dlinear import generate
    generate()
    environment={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'seed':cfg['seed'],'tslib':json.loads((ROOT/'vendor/tslib/PROVENANCE.json').read_text()),'canonical_sha256':sha(ROOT/cfg['canonical_file']),'config_sha256':sha(ROOT/'configs/dlinear.json'),'code_sha256':{name:sha(ROOT/name) for name in ['dlinear_core.py','run_dlinear.py','report_dlinear.py']},'total_seconds':time.perf_counter()-started}
    (OUT/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    status_path.write_text(json.dumps({'status':'complete','stage':'DLinear only','seconds':time.perf_counter()-started,'models_not_run':['PatchTST','TimesNet','iTransformer'],'causality':'all tests passed'},indent=2),encoding='utf-8')
    print('DLinear stage complete. STOP.',flush=True)

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        (OUT/'run_status.json').write_text(json.dumps({'status':'failed','stage':'DLinear only','error':repr(error)},indent=2),encoding='utf-8')
        raise
