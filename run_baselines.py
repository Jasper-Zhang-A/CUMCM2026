"""One command: tests -> walk-forward -> complete-output audit -> metrics/figures/report."""
from pathlib import Path
import hashlib, importlib.metadata, json, os, platform, subprocess, sys, time, traceback
import numpy as np
import pandas as pd
from seasonal import ROOT, EventTime, ActualStore, SeasonalForecaster, config, load_actuals

AUDIT=ROOT/'artifacts/leakage_audit'

def run_tests(filename, phase):
    completed=subprocess.run([sys.executable,'-m','pytest',filename,'-q',f'--junitxml={AUDIT/phase}.xml'],cwd=ROOT,text=True,encoding='utf-8',capture_output=True)
    (AUDIT/f'{phase}.txt').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
    print(completed.stdout,flush=True)
    if completed.returncode: raise RuntimeError(f'Causality tests failed: {phase}; all downstream work stopped')

def walk_forward(settings):
    actuals=load_actuals(settings)
    actuals.to_csv(ROOT/'artifacts/baselines/canonical_actuals.csv',index=False)
    store=ActualStore(actuals)
    dates=pd.date_range(settings['first_forecast_day'],settings['last_forecast_day'])
    pending=ROOT/'artifacts/baselines/predictions.pending.csv'
    lineage=[]; runtimes=[]; first_write=True; total_records=0
    for day in dates:
        decision=EventTime(day,settings['decision_rank'])
        day_blocks=[]
        for series in settings['series_sheets']:
            history=store.history_before(series,decision)
            assert len(history.frame)==history.frame.source_day.nunique()*settings['horizon']
            expected_days=int((day-pd.Timestamp(settings['first_source_day'])).days)
            assert history.frame.source_day.nunique()==expected_days
            # All models finish prediction before any truth for this series/day is revealed.
            forecasts=[]
            for name in settings['models']:
                started=time.perf_counter()
                model=SeasonalForecaster(name).fit(history)
                prediction=model.predict(decision,settings['horizon'])
                used=model.used
                source_days=sorted(pd.Timestamp(source_day) for source_day in used.source_day.unique())
                assert all(source_day<day for source_day in source_days)
                max_available=EventTime(used.available_at.max(),int(used.event_rank_available.max()))
                assert max_available<decision
                if name in ['B1','B2']:
                    assert source_days==[day-pd.Timedelta(days=1 if name=='B1' else 7)]
                elif name=='B3': assert source_days==list(pd.date_range(day-pd.Timedelta(days=7),day-pd.Timedelta(days=1)))
                elif name=='B4': assert source_days==[past for past in pd.date_range(settings['first_source_day'],day-pd.Timedelta(days=1)) if past.dayofweek==day.dayofweek]
                else:
                    assert len(used)==1 and used.interval_end.iloc[0]==day and used.slot_id.iloc[0]==143
                lineage_id=f'{series}:{name}:{day.date()}'
                forecasts.append((name,prediction.copy(),lineage_id,max_available))
                lineage.append({'lineage_id':lineage_id,'series':series,'model':name,'decision_time':day,'decision_rank':decision.rank,
                    'used_source_days':json.dumps([str(source_day.date()) for source_day in source_days]),
                    'used_observations':len(used),'input_interval_start_min':used.interval_start.min(),
                    'input_interval_end_max':used.interval_end.max(),'input_available_at_max':max_available.timestamp,
                    'input_event_rank_max':max_available.rank,'available_history_days':expected_days,
                    'scaler':'none','full_year_statistics_used':False})
                runtimes.append({'series':series,'model':name,'forecast_date':day,'fit_predict_seconds':time.perf_counter()-started,'selected_observations':len(used)})
            truth=store.reveal_targets(series,decision)
            assert len(truth)==settings['horizon']
            assert (truth.interval_start>=day).all()
            assert (truth.source_day==day).all()
            for name,prediction,lineage_id,max_available in forecasts:
                block=pd.DataFrame({'series':series,'model':name,'forecast_date':day,'decision_time':day,'decision_rank':decision.rank,
                    'target_slot':truth.slot_id.to_numpy(),'horizon_step':truth.slot_id.to_numpy()+1,
                    'interval_start':truth.interval_start.to_numpy(),'interval_end':truth.interval_end.to_numpy(),
                    'target_power_kw':truth.power_kw.to_numpy(),'pred_power_kw':prediction,
                    'history_start':history.frame.interval_start.min(),'history_end':history.frame.interval_end.max(),
                    'history_end_rank':0,'available_history_days':expected_days,
                    'input_available_at':max_available.timestamp,'input_available_rank':max_available.rank,'lineage_id':lineage_id})
                day_blocks.append(block)
        day_frame=pd.concat(day_blocks,ignore_index=True)
        day_frame.to_csv(pending,index=False,mode='w' if first_write else 'a',header=first_write,float_format='%.12g')
        first_write=False; total_records+=len(day_frame)
        if day.day==1 or day==dates[-1]: print(f'Forecasts frozen through {day.date()}; {total_records:,} records',flush=True)
    pd.DataFrame(lineage).to_csv(ROOT/'artifacts/baselines/lineage.csv',index=False)
    pd.DataFrame(runtimes).to_csv(ROOT/'artifacts/baselines/runtime.csv',index=False)
    pending.replace(ROOT/'artifacts/baselines/predictions.csv')
    return total_records

def main():
    settings=config(); start=time.perf_counter()
    AUDIT.mkdir(exist_ok=True,parents=True)
    status_path=ROOT/'artifacts/baselines/run_status.json'
    status={'stage':'Seasonal B0-B4 only','status':'running','causality_tests':'pending','deep_models':'NOT RUN'}
    status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
    try:
        assert settings['models']==list(SeasonalForecaster.MODELS)
        assert settings['scaler'] is None and not settings['allow_deep_models'] and not settings['allow_optimization'] and not settings['allow_result_workbook_writes']
        np.random.seed(settings['seed'])
        before_hash=hashlib.sha256((ROOT/settings['input_file']).read_bytes()).hexdigest()
        run_tests('tests/test_event_causality.py','preflight_tests')
        records=walk_forward(settings)
        run_tests('tests/test_prediction_artifacts.py','postflight_tests')
        (AUDIT/'causality_summary.json').write_text(json.dumps({'violations':0,'forecast_records':records,'forecast_origins_per_series_model':334,
            'event_order':'(available_at,rank_available) < (decision_time,2)', 'test_reports':['preflight_tests.xml','postflight_tests.xml'],
            'scalers_used':False,'future_actuals_used':False,'attachment3_used':False,'attachment4_used':False,'result_templates_used':False},indent=2),encoding='utf-8')
        from evaluate_all import evaluate
        evaluate()
        assert before_hash==hashlib.sha256((ROOT/settings['input_file']).read_bytes()).hexdigest()
        environment={'python':sys.version,'platform':platform.platform(),'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','openpyxl','pytest','matplotlib']},
            'seed':settings['seed'],'pytorch':'not used','TSLib_commit':'not used','CUDA_GPU':'not used (CPU deterministic baselines)',
            'trainable_neural_parameters':0,'scaler':'none','input_sha256':before_hash,
            'config_sha256':hashlib.sha256((ROOT/'configs/baselines.json').read_bytes()).hexdigest(),
            'code_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/'seasonal.py',ROOT/'run_baselines.py',ROOT/'evaluate_all.py',ROOT/'build_baseline_report.py',ROOT/'tests/test_event_causality.py',ROOT/'tests/test_prediction_artifacts.py']},
            'elapsed_seconds_before_report':time.perf_counter()-start}
        (ROOT/'artifacts/baselines/environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
        from build_baseline_report import build_report
        build_report()
        status.update(status='complete',causality_tests='all passed',forecast_records=records,elapsed_seconds=time.perf_counter()-start)
    except Exception:
        status.update(status='failed',error=traceback.format_exc())
        raise
    finally: status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Completed Seasonal B0-B4. No deep models, interpolation, result templates, or optimization executed.',flush=True)

if __name__=='__main__': main()
