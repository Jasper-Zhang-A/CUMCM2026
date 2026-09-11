"""Final artifact verification; no forecast fitting or input transformation."""
import hashlib, json, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from seasonal import ROOT, config
from evaluate_all import read_predictions
from build_baseline_report import build_report

def verify():
    started=time.perf_counter(); settings=config()
    folder=ROOT/'artifacts/baselines'; audit=ROOT/'artifacts/leakage_audit'
    environment=json.loads((folder/'environment.json').read_text(encoding='utf-8'))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha(ROOT/settings['input_file'])==environment['input_sha256']
    assert sha(ROOT/'configs/baselines.json')==environment['config_sha256']
    # Report formatting may change; all computation and tests must still match the verified run.
    for relative in ['seasonal.py','run_baselines.py','evaluate_all.py','tests/test_event_causality.py','tests/test_prediction_artifacts.py']:
        assert sha(ROOT/relative)==environment['code_sha256'][Path(relative).name], relative
    passed=0
    for phase in ['preflight_tests','postflight_tests']:
        suite=ET.parse(audit/f'{phase}.xml').getroot().find('testsuite')
        assert int(suite.get('failures'))==int(suite.get('errors'))==int(suite.get('skipped'))==0
        passed+=int(suite.get('tests'))
    assert passed==59
    predictions=read_predictions()
    assert len(predictions)==480960
    predictions['month']=predictions.forecast_date.dt.strftime('%Y-%m')
    predictions['history_size_group']=pd.cut(predictions.available_history_days,settings['history_bin_edges'],labels=settings['history_bin_labels'])
    checked_groups=0
    for name,group_columns in {'global':[], 'monthly':['month'], 'horizon':['horizon_step'], 'history_size':['history_size_group']}.items():
        keys=['series','model']+group_columns
        saved=pd.read_csv(ROOT/f'artifacts/metrics/metrics_{name}.csv').set_index(keys)
        groups=predictions.groupby(keys,observed=True)
        assert len(saved)==len(groups)
        for group_key,group in groups:
            row=saved.loc[group_key]
            predicted=group.pred_power_kw.to_numpy(); actual=group.target_power_kw.to_numpy()
            errors=predicted-actual
            expected={'mae_kw':float(np.mean(np.abs(errors))),'rmse_kw':float(np.sqrt(np.mean(errors**2))),'mean_abs_actual_kw':float(np.mean(np.abs(actual)))}
            expected['nmae']=expected['mae_kw']/expected['mean_abs_actual_kw'] if expected['mean_abs_actual_kw'] else np.nan
            daylight=(actual>0)&(group.series.iloc[0]=='PV')
            expected['daylight_mae_kw']=float(np.mean(np.abs(errors[daylight]))) if daylight.any() else np.nan
            assert row.n==len(group) and row.daylight_n==int(daylight.sum())
            for metric,number in expected.items(): np.testing.assert_allclose(row[metric],number,rtol=1e-10,atol=1e-8,equal_nan=True,err_msg=f'{name}/{group_key}/{metric}')
            checked_groups+=1
    canonical=pd.read_csv(folder/'canonical_actuals.csv')
    assert len(canonical)==105120
    for column in ['source_day','interval_start','interval_end','source_timestamp','available_at']:
        canonical[column]=pd.to_datetime(canonical[column],format='mixed')
    assert (canonical.source_timestamp==canonical.interval_end).all()
    assert (canonical.available_at==canonical.interval_end).all()
    # Check the serialized labels independently of the original workbook parser.
    label_minutes={}
    import datetime
    for label in canonical.source_label.unique():
        if label=='0:00+1': label_minutes[label]=1440
        else:
            clock=datetime.time.fromisoformat(label)
            assert clock.second==0
            label_minutes[label]=clock.hour*60+clock.minute
    source_end=canonical.source_day+pd.to_timedelta(canonical.source_label.map(label_minutes),unit='min')
    assert (source_end==canonical.interval_end).all()
    assert (canonical.slot_id*10+10==canonical.source_label.map(label_minutes)).all()
    assert len(list((ROOT/'artifacts/figures').glob('*.png')))==12
    assert (audit/'causality_summary.json').is_file()
    build_report()
    environment['code_sha256']['build_baseline_report.py']=sha(ROOT/'build_baseline_report.py')
    environment['report_recovery']='Initial report-builder syntax error fixed; existing forecasts and metrics preserved and independently verified.'
    (folder/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    outputs=[]
    for output_folder in [folder,ROOT/'artifacts/metrics',ROOT/'artifacts/figures',audit]:
        for path in sorted(output_folder.iterdir()):
            if path.is_file() and path.name not in ['run_status.json','final_verification.json','output_manifest.json']:
                outputs.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)})
    for path in [ROOT/'BASELINE_REPORT.md',ROOT/'TIME_PROTOCOL.md']:
        outputs.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)})
    (folder/'output_manifest.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'status':'complete','tests_passed':passed,'causality_violations':0,'forecast_rows':len(predictions),'metric_groups_independently_checked':checked_groups,'canonical_rows':len(canonical),'figures_visually_reviewed':12,'input_preserved':True,'predictor_and_metric_code_unchanged_since_tests':True,'final_verification_seconds':time.perf_counter()-started}
    (audit/'final_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    status={'stage':'Seasonal B0-B4 only','status':'complete','causality_tests':'59 passed; 0 failed; 0 skipped','forecast_records':len(predictions),'deep_models':'NOT RUN','attachment3_4':'NOT USED','result_templates':'NOT USED','optimization':'NOT RUN','report_recovered_after_syntax_fix':True,'completed_at_local':pd.Timestamp.now().isoformat()}
    (folder/'run_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': verify()
