"""Independent audits of every saved forecast and its input provenance."""
import json
import numpy as np
import pandas as pd
import pytest
from seasonal import ROOT, config, load_actuals
from evaluate_all import read_predictions, metrics

@pytest.fixture(scope='module')
def predictions(): return read_predictions()

@pytest.fixture(scope='module')
def actuals(): return load_actuals()

@pytest.fixture(scope='module')
def lineage():
    frame=pd.read_csv(ROOT/'artifacts/baselines/lineage.csv')
    for column in ['decision_time','input_interval_start_min','input_interval_end_max','input_available_at_max']:
        frame[column]=pd.to_datetime(frame[column],format='mixed')
    return frame

def test_all_dates_models_slots_present(predictions):
    assert len(predictions)==334*144*5*2
    assert not predictions.duplicated(['series','model','forecast_date','target_slot']).any()
    counts=predictions.groupby(['series','model','forecast_date']).target_slot.agg(['size','min','max'])
    assert len(counts)==334*5*2
    assert (counts['size']==144).all() and (counts['min']==0).all() and (counts['max']==143).all()
    assert set(predictions.series)=={'Load','PV'} and set(predictions.model)=={'B0','B1','B2','B3','B4'}
    assert sorted(predictions.forecast_date.unique())==list(pd.date_range('2025-02-01','2025-12-31'))
    assert np.isfinite(predictions[['target_power_kw','pred_power_kw']].to_numpy()).all()

def test_all_saved_availability_events_precede_decision(predictions):
    for time_column,rank_column in [('history_end','history_end_rank'),('input_available_at','input_available_rank')]:
        assert ((predictions[time_column]<predictions.decision_time)|((predictions[time_column]==predictions.decision_time)&(predictions[rank_column]<predictions.decision_rank))).all()
    assert (predictions.history_end==predictions.decision_time).all()
    assert (predictions.history_end_rank==0).all() and (predictions.decision_rank==2).all()
    assert (predictions.interval_start>=predictions.decision_time).all()
    assert (predictions.interval_end-predictions.interval_start==pd.Timedelta(minutes=10)).all()
    assert (predictions.interval_start==predictions.forecast_date+pd.to_timedelta(predictions.target_slot*10,unit='min')).all()
    assert (predictions.horizon_step==predictions.target_slot+1).all()
    assert (predictions.available_history_days==(predictions.forecast_date-pd.Timestamp('2025-01-01')).dt.days).all()

def test_lineage_all_origins_exact_source_days(predictions,lineage):
    assert len(lineage)==3340 and not lineage.lineage_id.duplicated().any()
    assert set(predictions.lineage_id)==set(lineage.lineage_id)
    assert (lineage.scaler=='none').all()
    assert not lineage.full_year_statistics_used.any()
    indexed=lineage.set_index('lineage_id')
    maximum_events=indexed.loc[predictions.lineage_id,'input_available_at_max'].reset_index(drop=True)
    assert maximum_events.equals(predictions.input_available_at)
    for row in lineage.itertuples():
        days=[pd.Timestamp(day) for day in json.loads(row.used_source_days)]
        decision=row.decision_time
        assert all(day<decision for day in days)
        if row.model in ['B0','B1']: expected=[decision-pd.Timedelta(days=1)]
        elif row.model=='B2': expected=[decision-pd.Timedelta(days=7)]
        elif row.model=='B3': expected=list(pd.date_range(decision-pd.Timedelta(days=7),decision-pd.Timedelta(days=1)))
        else: expected=[day for day in pd.date_range('2025-01-01',decision-pd.Timedelta(days=1)) if day.dayofweek==decision.dayofweek]
        assert days==expected
        assert row.used_observations==(1 if row.model=='B0' else len(days)*144)
        assert row.input_interval_end_max<=decision
        assert row.input_available_at_max==row.input_interval_end_max
        assert row.input_event_rank_max==0
        assert row.input_interval_start_min==(decision-pd.Timedelta(minutes=10) if row.model=='B0' else min(days))
        assert row.input_interval_end_max==max(days)+pd.Timedelta(days=1)

@pytest.mark.parametrize('series',['Load','PV'])
@pytest.mark.parametrize('model',['B0','B1','B2','B3','B4'])
def test_every_prediction_against_independent_array_formula(predictions,actuals,series,model):
    frame=actuals.loc[actuals.series==series].sort_values(['source_day','slot_id'])
    power_matrix=frame.power_kw.to_numpy().reshape(365,144)
    days=pd.date_range('2025-01-01','2025-12-31')
    expected=[]
    for day_index in range(31,365):
        if model=='B0': pred=np.repeat(power_matrix[day_index-1,-1],144)
        elif model=='B1': pred=power_matrix[day_index-1].copy()
        elif model=='B2': pred=power_matrix[day_index-7].copy()
        elif model=='B3': pred=sum(power_matrix[day_index-lag] for lag in range(1,8))/7
        else:
            selected=[index for index in range(day_index) if days[index].dayofweek==days[day_index].dayofweek]
            pred=sum(power_matrix[index] for index in selected)/len(selected)
        expected.append(pred)
    subset=predictions.loc[(predictions.series==series)&(predictions.model==model)].sort_values(['forecast_date','target_slot'])
    np.testing.assert_allclose(subset.pred_power_kw.to_numpy(),np.asarray(expected).ravel(),rtol=1e-10,atol=1e-8)
    np.testing.assert_array_equal(subset.target_power_kw.to_numpy(),power_matrix[31:].ravel())

def test_metrics_zero_denominator_and_daylight_mask():
    toy=pd.DataFrame({'series':['PV']*4,'model':['B0']*4,'pred_power_kw':[2.,3.,1.,6.],'target_power_kw':[0.,0.,2.,4.]})
    output=metrics(toy,['series','model']).iloc[0]
    assert output.mae_kw==2 and output.rmse_kw==pytest.approx(np.sqrt(4.5))
    assert output.nmae==pytest.approx(2/1.5)
    assert output.daylight_mae_kw==1.5 and output.daylight_n==2
    night=toy.iloc[:2]
    output=metrics(night,['series','model']).iloc[0]
    assert np.isnan(output.nmae) and np.isnan(output.daylight_mae_kw)
    assert not output.nmae_defined
