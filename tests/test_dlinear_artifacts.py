import json
import numpy as np
import pandas as pd
import pytest
from seasonal import EventTime
from dlinear_core import ROOT, OUT, settings, canonical, allowed_history, frame_hash, sha, DLinearForecaster

@pytest.fixture(scope='module')
def actuals(): return canonical()

@pytest.fixture(scope='module')
def predictions():
    frame=pd.read_csv(OUT/'predictions_seed2026.csv')
    for column in ['decision_time','forecast_date','interval_start','interval_end','context_available_at','scaler_fit_end','model_training_cutoff']:
        frame[column]=pd.to_datetime(frame[column],format='mixed')
    return frame

def test_operational_grid_and_events(predictions):
    assert len(predictions)==2*334*144
    assert not predictions.duplicated(['series','forecast_date','target_slot']).any()
    assert (predictions.interval_start>=predictions.decision_time).all()
    assert (predictions.interval_start==predictions.forecast_date+pd.to_timedelta(predictions.target_slot*10,unit='min')).all()
    assert (predictions.interval_end-predictions.interval_start==pd.Timedelta(minutes=10)).all()
    for column,rank in [('context_available_at','context_available_rank'),('scaler_fit_end','scaler_fit_rank')]:
        assert ((predictions[column]<predictions.decision_time)|((predictions[column]==predictions.decision_time)&(predictions[rank]<predictions.decision_rank))).all()
    assert (predictions.scaler_fit_end==predictions.model_training_cutoff).all()
    assert (predictions.scaler_fit_rank< predictions.model_training_cutoff_rank).all()
    assert (predictions.model_training_cutoff<=predictions.decision_time).all()
    assert np.isfinite(predictions.pred_power_kw).all()
    grouped=predictions.groupby(['series',predictions.forecast_date.dt.month])
    assert (grouped.checkpoint_sha256.nunique()==1).all()
    assert (grouped.model_training_cutoff.nunique()==1).all()
    assert predictions.checkpoint_sha256.nunique()==22

def test_every_checkpoint_training_and_scaler_hash(actuals):
    audit=pd.read_csv(OUT/'training_sample_audit.csv')
    assert len(audit)>=28
    for row in audit.itertuples():
        meta=json.loads((ROOT/row.checkpoint_path).with_suffix('.json').read_text())
        assert sha(ROOT/row.checkpoint_path)==row.checkpoint_sha256
        history=allowed_history(actuals,row.series,row.training_cutoff,int(row.available_days))
        assert row.raw_points==len(history.frame)==row.available_days*144
        assert row.window_count==row.raw_points-row.seq_len-144+1
        assert row.effective_day_cycles==row.available_days
        assert row.training_data_hash==row.scaler_data_hash==frame_hash(history.frame)
        assert EventTime(row.last_window_target_end,int(row.last_window_target_available_rank))<EventTime(row.training_cutoff,2)
        assert EventTime(row.scaler_fit_end,int(row.scaler_fit_rank))<EventTime(row.training_cutoff,2)
        assert meta['scaler_mean']==float(history.frame.power_kw.to_numpy().mean())
        assert meta['scaler_std']==float(history.frame.power_kw.to_numpy().std(ddof=0))

@pytest.mark.parametrize('series',['Load','PV'])
@pytest.mark.parametrize('day',['2025-02-01','2025-03-01'])
def test_real_checkpoint_future_contamination(actuals,predictions,series,day):
    subset=predictions.loc[(predictions.series==series)&(predictions.forecast_date==pd.Timestamp(day))].sort_values('target_slot')
    learner=DLinearForecaster.load(ROOT/subset.checkpoint_path.iloc[0])
    original=learner.predict(allowed_history(actuals,series,day))
    changed=actuals.copy();changed.loc[changed.source_day>=pd.Timestamp(day),'power_kw']=-987654321.0
    candidate=learner.predict(allowed_history(changed,series,day))
    np.testing.assert_array_equal(original,candidate)
    np.testing.assert_allclose(original,subset.pred_power_kw,rtol=1e-10,atol=1e-8)

def test_january_only_frozen_selection(actuals):
    frozen=json.loads((OUT/'frozen_selection.json').read_text())
    assert frozen['selection_source_hash']==frame_hash(actuals.loc[actuals.source_day<pd.Timestamp('2025-02-01')])
    scored=pd.read_csv(OUT/'january_validation.csv')
    assert len(scored)==6
    for series in ['Load','PV']:
        candidates=scored.loc[scored.series==series]
        assert set(candidates.seq_len)=={144,432,1008}
        best=candidates.sort_values(['validation_mae_kw','seq_len']).iloc[0]
        assert frozen[series]['seq_len']==best.seq_len
        assert frozen[series]['epochs']==best.best_epoch


def test_low_data_fixed_targets_and_skips():
    cfg=settings();metrics=pd.read_csv(OUT/'metrics_low_data.csv')
    assert len(metrics)==2*4*6*6
    assert not metrics.duplicated(['series','model','target_group','history_days']).any()
    pred=pd.read_csv(OUT/'predictions_low_data.csv')
    assert (pred.available_history_days==pred.history_days).all()
    for row in metrics.itertuples():
        assert row.target_date==cfg['low_data_dates'][row.target_group]
        subset=pred.loc[(pred.series==row.series)&(pred.model==row.model)&(pred.target_group==row.target_group)&(pred.history_days==row.history_days)]
        if row.status=='skipped': assert len(subset)==0 and pd.isna(row.MAE) and isinstance(row.reason,str)
        else:
            assert len(subset)==144
            assert pd.to_datetime(subset.history_start).min()==pd.Timestamp(row.target_date)-pd.Timedelta(days=row.history_days)
            assert np.isfinite(row.MAE)
    # At each valid budget all methods see exactly the same target actual vector.
    for _,group in pred.groupby(['series','target_group','history_days']):
        assert (group.groupby('target_slot').target_power_kw.nunique()==1).all()
