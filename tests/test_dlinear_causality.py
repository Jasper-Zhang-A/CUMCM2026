import numpy as np
import pandas as pd
import pytest
from seasonal import EventTime
from dlinear_core import canonical, allowed_history, CausalScaler, window_starts, frame_hash, train, DLinearForecaster

@pytest.fixture(scope='module')
def actuals(): return canonical()

@pytest.mark.parametrize('series',['Load','PV'])
@pytest.mark.parametrize('cutoff,days',[('2025-02-01',31),('2025-03-01',59),('2025-12-01',334)])
@pytest.mark.parametrize('seq_len',[144,432,1008])
def test_training_windows_end_before_cutoff_event(actuals,series,cutoff,days,seq_len):
    history=allowed_history(actuals,series,cutoff)
    starts=window_starts(history,seq_len)
    assert len(history.frame)==days*144
    assert len(starts)==days*144-seq_len-144+1
    ends=history.frame.iloc[starts+seq_len+143]
    assert (ends.source_day<pd.Timestamp(cutoff)).all()
    assert all(EventTime(timestamp,rank)<EventTime(cutoff,2) for timestamp,rank in ends[['available_at','event_rank_available']].itertuples(index=False,name=None))

@pytest.mark.parametrize('series',['Load','PV'])
def test_feb_excludes_feb_mar_includes_feb(actuals,series):
    feb=allowed_history(actuals,series,'2025-02-01').frame
    mar=allowed_history(actuals,series,'2025-03-01').frame
    assert set(feb.source_day.dt.month)=={1}
    assert set(mar.source_day.dt.month)=={1,2}
    assert feb.source_day.max()==pd.Timestamp('2025-01-31')
    assert mar.source_day.max()==pd.Timestamp('2025-02-28')

@pytest.mark.parametrize('series',['Load','PV'])
def test_scaler_poisoning_and_fit_event(actuals,series):
    first=CausalScaler().fit(allowed_history(actuals,series,'2025-02-01'))
    poisoned=actuals.copy();poisoned.loc[poisoned.source_day>=pd.Timestamp('2025-02-01'),'power_kw']=1e20
    second=CausalScaler().fit(allowed_history(poisoned,series,'2025-02-01'))
    assert first.mean==second.mean and first.std==second.std and first.data_hash==second.data_hash
    assert first.fit_event==EventTime('2025-02-01',0) and first.fit_event<EventTime('2025-02-01',2)
    assert first.mean!=poisoned.loc[poisoned.series==series].power_kw.mean()

@pytest.mark.parametrize('days',[7,14,30,60,90,180])
def test_low_data_budget_exact(actuals,days):
    history=allowed_history(actuals,'Load','2025-09-23',days)
    assert len(history.frame)==days*144
    assert history.frame.source_day.min()==pd.Timestamp('2025-09-23')-pd.Timedelta(days=days)
    if days==7:
        with pytest.raises(ValueError,match='context_plus_target'): window_starts(history,1008)

def test_unavailable_180_days_is_explicit(actuals):
    with pytest.raises(ValueError,match='insufficient_available_history'): allowed_history(actuals,'Load','2025-03-20',180)

@pytest.mark.parametrize('series',['Load','PV'])
def test_retraining_future_contamination_bitwise_and_checkpoint(actuals,tmp_path,series):
    history=allowed_history(actuals,series,'2025-02-01',7)
    first=train(history,144,2,2026,tmp_path/'first.pt','unit_test',device='cpu')
    contaminated=actuals.copy()
    future=contaminated.source_day>=pd.Timestamp('2025-02-01')
    contaminated.loc[future,'power_kw']=np.arange(future.sum())*1e9
    second_history=allowed_history(contaminated,series,'2025-02-01',7)
    second=train(second_history,144,2,2026,tmp_path/'second.pt','unit_test',device='cpu')
    assert first.training_hash==second.training_hash==frame_hash(history.frame)
    first_pred=first.predict(history);second_pred=second.predict(second_history)
    np.testing.assert_array_equal(first_pred,second_pred)
    restored=DLinearForecaster.load(tmp_path/'first.pt',device='cpu')
    np.testing.assert_array_equal(first_pred,restored.predict(history))
    assert restored.metadata['training_data_hash']==restored.metadata['scaler_data_hash']
    restored.metadata['selection_information_end']='2025-02-02'
    with pytest.raises(AssertionError,match='validation information'):
        restored.predict(history)
    with open(tmp_path/'first.pt','ab') as stream: stream.write(b'tampering')
    with pytest.raises(AssertionError): DLinearForecaster.load(tmp_path/'first.pt',device='cpu')
