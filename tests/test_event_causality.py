import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from seasonal import EventTime, CausalHistory, ActualStore, SeasonalForecaster, config, load_actuals

@pytest.fixture(scope='session')
def actuals(): return load_actuals()

@pytest.fixture(scope='session')
def synthetic():
    starts=pd.date_range('2025-01-01',periods=38*144,freq='10min')
    slots=np.arange(len(starts))%144
    day_index=np.arange(len(starts))//144
    return pd.DataFrame({'series':'Load','source_day':starts.normalize(),'slot_id':slots,
        'interval_start':starts,'interval_end':starts+pd.Timedelta(minutes=10),
        'source_label':[f'{(slot+1)//6:02d}:{(slot+1)%6*10:02d}' for slot in slots],
        'source_timestamp':starts+pd.Timedelta(minutes=10),
        'power_kw':day_index*1000.0+slots,
        'available_at':starts+pd.Timedelta(minutes=10),'event_rank_available':0})

def test_event_lexicographic_order():
    midnight=pd.Timestamp('2025-02-01')
    assert EventTime(midnight,0)<EventTime(midnight,1)<EventTime(midnight,2)
    assert EventTime(midnight-pd.Timedelta(minutes=10),2)<EventTime(midnight,0)
    assert EventTime(midnight+pd.Timedelta(minutes=10),0)>EventTime(midnight,2)
    assert not EventTime(midnight,2)<EventTime(midnight,2)

@pytest.mark.parametrize('series',['Load','PV'])
def test_february_first_full_january_only(actuals,series):
    decision=EventTime('2025-02-01',2)
    history=ActualStore(actuals).history_before(series,decision)
    assert len(history.frame)==31*144
    assert history.frame.source_day.nunique()==31
    assert history.frame.interval_end.max()==decision.timestamp
    assert EventTime(history.frame.available_at.max(),0)<decision
    assert (history.frame.source_day<pd.Timestamp('2025-02-01')).all()
    assert not (history.frame.source_day==pd.Timestamp('2025-02-01')).any()

@pytest.mark.parametrize('series',['Load','PV'])
def test_canonical_source_labels_and_daily_coverage(actuals,series):
    from seasonal import source_minutes
    frame=actuals.loc[actuals.series==series]
    for day,group in frame.groupby('source_day'):
        assert group.slot_id.tolist()==list(range(144))
        assert group.interval_start.iloc[0]==day
        assert group.interval_end.iloc[-1]==day+pd.Timedelta(days=1)
        assert (group.interval_start.iloc[1:].to_numpy()==group.interval_end.iloc[:-1].to_numpy()).all()
    # serialized datetime.time labels include seconds, test offsets before serialization separately
    from openpyxl import load_workbook
    workbook=load_workbook(Path(__file__).parents[1]/config()['input_file'],read_only=True)
    raw=list(next(workbook[config()['series_sheets'][series]].values))[1:]
    offsets=np.array([source_minutes(label) for label in raw])
    assert (offsets==np.arange(1,145)*10).all()
    workbook.close()

@pytest.mark.parametrize('model',['B0','B1','B2','B3','B4'])
def test_exact_source_selection_and_formula(synthetic,model):
    decision=EventTime('2025-02-01',2)
    history=ActualStore(synthetic).history_before('Load',decision)
    fitted=SeasonalForecaster(model).fit(history)
    day=decision.timestamp
    if model=='B0':
        expected_days=[day-pd.Timedelta(days=1)]
        expected=np.repeat(30143.0,144)
        assert fitted.used.slot_id.tolist()==[143]
    elif model=='B1': expected_days=[day-pd.Timedelta(days=1)]; expected=30000+np.arange(144)
    elif model=='B2': expected_days=[day-pd.Timedelta(days=7)]; expected=24000+np.arange(144)
    elif model=='B3': expected_days=[day-pd.Timedelta(days=lag) for lag in range(1,8)]; expected=27000+np.arange(144)
    else:
        expected_days=[day-pd.Timedelta(days=lag) for lag in [7,14,21,28]]
        expected=13500+np.arange(144)
    assert sorted(fitted.used.source_day.unique())==sorted(expected_days)
    np.testing.assert_allclose(fitted.predict(decision,144),expected)
    assert (fitted.used.source_day<day).all()
    for available_at,rank in fitted.used[['available_at','event_rank_available']].itertuples(index=False,name=None):
        assert EventTime(available_at,int(rank))<decision

@pytest.mark.parametrize('series',['Load','PV'])
@pytest.mark.parametrize('model',['B0','B1','B2','B3','B4'])
def test_future_poisoning_does_not_change_forecast(actuals,series,model):
    decision=EventTime('2025-02-01',2)
    original=SeasonalForecaster(model).fit(ActualStore(actuals).history_before(series,decision)).predict(decision,144)
    poisoned=actuals.copy()
    future=poisoned.source_day>=decision.timestamp
    poisoned.loc[future,'power_kw']=np.arange(future.sum())*999999.0-1e12
    candidate=SeasonalForecaster(model).fit(ActualStore(poisoned).history_before(series,decision)).predict(decision,144)
    np.testing.assert_array_equal(candidate,original)
    # Appending the whole future must give the same result as physically truncating it.
    truncated=actuals.loc[~future].copy()
    cutoff=SeasonalForecaster(model).fit(ActualStore(truncated).history_before(series,decision)).predict(decision,144)
    np.testing.assert_array_equal(cutoff,original)

@pytest.mark.parametrize('model',['B0','B1','B2','B3','B4'])
def test_full_year_input_rejected(actuals,model):
    history=CausalHistory(actuals.loc[actuals.series=='Load'].copy(),EventTime('2025-02-01',2))
    with pytest.raises(AssertionError,match='Future source day'):
        SeasonalForecaster(model).fit(history)

@pytest.mark.parametrize('alteration',['late_availability','same_rank','future_source_day','shuffle','missing_slot','duplicate_slot'])
def test_invalid_inputs_are_rejected(synthetic,alteration):
    history=ActualStore(synthetic).history_before('Load',EventTime('2025-02-01',2))
    frame=history.frame.copy()
    if alteration=='late_availability': frame.loc[frame.index[-1],'available_at']=pd.Timestamp('2025-02-01 00:10')
    elif alteration=='same_rank': frame.loc[frame.index[-1],'event_rank_available']=2
    elif alteration=='future_source_day': frame.loc[frame.index[-1],'source_day']=pd.Timestamp('2025-02-01')
    elif alteration=='shuffle': frame=frame.iloc[::-1].reset_index(drop=True)
    elif alteration=='missing_slot': frame=frame.iloc[:-1]
    else: frame=pd.concat([frame,frame.tail(1)],ignore_index=True)
    with pytest.raises(AssertionError): CausalHistory(frame,history.decision).validate()

@pytest.mark.parametrize('series',['Load','PV'])
@pytest.mark.parametrize('day',['2025-02-01','2025-03-01','2025-07-01','2025-12-31'])
def test_targets_are_future_and_144_slots(actuals,series,day):
    decision=EventTime(day,2)
    targets=ActualStore(actuals).reveal_targets(series,decision)
    assert len(targets)==144
    assert (targets.interval_start>=decision.timestamp).all()
    assert targets.interval_start.iloc[0]==decision.timestamp
    assert targets.interval_end.iloc[-1]==decision.timestamp+pd.Timedelta(days=1)

def test_no_scaler_or_full_year_statistics_in_predictor():
    settings=config()
    assert settings['scaler'] is None
    source=(Path(__file__).parents[1]/'seasonal.py').read_text(encoding='utf-8-sig')
    tree=ast.parse(source)
    forbidden={'StandardScaler','MinMaxScaler','train_test_split','std','var','shuffle','random_split'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
            assert name not in forbidden
    # B3/B4 are the only fitted averages and operate on selected historical rows.
    assert 'selected.groupby(\'slot_id\').power_kw.mean()' in source

@pytest.mark.parametrize('model',['B0','B1','B2','B3','B4'])
def test_save_load_exact_prediction(synthetic,tmp_path,model):
    decision=EventTime('2025-02-01',2)
    fitted=SeasonalForecaster(model).fit(ActualStore(synthetic).history_before('Load',decision))
    path=tmp_path/'model.json'; fitted.save(path)
    restored=SeasonalForecaster.load(path)
    np.testing.assert_array_equal(fitted.predict(decision,144),restored.predict(decision,144))
