"""Causal univariate seasonal forecasters. No scaling or access to future actuals."""
from dataclasses import dataclass
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent

def config():
    return json.loads((ROOT/'configs/baselines.json').read_text(encoding='utf-8-sig'))

@dataclass(frozen=True, order=True)
class EventTime:
    timestamp: pd.Timestamp
    rank: int
    def __post_init__(self):
        object.__setattr__(self, 'timestamp', pd.Timestamp(self.timestamp))
        assert self.rank in (0, 1, 2), 'Invalid event rank'
        assert not pd.isna(self.timestamp)

def source_minutes(source_label):
    import datetime
    if isinstance(source_label, datetime.time):
        assert source_label.second == 0 and source_label.microsecond == 0
        return source_label.hour*60 + source_label.minute
    if isinstance(source_label, str):
        match = re.fullmatch(r'(\d{1,2}):(\d{1,2})(\+1)?', source_label)
        assert match, f'Unsupported source label: {source_label!r}'
        hour, minute = int(match[1]), int(match[2])
        assert 0 <= minute < 60 and 0 <= hour <= 24
        assert hour < 24 or minute == 0
        return hour*60 + minute + (1440 if match[3] else 0)
    raise AssertionError(f'Unsupported source type {type(source_label)}')

def load_actuals(settings=None):
    settings = settings or config()
    workbook = load_workbook(ROOT/settings['input_file'], read_only=True, data_only=True)
    pieces = []
    for series, sheet_name in settings['series_sheets'].items():
        rows = list(workbook[sheet_name].values)
        labels = rows[0][1:]
        offsets = np.array([source_minutes(source_label) for source_label in labels])
        assert sorted(offsets.tolist()) == list(range(10,1441,10)), 'Missing/duplicated source labels'
        source_days = pd.DatetimeIndex([row[0] for row in rows[1:]])
        assert source_days.equals(pd.date_range(settings['first_source_day'], settings['last_source_day']))
        powers = np.asarray([row[1:] for row in rows[1:]], dtype=float)
        assert powers.shape == (len(source_days),settings['horizon'])
        assert np.isfinite(powers).all(), 'Missing/nonfinite actuals must be audited'
        source_day = source_days.repeat(settings['horizon'])
        end_offsets = np.tile(offsets,len(source_days))
        intervals_end = source_day + pd.to_timedelta(end_offsets,unit='min')
        frame = pd.DataFrame({'series':series,'source_day':source_day,
            'slot_id':end_offsets//settings['interval_minutes']-1,
            'interval_start':intervals_end-pd.Timedelta(minutes=settings['interval_minutes']),
            'interval_end':intervals_end,
            'source_label':np.tile([source_label.isoformat() if hasattr(source_label,'isoformat') else source_label for source_label in labels],len(source_days)),
            'source_timestamp':intervals_end,'power_kw':powers.ravel(),
            'available_at':intervals_end,'event_rank_available':settings['actual_available_rank']})
        frame = frame.sort_values(['source_day','slot_id']).reset_index(drop=True)
        assert not frame.duplicated(['source_day','slot_id']).any()
        pieces.append(frame)
    workbook.close()
    return pd.concat(pieces,ignore_index=True)

@dataclass
class CausalHistory:
    frame: pd.DataFrame
    decision: EventTime
    horizon: int = 144
    def validate(self):
        frame = self.frame
        assert not frame.empty
        assert frame.series.nunique() == 1, 'Univariate input only'
        assert self.decision.rank == 2
        assert self.decision.timestamp == self.decision.timestamp.normalize()
        assert (frame.source_day < self.decision.timestamp.normalize()).all(), 'Future source day in history'
        earlier = ((frame.available_at < self.decision.timestamp) |
                   ((frame.available_at == self.decision.timestamp) & (frame.event_rank_available < self.decision.rank)))
        assert earlier.all(), 'Input available_event must precede decision_event'
        assert (frame.available_at == frame.interval_end).all(), 'Actual availability must follow completed interval'
        assert (frame.event_rank_available == 0).all()
        assert (frame.interval_end-frame.interval_start == pd.Timedelta(minutes=10)).all()
        assert (frame.interval_start == frame.source_day + pd.to_timedelta(frame.slot_id*10,unit='min')).all()
        assert np.isfinite(frame.power_kw).all()
        assert not frame.duplicated(['source_day','slot_id']).any()
        assert frame.interval_start.is_monotonic_increasing, 'No shuffled history'
        groups=frame.groupby('source_day').slot_id
        assert (groups.size()==self.horizon).all(), 'Incomplete history day'
        assert (groups.min()==0).all() and (groups.max()==self.horizon-1).all()
        return self

class ActualStore:
    """History access and settlement access are separate operations."""
    def __init__(self, actuals):
        self._actuals=actuals.copy()
    def history_before(self, series, decision):
        frame=self._actuals
        event_before=(frame.available_at < decision.timestamp) | ((frame.available_at==decision.timestamp)&(frame.event_rank_available<decision.rank))
        mask=(frame.series==series)&(frame.source_day<decision.timestamp.normalize())&event_before
        return CausalHistory(frame.loc[mask].copy().reset_index(drop=True),decision).validate()
    def reveal_targets(self,series,decision):
        return self._actuals.loc[(self._actuals.series==series)&(self._actuals.source_day==decision.timestamp.normalize())].sort_values('slot_id').copy()

class Forecaster:
    def fit(self, history): raise NotImplementedError
    def predict(self, context, horizon): raise NotImplementedError
    def save(self, path): raise NotImplementedError
    @classmethod
    def load(cls, path): raise NotImplementedError

class SeasonalForecaster(Forecaster):
    MODELS=('B0','B1','B2','B3','B4')
    def __init__(self, model):
        assert model in self.MODELS
        self.model=model
    def fit(self, history):
        history.validate()
        source=history.frame
        day=history.decision.timestamp.normalize()
        if self.model=='B0': selected=source.tail(1)
        elif self.model=='B1': selected=source.loc[source.source_day==day-pd.Timedelta(days=1)]
        elif self.model=='B2': selected=source.loc[source.source_day==day-pd.Timedelta(days=7)]
        elif self.model=='B3': selected=source.loc[source.source_day>=day-pd.Timedelta(days=7)]
        else: selected=source.loc[source.source_day.dt.dayofweek==day.dayofweek]
        assert not selected.empty
        if self.model=='B1': assert len(selected)==history.horizon
        if self.model=='B2': assert len(selected)==history.horizon
        if self.model=='B3': assert len(selected)==7*history.horizon
        self.decision=history.decision
        self.used=selected.copy()
        if self.model=='B0': self.prediction=np.repeat(float(selected.power_kw.iloc[-1]),history.horizon)
        else:
            self.prediction=selected.groupby('slot_id').power_kw.mean().reindex(range(history.horizon)).to_numpy()
        assert np.isfinite(self.prediction).all()
        return self
    def predict(self, context, horizon):
        assert isinstance(context,EventTime) and context==self.decision
        assert horizon==len(self.prediction)
        return self.prediction.copy()
    def save(self,path):
        payload={'model':self.model,'decision_time':self.decision.timestamp.isoformat(),'rank':self.decision.rank,'prediction':self.prediction.tolist()}
        Path(path).write_text(json.dumps(payload),encoding='utf-8')
    @classmethod
    def load(cls,path):
        payload=json.loads(Path(path).read_text(encoding='utf-8'))
        instance=cls(payload['model']); instance.decision=EventTime(payload['decision_time'],payload['rank'])
        instance.prediction=np.asarray(payload['prediction'],dtype=float)
        return instance
