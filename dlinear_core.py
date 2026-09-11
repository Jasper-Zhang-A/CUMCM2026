"""DLinear adapter with event-causal training and checkpoint provenance."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, random, sys, time
import numpy as np
import pandas as pd
import torch
from seasonal import ROOT, EventTime, CausalHistory, ActualStore
sys.path.insert(0,str(ROOT/'vendor/tslib'))
from models.DLinear import Model

OUT=ROOT/'artifacts/dlinear'

def settings(): return json.loads((ROOT/'configs/dlinear.json').read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical():
    frame=pd.read_csv(ROOT/settings()['canonical_file'])
    for column in ['source_day','interval_start','interval_end','source_timestamp','available_at']:
        frame[column]=pd.to_datetime(frame[column],format='mixed')
    return frame

def allowed_history(actuals,series,cutoff,days=None):
    history=ActualStore(actuals).history_before(series,EventTime(cutoff,2))
    if days is not None:
        history.frame=history.frame.loc[history.frame.source_day>=pd.Timestamp(cutoff)-pd.Timedelta(days=days)].copy().reset_index(drop=True)
        if len(history.frame)!=days*144: raise ValueError('insufficient_available_history')
    return history.validate()

def frame_hash(frame):
    ordered=frame[['series','source_day','slot_id','interval_start','interval_end','power_kw','available_at','event_rank_available']]
    return hashlib.sha256(ordered.to_csv(index=False,float_format='%.17g').encode('utf-8')).hexdigest()

def window_starts(history,seq_len,pred_len=144,stride=1):
    history.validate()
    count=len(history.frame)-seq_len-pred_len+1
    if count<=0: raise ValueError('insufficient_points_for_context_plus_target')
    starts=np.arange(0,count,stride,dtype=np.int64)
    ends=starts+seq_len+pred_len-1
    frame=history.frame
    available=frame.available_at.iloc[ends]
    ranks=frame.event_rank_available.iloc[ends]
    assert ((available<history.decision.timestamp)|((available==history.decision.timestamp)&(ranks<history.decision.rank))).all()
    assert (frame.source_day.iloc[ends]<history.decision.timestamp).all()
    return starts

class CausalScaler:
    def fit(self,history):
        history.validate()
        powers=history.frame.power_kw.to_numpy(dtype=np.float64)
        self.mean=float(powers.mean()); self.std=float(powers.std(ddof=0))
        if self.std==0: self.std=1.0
        self.fit_event=EventTime(history.frame.available_at.max(),0)
        assert self.fit_event<history.decision
        self.data_hash=frame_hash(history.frame)
        return self
    def transform(self,powers): return (np.asarray(powers,dtype=np.float64)-self.mean)/self.std
    def inverse(self,powers): return np.asarray(powers,dtype=np.float64)*self.std+self.mean

def setup(seed,device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(settings()['torch_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if device=='cuda':
        assert torch.cuda.is_available(), 'CUDA unavailable; do not silently change registered backend'
        torch.cuda.manual_seed_all(seed)

def make_model(seq_len):
    cfg=settings()
    return Model(SimpleNamespace(task_name='long_term_forecast',seq_len=seq_len,pred_len=cfg['pred_len'],moving_avg=cfg['moving_avg'],enc_in=1),individual=False)

class DLinearForecaster:
    def __init__(self,seq_len,seed=2026,device=None):
        self.seq_len=seq_len; self.seed=seed; self.device=device or settings()['device']
    def initialize(self,history):
        setup(self.seed,self.device)
        self.scaler=CausalScaler().fit(history)
        self.model=make_model(self.seq_len).to(self.device)
        self.cutoff=history.decision
        self.training_hash=frame_hash(history.frame)
        return self
    def predict(self,history,horizon=144):
        history.validate()
        assert horizon==144 and self.cutoff<=history.decision
        if hasattr(self,'metadata') and self.metadata.get('selection_information_end'):
            assert EventTime(self.metadata['selection_information_end'],0)<history.decision, 'Selected checkpoint contains later validation information'
        context=history.frame.tail(self.seq_len)
        assert len(context)==self.seq_len
        assert self.scaler.fit_event<history.decision
        assert EventTime(context.available_at.max(),0)<history.decision
        assert (context.source_day<history.decision.timestamp).all()
        encoded=torch.tensor(self.scaler.transform(context.power_kw.to_numpy()),dtype=torch.float32,device=self.device).reshape(1,-1,1)
        self.model.eval()
        with torch.no_grad(): result=self.model(encoded,None,None,None).detach().cpu().numpy()[0,:,0]
        return self.scaler.inverse(result)
    def save(self,path,metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':{key:power.detach().cpu() for key,power in self.model.state_dict().items()}},path)
        self.metadata={**metadata,'checkpoint_sha256':sha(path),'checkpoint_path':str(path.relative_to(ROOT)),
            'seq_len':self.seq_len,'seed':self.seed,'training_data_hash':self.training_hash,
            'scaler_mean':self.scaler.mean,'scaler_std':self.scaler.std,'scaler_fit_end':self.scaler.fit_event.timestamp.isoformat(),
            'scaler_fit_rank':0,'scaler_data_hash':self.scaler.data_hash,
            'model_training_cutoff':self.cutoff.timestamp.isoformat(),'model_training_cutoff_rank':2,
            'config_sha256':sha(ROOT/'configs/dlinear.json'),'tslib_commit':json.loads((ROOT/'vendor/tslib/PROVENANCE.json').read_text())['commit']}
        self.metadata['core_code_sha256']=sha(Path(__file__))
        path.with_suffix('.json').write_text(json.dumps(self.metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        return self
    @classmethod
    def load(cls,path,device=None):
        path=Path(path);meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        assert sha(path)==meta['checkpoint_sha256']
        result=cls(meta['seq_len'],meta['seed'],device);result.metadata=meta
        result.model=make_model(result.seq_len).to(result.device)
        result.model.load_state_dict(torch.load(path,map_location=result.device,weights_only=True)['state_dict'])
        result.scaler=CausalScaler();result.scaler.mean=meta['scaler_mean'];result.scaler.std=meta['scaler_std']
        result.scaler.fit_event=EventTime(meta['scaler_fit_end'],meta['scaler_fit_rank']);result.scaler.data_hash=meta['scaler_data_hash']
        result.cutoff=EventTime(meta['model_training_cutoff'],2);result.training_hash=meta['training_data_hash']
        return result

def train(history,seq_len,epochs,seed,checkpoint,phase,validation=None,device=None):
    cfg=settings();checkpoint=Path(checkpoint)
    if checkpoint.exists() and checkpoint.with_suffix('.json').exists():
        cached=DLinearForecaster.load(checkpoint,device)
        assert cached.training_hash==frame_hash(history.frame) and cached.seq_len==seq_len and cached.seed==seed
        assert cached.metadata['config_sha256']==sha(ROOT/'configs/dlinear.json')
        assert cached.metadata['core_code_sha256']==sha(Path(__file__))
        assert cached.metadata['requested_epochs']==epochs and cached.metadata['phase']==phase
        return cached
    start_time=time.perf_counter()
    indices=window_starts(history,seq_len,cfg['pred_len'],cfg['window_stride'])
    learner=DLinearForecaster(seq_len,seed,device).initialize(history)
    powers=torch.tensor(learner.scaler.transform(history.frame.power_kw.to_numpy()),dtype=torch.float32,device=learner.device)
    # GPU unfold is a view; training batches stay chronological, no shuffled split or windows.
    windows=powers.unfold(0,seq_len+144,cfg['window_stride'])
    assert len(windows)==len(indices)
    optimizer=torch.optim.Adam(learner.model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    records=[];best_loss=float('inf');best_epoch=0;best_state=None;stale=0
    for epoch in range(1,epochs+1):
        learner.model.train();loss_sum=0.0;count=0
        for batch_start in range(0,len(windows),cfg['batch_size']):
            batch=windows[batch_start:batch_start+cfg['batch_size']]
            context=batch[:,:seq_len].unsqueeze(-1);target=batch[:,seq_len:].unsqueeze(-1)
            optimizer.zero_grad(set_to_none=True)
            forecast=learner.model(context,None,None,None)
            loss=torch.mean((forecast-target)**2)
            assert torch.isfinite(loss)
            loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(batch);count+=len(batch)
        record={'epoch':epoch,'training_mse_scaled':loss_sum/count}
        if validation is not None:
            errors=[]
            for valid_history,true_powers in validation:
                forecast=learner.predict(valid_history)
                errors.extend(np.abs(forecast-true_powers))
            validation_mae=float(np.mean(errors));record['validation_mae_kw']=validation_mae
            if validation_mae<best_loss-cfg['minimum_delta_kw']:
                best_loss=validation_mae;best_epoch=epoch;best_state={key:power.detach().clone() for key,power in learner.model.state_dict().items()};stale=0
            else: stale+=1
            if stale>=cfg['patience']:
                records.append(record);break
        records.append(record)
    if validation is not None:
        assert best_state is not None;learner.model.load_state_dict(best_state)
    else: best_epoch=epochs
    frame=history.frame
    meta={'phase':phase,'series':frame.series.iloc[0],'training_cutoff':history.decision.timestamp.isoformat(),
        'raw_points':len(frame),'available_days':frame.source_day.nunique(),'window_count':len(indices),
        'effective_day_cycles':frame.source_day.nunique(),'training_start':frame.interval_start.min().isoformat(),
        'training_end':frame.interval_end.max().isoformat(),'window_stride':cfg['window_stride'],
        'last_window_target_end':frame.interval_end.iloc[int(indices[-1])+seq_len+143].isoformat(),
        'last_window_target_available_rank':0,'requested_epochs':epochs,'epochs_executed':len(records),'best_epoch':best_epoch,
        'selection_validation_mae_kw':best_loss if validation is not None else None,
        'parameter_count':sum(parameter.numel() for parameter in learner.model.parameters()),
        'training_seconds':time.perf_counter()-start_time,'epoch_curve':records}
    meta['selection_information_end']=cfg['selection_final_cutoff'] if validation is not None else None
    learner.save(checkpoint,meta)
    print(f'{phase} {meta["series"]} L={seq_len} days={meta["available_days"]} windows={len(indices)} epochs={len(records)} best={best_epoch} seconds={meta["training_seconds"]:.1f}',flush=True)
    return learner
