"""Independent delivery checks after the complete DLinear pipeline."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/dlinear'

def main():
    assert json.loads((OUT/'run_status.json').read_text())['status']=='complete'
    preserved=json.loads((ROOT/'artifacts/baselines/output_manifest.json').read_text())
    for item in preserved:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
    pred=pd.read_csv(OUT/'predictions_seed2026.csv')
    scores=pd.read_csv(OUT/'metrics_dlinear_global.csv').set_index('series')
    for series, group in pred.groupby('series'):
        error=group.pred_power_kw.to_numpy()-group.target_power_kw.to_numpy()
        actual=group.target_power_kw.to_numpy()
        expected=[np.abs(error).mean(),np.sqrt(np.square(error).mean()),np.abs(error).mean()/np.abs(actual).mean()]
        np.testing.assert_allclose(scores.loc[series,['mae_kw','rmse_kw','nmae']].to_numpy(dtype=float),expected,rtol=1e-12)
        if series=='PV':
            np.testing.assert_allclose(scores.loc[series,'daylight_mae_kw'],np.abs(error[actual>0]).mean(),rtol=1e-12)
    low=pd.read_csv(OUT/'predictions_low_data.csv')
    for _,group in low.loc[low.model.isin(['B0','B1','B2','B3'])].groupby(['series','model','target_group','target_slot']):
        assert group.pred_power_kw.nunique()==1, 'Fixed recent-day seasonal methods must not vary with budgets >=7'
    trigger=json.loads((OUT/'multiseed_trigger.json').read_text())['series_improved_seed2026']
    seeds=pd.read_csv(OUT/'metrics_by_seed.csv')
    for series in ['Load','PV']:
        assert set(seeds.loc[seeds.series==series,'seed'])==({2026,2027,2028} if series in trigger else {2026})
    assert len(list((OUT/'figures').glob('*.png')))==7
    summary={'status':'passed','preserved_baseline_files':len(preserved),'seed2026_prediction_rows':len(pred),'low_data_prediction_rows':len(low),'metrics_recomputed_independently':True,'seed_trigger_verified':True,'fixed_recent_day_baselines_budget_invariant':True}
    (OUT/'delivery_qa.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    paths=sorted([path for path in OUT.rglob('*') if path.is_file() and 'test_tmp' not in path.parts and path.name!='output_manifest.json'])
    paths += [ROOT/name for name in ['MODEL_SELECTION.md','MODEL_COMPARISON.md','DLINEAR_PROTOCOL.md','dlinear_core.py','run_dlinear.py','report_dlinear.py','dlinear_plots.py','verify_dlinear_outputs.py','configs/dlinear.json']]
    manifest=[{'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    (OUT/'output_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
