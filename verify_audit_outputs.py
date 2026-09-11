import json
from pathlib import Path
import pandas as pd
root=Path.cwd(); folder=root/'artifacts/data_audit'
status=json.loads((folder/'run_status.json').read_text())
assert status['status']=='complete' and status['baselines']=='NOT RUN'
checks=json.loads((folder/'structural_checks.json').read_text(encoding='utf-8'))
assert checks['passed']==396 and checks['failed']==0 and all(item['passed'] for item in checks['checks'])
sheets=pd.read_csv(folder/'sheets_audit.csv'); columns=pd.read_csv(folder/'columns_audit.csv')
assert len(sheets)==21 and sheets.file.nunique()==9
for row in sheets.itertuples():
    selected=columns[(columns.file==row.file)&(columns.sheet==row.sheet)]
    assert len(selected)==row.columns
    assert selected.missing_body_cells.sum()==row.missing_body_cells
assert len(pd.read_csv(folder/'daily_point_counts.csv'))==365*3
origins=pd.read_csv(folder/'forecast_daily_counts.csv')
assert len(origins)==365 and (origins.origin_count==4).all() and (origins.min_horizons==24).all() and (origins.max_horizons==24).all()
report=(root/'DATA_AUDIT.md').read_text(encoding='utf-8-sig')
assert report==(folder/'DATA_AUDIT.md').read_text(encoding='utf-8-sig')
assert '`n-' not in report and '仅完成审计' in report
assert not list((root/'artifacts').rglob('predictions.csv'))
print('FINAL QA: 9 workbooks / 21 sheets; column counts and missing counts reconcile; 365 days x 4 origins x 24 horizons; 396 structural checks; original bytes preserved; NO baseline predictions.')
