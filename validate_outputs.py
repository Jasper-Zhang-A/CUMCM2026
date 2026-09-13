import json
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parent; out=root/'artifacts/dispatch'
summary=json.loads((out/'dispatch_summary.json').read_text())
assert {x['case'] for x in summary}=={'q2','q3','q4-2','q4-3'}
for x in summary: assert x['max_soc_violation']<=1e-6
for p in out.glob('*_audit.csv'):
    d=pd.read_csv(p)
    assert (d[['plan_kwh','grid_kwh','charge_kwh','discharge_kwh','emergency_kwh']]>=-1e-8).all().all()
    assert d.soc_kwh.between(1200-1e-6,10800+1e-6).all()
    assert (d.cost_cny-(d.plan_kwh*d.price_actual+d.emergency_kwh*d.price_actual*5)).abs().max()<1e-4
print('validated no-negative flows, SOC bounds, emergency-only deficit billing, and four-case completion')
