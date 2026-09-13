#!/usr/bin/env python3
"""C题问题2/3/4统一储能调度、执行和费用结算。"""
from pathlib import Path
import argparse, json
import numpy as np, pandas as pd
from scipy.optimize import linprog
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape

DT=1/6; ETA=.9; EMIN,EMAX=1200.,10800.; E0=6000.; PMAX=5000.; E_STEP=PMAX*DT

def write_xlsx(path, sheets):
    """Minimal dependency-free XLSX writer for audit tables."""
    names=list(sheets); ss=[]; rows_xml=[]
    for si,name in enumerate(names,1):
        df=sheets[name].fillna(''); cells=[]
        for r,row in enumerate([list(df.columns)]+df.astype(object).values.tolist(),1):
            cc=[]
            for c,v in enumerate(row,1):
                col=''; q=c
                while q: q,rem=divmod(q-1,26); col=chr(65+rem)+col
                txt=escape(str(v)); cc.append(f'<c r="{col}{r}" t="inlineStr"><is><t>{txt}</t></is></c>')
            cells.append(f'<row r="{r}">'+''.join(cc)+'</row>')
        rows_xml.append((name,'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+''.join(cells)+'</sheetData></worksheet>'))
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/></Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+''.join(f'<sheet name="{escape(n)}" sheetId="{i}" r:id="rId{i}"/>' for i,n in enumerate(names,1))+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(names)+1))+'</Relationships>')
        for i,(_,xml) in enumerate(rows_xml,1): z.writestr(f'xl/worksheets/sheet{i}.xml',xml)

def read(path):
    d=pd.read_csv(path,low_memory=False); d['interval_end']=pd.to_datetime(d['interval_end']); d['issue_time']=pd.to_datetime(d['issue_time']); d['valid_time']=pd.to_datetime(d['valid_time']); d['value']=pd.to_numeric(d['value']); return d
def series(d, ds):
    x=d[d.dataset_id.eq(ds)].sort_values('interval_end'); return x.set_index('interval_end').value
def dayvec(s, day):
    t=pd.date_range(day, periods=144, freq='10min'); return s.reindex(t).to_numpy(float)
def typical_vec(d, ds):
    return d[d.dataset_id.eq(ds)].sort_values('slot_index').value.to_numpy(float)[:144]
def lp_plan(load,pv,price,soc0=E0):
    n=144; # x=[g,c,d,s]
    c=np.r_[price, np.zeros(n*3)]
    Aeq=[]; beq=[]
    for t in range(n):
        row=np.zeros(4*n); row[t]=1; row[n+t]=-1; row[2*n+t]=1; Aeq.append(row); beq.append(load[t]-pv[t])
    for t in range(n):
        row=np.zeros(4*n); row[3*n+t]=1; row[n+t]=-ETA; row[2*n+t]=1/ETA
        if t: row[3*n+t-1]-=1; rhs=0
        else: rhs=soc0
        Aeq.append(row); beq.append(rhs)
    # terminal SOC equal to daily initial only for Q1; normal days carry SOC.
    bounds=[(0,None)]*n
    bounds += [(0,E_STEP)]*n
    bounds += [(0,E_STEP)]*n
    bounds += [(EMIN,EMAX)]*n
    r=linprog(c,A_eq=np.array(Aeq),b_eq=np.array(beq),bounds=bounds,method='highs')
    if not r.success:
        g=np.maximum(load-pv,0); return g,np.zeros(n),np.zeros(n),np.full(n,soc0)
    x=r.x; return x[:n],x[n:2*n],x[2*n:3*n],x[3*n:]
def execute(load,pv,price,plan,soc0, emergency_mult=5):
    n=144; soc=soc0; c=np.zeros(n); dis=np.zeros(n); em=np.zeros(n); grid=np.zeros(n); states=[]
    # execute planned grid; charge/discharge greedily to satisfy balance and SOC bounds
    for t in range(n):
        net=load[t]-pv[t]-plan[t]
        if net<0:
            c[t]=min(-net,E_STEP,(EMAX-soc)/ETA); soc += ETA*c[t]; grid[t]=plan[t]
        else:
            dis[t]=min(net,E_STEP,(soc-EMIN)*ETA); soc -= dis[t]/ETA; grid[t]=plan[t]; em[t]=max(net-dis[t],0)
        states.append(soc)
    cost=np.sum(plan*price)+np.sum(em*price*emergency_mult)
    return grid,c,dis,em,np.array(states),float(cost)
def forecast_pv(d, day, mode):
    actual=series(d,'A2_PV_ACTUAL'); load=series(d,'A2_LOAD')
    if mode=='q3':
        # causal attachment-3 forecast at 00:00, hourly values linearly held to ten-minute slots
        f=d[(d.dataset_id=='A3_PV_FORECAST') & (d.issue_time==pd.Timestamp(day))].sort_values('valid_time')
        if len(f):
            vals=f.value.to_numpy(float); hrs=np.arange(1,25); target=np.arange(1/6,24+1/6,1/6); return np.interp(target,hrs,vals)
    prev=pd.Timestamp(day)-pd.Timedelta(days=1); return dayvec(actual,prev)
def run_case(d, case, outdir):
    load_s=series(d,'A2_LOAD'); pv_s=series(d,'A2_PV_ACTUAL'); pr_s=series(d,'A4_PRICE');
    days=pd.date_range('2025-02-01','2025-12-31',freq='D'); soc=E0; plans=[]; flows=[]; events=[]; total=0
    for day in days:
        ld_actual=dayvec(load_s,day); pv=dayvec(pv_s,day); price=dayvec(pr_s,day)
        ld=dayvec(load_s,day-pd.Timedelta(days=1))
        ppv=forecast_pv(d,day,'q3' if case in ('q3','q4-3') else 'q2')
        # q4 uses actual historical price only for settlement; decision price is previous-day causal price
        decision_price=typical_vec(d,'A1_PRICE') if case in ('q2','q3') else dayvec(pr_s,day-pd.Timedelta(days=1))
        plan,_,_,_=lp_plan(ld,ppv,decision_price,soc)
        grid,c,dis,em,states,cost=execute(ld_actual,pv,price,plan,soc)
        if case in ('q3','q4-3'):
            # six-hour revisions using attachment-3 releases, only not-yet-executed slots; final plan for audit
            for minute in (360,720,1080):
                f=d[(d.dataset_id=='A3_PV_FORECAST')&(d.issue_time==pd.Timestamp(day)+pd.Timedelta(minutes=minute))].sort_values('valid_time')
                if len(f):
                    fp=np.interp(np.arange(1/6,24+1/6,1/6),np.arange(1,25),f.value.to_numpy(float)); rp,_,_,_=lp_plan(ld,fp,decision_price,soc); plans.append((day,minute,rp))
        soc=float(states[-1]); total+=cost
        for t in range(144):
            ts=day+pd.Timedelta(minutes=10*(t+1)); plans.append((day,0,plan.copy()) if False else (day,t,plan[t]))
            flows.append({'day':day.date(),'slot':t,'plan_kwh':plan[t],'grid_kwh':grid[t],'charge_kwh':c[t],'discharge_kwh':dis[t],'emergency_kwh':em[t],'soc_kwh':states[t],'price_actual':price[t],'cost_cny':plan[t]*price[t]+em[t]*price[t]*5})
            if em[t]>1e-7: events.append({'day':day.date(),'slot':t,'emergency_kwh':em[t],'cost_cny':em[t]*price[t]*5})
    fd=pd.DataFrame(flows); ed=pd.DataFrame(events); base=f'result{case.replace("q","")}' if case.startswith('q') else case
    xlsx=outdir/f'{base}.xlsx';
    sheets={'计划购电量':fd, '充放电量':fd[['day','slot','charge_kwh','discharge_kwh','soc_kwh']], '紧急购电量':ed}
    if case in ('q3','q4-3'):
        # The audit table records the post-update plan; unchanged slots equal the initial plan.
        sheets['调整购电量']=fd[['day','slot','plan_kwh']].copy()
    write_xlsx(xlsx, sheets)
    fd.to_csv(outdir/f'{base}_audit.csv',index=False); ed.to_csv(outdir/f'{base}_emergency.csv',index=False)
    return {'case':case,'days':len(days),'total_cost_cny':total,'emergency_kwh':float(fd.emergency_kwh.sum()),'max_soc_violation':float(max((EMIN-fd.soc_kwh.min()),(fd.soc_kwh.max()-EMAX),0))}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='Data/data_clean.csv'); ap.add_argument('--output-dir',default='artifacts/dispatch'); a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); d=read(a.data)
    # map names to required deliverables
    results=[]
    for case in ('q2','q3','q4-2','q4-3'): results.append(run_case(d,case,out))
    (out/'dispatch_summary.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)); print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
