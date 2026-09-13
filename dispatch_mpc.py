"""Linear battery dispatch helpers for CUMCM2026 questions 2--4.

All powers are kW and each array element is one ten minute interval.  Energy
flows returned by this module are kWh.  The model keeps the battery state in
the physical interval [1200,10800] and uses the stated 90% round-trip leg
efficiency (SOC update ``soc += .9*charge - discharge/.9``).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

DT_H = 1.0 / 6.0
ETA = 0.9
P_MAX_KW = 5000.0
E_MAX_KWH = 12000.0
SOC_MIN = 1200.0
SOC_MAX = 10800.0


def optimize_schedule(load_kw, pv_kw, price_cny, initial_soc,
                      terminal_soc=None, fixed_grid=None):
    """Solve one horizon linear dispatch problem.

    ``fixed_grid`` optionally fixes grid purchases at selected slots (NaN
    leaves a slot free).  The objective is energy purchase cost plus a tiny
    throughput regulariser.  A terminal SOC target defaults to the initial
    SOC, preventing day-to-day drift while preserving cross-day continuity.
    """
    load = np.asarray(load_kw, float); pv = np.asarray(pv_kw, float)
    price = np.asarray(price_cny, float); n = len(load)
    if not (len(pv) == len(price) == n): raise ValueError("horizon lengths differ")
    if terminal_soc is None: terminal_soc = float(initial_soc)
    # x=[grid, charge, discharge, soc(1..n)] ; SOC at t=0 is parameter.
    ng = 4*n; g=slice(0,n); c=slice(n,2*n); d=slice(2*n,3*n); s=slice(3*n,4*n)
    obj=np.zeros(ng); obj[g]=price; obj[c]=1e-6; obj[d]=1e-6
    Aeq=[]; beq=[]
    for t in range(n):
        row=np.zeros(ng); row[t]=1; row[n+t]=-1; row[2*n+t]=1
        Aeq.append(row); beq.append(load[t]-pv[t])
        row=np.zeros(ng); row[3*n+t]=1; row[n+t]=-ETA; row[2*n+t]=1/ETA
        if t: row[3*n+t-1]=-1; rhs=0
        else: rhs=float(initial_soc)
        Aeq.append(row); beq.append(rhs)
    # Explicit terminal SOC target (the last SOC variable).
    row=np.zeros(ng); row[3*n+n-1]=1; Aeq.append(row); beq.append(float(terminal_soc))
    bounds=[(0,None)]*ng
    for t in range(n): bounds[n+t]=(0,P_MAX_KW*DT_H); bounds[2*n+t]=(0,P_MAX_KW*DT_H); bounds[3*n+t]=(SOC_MIN,SOC_MAX)
    if fixed_grid is not None:
        fg=np.asarray(fixed_grid,float)
        for t,v in enumerate(fg):
            if np.isfinite(v): bounds[t]=(max(0,float(v)),max(0,float(v)))
    res=linprog(obj,A_eq=np.asarray(Aeq),b_eq=np.asarray(beq),bounds=bounds,method="highs")
    if not res.success: raise RuntimeError(f"dispatch LP failed: {res.message}")
    x=res.x
    return {"grid_kwh":x[g],"charge_kwh":x[c],"discharge_kwh":x[d],"soc_kwh":x[s],"objective_cny":float(res.fun)}


def execute_plan(load_kw, pv_kw, price_cny, plan_grid_kwh, charge_kwh,
                 discharge_kwh, initial_soc, emergency_multiplier=5.0):
    """Execute a day-ahead battery plan against realised load/PV.

    Planned grid is delivered first; any positive physical deficit is bought
    as emergency energy at ``emergency_multiplier * price``.  Battery flows
    are clipped to preserve SOC and are never allowed to create energy.
    """
    load=np.asarray(load_kw,float); pv=np.asarray(pv_kw,float); price=np.asarray(price_cny,float)
    gp=np.asarray(plan_grid_kwh,float); c=np.asarray(charge_kwh,float); d=np.asarray(discharge_kwh,float)
    n=len(load); soc=np.empty(n+1); soc[0]=initial_soc; emergency=np.zeros(n); grid_actual=np.zeros(n)
    c_exec=np.zeros(n); d_exec=np.zeros(n)
    for t in range(n):
        # enforce SOC bounds by clipping battery actions
        cmax=min(c[t], (SOC_MAX-soc[t])/ETA); dmax=min(d[t], (soc[t]-SOC_MIN)*ETA)
        c_exec[t]=max(0,cmax); d_exec[t]=max(0,dmax)
        soc[t+1]=soc[t]+ETA*c_exec[t]-d_exec[t]/ETA
        net=load[t]+c_exec[t]-pv[t]-d_exec[t]
        grid_actual[t]=max(0,gp[t]); emergency[t]=max(0,net-grid_actual[t])
    return {"grid_actual_kwh":grid_actual,"emergency_kwh":emergency,
            "charge_kwh":c_exec,"discharge_kwh":d_exec,"soc_kwh":soc[1:],
            "planned_cost_cny":float(np.sum(gp*price)),
            "emergency_cost_cny":float(np.sum(emergency*price*emergency_multiplier)),
            "total_cost_cny":float(np.sum(gp*price)+np.sum(emergency*price*emergency_multiplier))}


def validate_trace(trace, initial_soc, tol=1e-6):
    """Return a compact constraint audit dictionary for an execution trace."""
    soc=np.r_[float(initial_soc), np.asarray(trace["soc_kwh"],float)]
    c=np.asarray(trace["charge_kwh"],float); d=np.asarray(trace["discharge_kwh"],float)
    return {
        "soc_min_kwh":float(np.min(soc)), "soc_max_kwh":float(np.max(soc)),
        "soc_bounds_ok":bool(np.min(soc)>=SOC_MIN-tol and np.max(soc)<=SOC_MAX+tol),
        "charge_power_ok":bool(np.max(c)<=P_MAX_KW*DT_H+tol),
        "discharge_power_ok":bool(np.max(d)<=P_MAX_KW*DT_H+tol),
        "nonnegative_flows_ok":bool(np.min(np.r_[c,d,np.asarray(trace.get("emergency_kwh",[]),float)])>=-tol),
        "terminal_soc_kwh":float(soc[-1]),
    }


def settlement_cost(plan, revised, price, emergency_kwh=None, emergency_multiplier=5.0):
    """Question-3 settlement: downward deviation refunded at 0.5 p,
    upward deviation charged at 1.5 p; emergency is charged at 5 p."""
    plan=np.asarray(plan,float); revised=np.asarray(revised,float); p=np.asarray(price,float)
    down=np.maximum(plan-revised,0); up=np.maximum(revised-plan,0)
    common=np.minimum(plan,revised)
    c=float(np.sum(common*p + down*0.5*p + up*1.5*p))
    if emergency_kwh is not None: c += float(np.sum(np.asarray(emergency_kwh)*p*emergency_multiplier))
    return c


def revise_remaining_schedule(load_forecast, pv_forecast, price_forecast,
                              current_soc, previous_grid, executed_until,
                              terminal_soc=None):
    """Re-optimise only slots ``executed_until:`` for Q3/Q4 MPC.

    ``executed_until`` is a zero-based count of already settled ten-minute
    intervals.  Earlier grid quantities remain exactly as delivered; the
    returned vector is a post-adjustment plan (full-day cumulative amounts,
    rather than deltas), matching the result3 template convention.
    """
    prev=np.asarray(previous_grid,float); n=len(prev)
    k=int(np.clip(executed_until,0,n))
    lf=np.asarray(load_forecast,float)[k:]; pf=np.asarray(pv_forecast,float)[k:]; pr=np.asarray(price_forecast,float)[k:]
    if terminal_soc is None: terminal_soc=float(current_soc)
    sol=optimize_schedule(lf,pf,pr,current_soc,terminal_soc=terminal_soc)
    out=prev.copy(); out[k:]=sol["grid_kwh"]
    return out, sol
