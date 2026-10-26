import sys, json, numpy as np, pandas as pd
sys.argv=['x','none']
exec(open('v4_var.py').read().split('if __name__')[0])
sys.path.insert(0,'/home/claude/systemplus-backtest/scripts')
from common import read_store
t1 = sel(w52=(7.5,30)); t2 = sel(w52=(7.5,30), nb=())
t2 = t2[~t2.index.isin(t1.index)]
t = pd.concat([t1.assign(rw=1.0), t2.assign(rw=0.5)]).sort_values(["date","rw","atr_mult"], ascending=[True,False,True])
def run(tr, risk_pct, cap_pct, per_day=3, start_eq=20000):
    by_day = {d: g for d, g in tr.groupby("date")}
    cash = eq = start_eq; held, curve = {}, []
    for d in closes.index:
        for k in list(held):
            p = held[k]
            while p["fills"] and p["fills"][0][0] == d:
                _, frac, px, _ = p["fills"].pop(0)
                q = min(p["sh0"]*frac, p["sh"]); cash += q*px; p["sh"] -= q
            if p["sh"] <= 1e-6: del held[k]
        row = closes.loc[d]
        for p in held.values():
            v = row.get(p["t"])
            if pd.notna(v): p["last"] = v
        eq = cash + sum(p["sh"]*p["last"] for p in held.values())
        n = 0
        for k, r in (by_day[d].iterrows() if d in by_day else []):
            if n >= per_day: break
            if any(p["t"] == r.ticker for p in held.values()): continue
            sh = np.floor(min(eq*risk_pct*r.rw/100/r.risk, eq*cap_pct/100/r.entry)*10)/10
            sh = min(sh, np.floor(cash/r.entry*10)/10)
            if sh <= 0: continue
            cash -= sh*r.entry
            held[k] = dict(t=r.ticker, sh=sh, sh0=sh, fills=list(r.fills), last=r.entry); n += 1
        curve.append((d, cash + sum(p["sh"]*p["last"] for p in held.values())))
    return pd.DataFrame(curve, columns=["date","eq"]).set_index("date")["eq"]
c20 = run(t, 1, 20); c10 = run(t, 1, 10)
yrs=(c20.index[-1]-c20.index[0]).days/365.25
for n,c in [("20",c20),("10",c10)]:
    dd=c/c.cummax()-1
    print(n, round(c.iloc[-1]), '%.1f'%(100*((c.iloc[-1]/20000)**(1/yrs)-1)), '%.1f'%(100*dd.min()), (c.resample('YE').last().pct_change()*100).round(1).tolist(), 'Jahr1 %.1f'%(100*(c.resample('YE').last().iloc[0]/20000-1)))
m = c20.resample('ME').last(); m.iloc[0]=c20.iloc[0]
json.dump([round(v) for v in m], open('half20.json','w'))
