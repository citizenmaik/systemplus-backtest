import numpy as np, pandas as pd
exec(open('v4_half.py').read().split('c20 = run')[0])
rng = np.random.default_rng(7); tk = t.ticker.unique(); out = []
for i in range(12):
    a = set(rng.choice(tk, len(tk)//2, replace=False))
    for m in (t.ticker.isin(a), ~t.ticker.isin(a)):
        c = run(t[m], 1, 15, start_eq=10000)
        yrs = (c.index[-1]-c.index[0]).days/365.25; r = c.pct_change().dropna()
        out.append(((c.iloc[-1]/10000)**(1/yrs)*100-100, 100*(c/c.cummax()-1).min(), r.mean()/r.std()*np.sqrt(252)))
    print(i, [tuple(round(v,2) for v in o) for o in out[-2:]], flush=True)
o = np.array(out)
print("CAGR P10/Median/P90", np.percentile(o[:,0],[10,50,90]).round(1), "min/max", o[:,0].min().round(1), o[:,0].max().round(1))
print("DD Median", np.median(o[:,1]).round(1), "Sharpe P10/Median/P90", np.percentile(o[:,2],[10,50,90]).round(2))
