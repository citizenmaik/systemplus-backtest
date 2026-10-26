import numpy as np, pandas as pd
exec(open('v4_half.py').read().split('c20 = run')[0])
cl = t[~t.open]
print("Trades", len(t), "geschlossen", len(cl), "Ø R", round(cl.R.mean(),3))
s = cl.R.sort_values(ascending=False)
for k in (1, 5, 10, 20):
    print(f"ohne Top {k}: Ø R {cl.R.drop(s.index[:k]).mean():.3f}")
print("Top 10:", [(t.loc[i,'ticker'], str(t.loc[i,'date'].date()), round(v,1), t.loc[i,'half']) for i, v in s.head(10).items()])
print("R-Summe A/B:", cl.groupby('half').R.agg(['size','mean','sum','median']).round(2).to_dict('index'))
for h in "AB":
    z = cl[cl.half == h].R.sort_values(ascending=False)
    print(h, "ohne Top 5: %.3f" % z.iloc[5:].mean(), "ohne Top 10: %.3f" % z.iloc[10:].mean())
# 2000 zufällige Teilungen nach Ticker
rng = np.random.default_rng(1)
tk = cl.ticker.unique(); res = []
for _ in range(2000):
    a = set(rng.choice(tk, len(tk)//2, replace=False))
    m = cl.ticker.isin(a)
    res.append((cl.R[m].mean(), cl.R[~m].mean()))
r = np.array(res).ravel()
print("Ø R einer zufälligen Hälfte: P5 %.2f  P25 %.2f  Median %.2f  P75 %.2f  P95 %.2f" % tuple(np.percentile(r, [5,25,50,75,95])))
ra = cl[cl.half=="A"].R.mean(); print("Hälfte A Ø R %.3f liegt bei Perzentil %.0f" % (ra, 100*(r < ra).mean()))
print("Anteil Hälften mit Ø R < 0:", round(100*(r<0).mean(),1), "%")
