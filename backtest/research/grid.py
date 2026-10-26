import sys, numpy as np, pandas as pd
exec(open('v4_half.py').read().split('c20 = run')[0])
mode = MODE
if mode == "aus":
    tt = sel(w52=(7.5,30), nb=()).assign(rw=1.0).sort_values(["date","atr_mult"])
else:
    tt = t
rows=[]
for rp in (1.0, 1.25, 1.5):
    for cp in (10, 15, 20):
        c = run(tt, rp, cp)
        yrs=(c.index[-1]-c.index[0]).days/365.25
        dd=c/c.cummax()-1; best=cur=0
        for u in (dd<0): cur=cur+1 if u else 0; best=max(best,cur)
        ye=c.resample("YE").last(); yr=(ye/pd.concat([pd.Series([20000.0]),ye.iloc[:-1]]).to_numpy()-1)*100
        d=c.pct_change().dropna(); cagr=(c.iloc[-1]/20000)**(1/yrs)-1
        m=c.resample("ME").last().pct_change().dropna()
        rows.append(dict(Risiko=rp, Deckel=cp, CAGR=round(100*cagr,1), MaxDD=round(100*dd.min(),1),
            Calmar=round(cagr/-dd.min(),2), Sharpe=round(d.mean()/d.std()*np.sqrt(252),2),
            Schlechtestes_Jahr=round(yr.min(),1), Verlustmonate=round(100*(m<0).mean()), Schlechtester_Monat=round(100*m.min(),1),
            DD_Tage=best, Ende=round(c.iloc[-1])))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"grid_{mode}.csv", index=False)
