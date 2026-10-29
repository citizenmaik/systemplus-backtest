import sys; sys.path.insert(0, "/home/claude/systemplus-backtest/backtest")
import numpy as np, pandas as pd
import v42_report as R, simulate as S
from signals import build
from growth import asof_features

# Modi: A = aktueller Code | B = SMA20 erst ab 1R (schriftl. M2) | C = SMA20 erst nach TP1
# I2a = Entry unter SMA20: nur SMA10 (100 % raus), sonst A
# I2b = Entry unter SMA20: SMA20-Regel erst scharf, wenn Close einmal > SMA20, sonst A
def make(mode):
    def m2(bars, i0):
        o,h,l,c,s10,s20,d=(bars[k] for k in ("open","high","low","close","sma10","sma20","date"))
        entry, stop = c[i0], s10[i0]; risk = entry-stop
        below = c[i0] < s20[i0]
        st = dict(rem=1.0, r=0.0, p=0.0); r1=tp1=False; fills=[]; armed = not below
        def fill(i,frac,px,tag):
            f=min(frac,st["rem"])
            if f>0:
                st["rem"]-=f; st["r"]+=f*(px-entry)/risk; st["p"]+=f*100*(px-entry)/entry; fills.append((d[i],f,px,tag))
        last=i0
        for i in range(i0+1,len(c)):
            last=i
            if l[i]<=stop: fill(i,st["rem"],min(o[i],stop),"SL")
            else:
                if not r1 and h[i]>=entry+risk: r1=True
                if mode=="I2a" and below:
                    if c[i]<s10[i]: fill(i,st["rem"],c[i],"Fail" if not r1 else "Exit10")
                elif not r1 and c[i]<s10[i]: fill(i,st["rem"],c[i],"Fail")
                else:
                    if not tp1 and c[i]<s10[i]: fill(i,0.4,c[i],"TP1"); tp1=True
                    gate = {"A":True,"I1":True,"I2a":True,"B":r1,"C":tp1,"I2b":armed}[mode]
                    if gate and c[i]<s20[i]: fill(i,st["rem"],c[i],"TP2")
                if c[i]>s20[i]: armed=True
            if st["rem"]<=1e-9: break
        op=st["rem"]>1e-9
        ro=st["rem"]*(c[last]-entry)/risk if op else 0; po=st["rem"]*100*(c[last]-entry)/entry if op else 0
        return dict(exit_date=d[last],R=st["r"]+ro,pct=st["p"]+po,open=op,r1=r1,bars_held=last-i0,fills=fills,
                    entry=entry,stop=stop,risk=risk,sl_dist=100*risk/entry, below20=below)
    return m2

x = build(start="2019-01-01", verbose=False)
m = (x.entry_sig & (x.raw_close > 5) & (x.avgvol30 > 1e6) & x.atr_pct.between(4.5, 10)
     & (x.mcap.isna() | (x.mcap > 300e6)) & (x.float_pct.isna() | (x.float_pct > 70))
     & x.off52.between(-30, -10) & (x.ticker != "NXH"))
sig0 = x[m]; closes = x.pivot_table(index="date", columns="ticker", values="close")
feat = asof_features(sig0[["date","ticker"]])
def met(c):
    r=c.pct_change().dropna(); yrs=(c.index[-1]-c.index[0]).days/365.25
    return 100*((c.iloc[-1]/c.iloc[0])**(1/yrs)-1), 100*(c/c.cummax()-1).min(), r.mean()/r.std()*np.sqrt(252)
rows=[]
for mode in ["A","I1","I2a","I2b","B","C"]:
    S.m2_exit = make("A" if mode=="I1" else mode)
    sig = sig0[sig0.close >= sig0.sma20] if mode=="I1" else sig0
    tr = S.simulate_trades(x, sig)
    tr = tr.join(x.loc[tr.index, ["atr_mult","atr_pct","off52","notbremse"]]).join(feat)
    tr = tr[tr.rev_accel > 0].copy(); tr["rw"]=np.where(tr.notbremse,0.5,1.0)
    c, taken, log, e = R.account(tr, closes)
    t = tr.loc[[k for k,_,_ in taken]]; b=t[t.below20]
    cg,dd,sh = met(c); c1=c[c.index<"2023-01-01"]; c2=c[c.index>="2023-01-01"]
    rows.append(dict(Variante=mode, CAGR=round(cg,1), MaxDD=round(dd,1), Sharpe=round(sh,2),
        CAGR_19_22=round(met(c1)[0],1), CAGR_23_26=round(met(c2)[0],1), Trades=len(t), avgR=round(t.R.mean(),2),
        Win=round(100*(t.R>0).mean()), Hold=round(t.bars_held.mean(),1), n_u20=len(b),
        avgR_u20=round(b.R.mean(),2) if len(b) else None, Hold_u20=round(b.bars_held.mean(),1) if len(b) else None,
        Exit1T_u20=round(100*(b.bars_held==1).mean()) if len(b) else None,
        avgR_ue20=round(t[~t.below20].R.mean(),2)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("/home/claude/systemplus-backtest/backtest/out/v42_sma20_varianten.csv", index=False)
