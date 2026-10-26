import numpy as np, pandas as pd
exec(open('v4_half.py').read().split('c20 = run')[0])   # t (V4.1 mit rw), run(), closes
x = pd.read_pickle(f"{SP}/x.pkl")
x = x[x.ticker.isin(t.ticker.unique())]
pos = {tk: g.index.to_numpy() for tk, g in x.groupby("ticker", sort=False)}
B = {}
for tk, ix in pos.items():
    sub = x.loc[ix]
    B[tk] = ({k: sub[k].to_numpy() for k in ("open","high","low","close","sma10","sma20","date","atr")}, {v: n for n, v in enumerate(ix)})

def m2(b, i0, stop):
    o,h,l,c,s10,s20,d = (b[k] for k in ("open","high","low","close","sma10","sma20","date"))
    entry = c[i0]; risk = entry - stop
    rem, rs, ps, r1, tp1, fills = 1.0, 0.0, 0.0, False, False, []
    def fill(i, f, px, tag):
        nonlocal rem, rs, ps
        f = min(f, rem)
        if f > 0:
            rem -= f; rs += f*(px-entry)/risk; ps += f*100*(px-entry)/entry; fills.append((d[i], f, px, tag))
    last = i0
    for i in range(i0+1, len(c)):
        last = i
        if l[i] <= stop: fill(i, rem, min(o[i], stop), "SL")
        else:
            if not r1 and h[i] >= entry + risk: r1 = True
            if not r1 and c[i] < s10[i]: fill(i, rem, c[i], "Fail")
            else:
                if not tp1 and c[i] < s10[i]: fill(i, .4, c[i], "TP1"); tp1 = True
                if c[i] < s20[i]: fill(i, rem, c[i], "TP2")
        if rem <= 1e-9: break
    op = rem > 1e-9
    if op: rs += rem*(c[last]-entry)/risk; ps += rem*100*(c[last]-entry)/entry
    return dict(R=rs, pct=ps, open=op, r1=r1, fills=fills, entry=entry, stop=stop, risk=risk,
                sl_dist=100*risk/entry, bars_held=last-i0)

def variant(fn):
    rows = {}
    for k, r in t.iterrows():
        b, loc = B[r.ticker]; i0 = loc[k]
        stop = fn(b, i0)
        if not stop < b["close"][i0]: continue
        rows[k] = m2(b, i0, stop)
    v = pd.DataFrame.from_dict(rows, orient="index")
    return t[["date","ticker","atr_mult","rw","half"]].join(v, how="inner")

S = {
 "SMA10 (aktuell)": lambda b,i: b["sma10"][i],
 "Entry - 1,0 ATR": lambda b,i: b["close"][i] - 1.0*b["atr"][i],
 "Entry - 1,5 ATR": lambda b,i: b["close"][i] - 1.5*b["atr"][i],
 "Entry - 2,0 ATR": lambda b,i: b["close"][i] - 2.0*b["atr"][i],
 "enger von SMA10 / -1 ATR": lambda b,i: max(b["sma10"][i], b["close"][i]-b["atr"][i]),
 "weiter von SMA10 / -1 ATR": lambda b,i: min(b["sma10"][i], b["close"][i]-b["atr"][i]),
 "Box-Unterkante": None,
}
xb = x[["box_bot"]]
S["Box-Unterkante"] = None
out = []
for name, fn in S.items():
    if fn is None: continue
    v = variant(fn)
    cl = v[~v.open]; w = cl.R > 0
    c = run(v.sort_values(["date","rw","atr_mult"], ascending=[True,False,True]), 1, 15, start_eq=10000)
    yrs = (c.index[-1]-c.index[0]).days/365.25
    dd = c/c.cummax()-1; cg = (c.iloc[-1]/10000)**(1/yrs)-1
    ye = c.resample("YE").last(); yr = (ye/pd.concat([pd.Series([10000.0]), ye.iloc[:-1]]).to_numpy()-1)*100
    out.append(dict(Stop=name, Trades=len(v), SL_Abstand=round(v.sl_dist.mean(),1), SL_Quote=round(100*cl.fills.map(lambda f: any(g=="SL" for *_,g in f)).mean()),
        Win=round(100*w.mean(),1), Avg_pct=round(cl.pct.mean(),2), Avg_R=round(cl.R.mean(),2), PF=round(cl.R[w].sum()/-cl.R[~w].sum(),2),
        R_A=round(cl[cl.half=="A"].R.mean(),2), R_B=round(cl[cl.half=="B"].R.mean(),2),
        R_19_22=round(cl[cl.date<"2023"].R.mean(),2), R_23_26=round(cl[cl.date>="2023"].R.mean(),2),
        Haltedauer=round(cl.bars_held.mean(),1), CAGR=round(100*cg,1), MaxDD=round(100*dd.min(),1), Calmar=round(cg/-dd.min(),2),
        Schlechtestes_Jahr=round(yr.min(),1)))
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv("slvar.csv", index=False)
