import sys, numpy as np, pandas as pd
exec(open('v4_half.py').read().split('def run(')[0])   # t, closes, sel
x = pd.read_pickle(f"{SP}/x.pkl")
x = x[x.ticker.isin(t.ticker.unique())].copy()
x["sma50"] = x.groupby("ticker")["close"].transform(lambda v: v.rolling(50).mean())
SECT = x.drop_duplicates("ticker").set_index("ticker")["sector_etf"].fillna("?")
pos = {tk: g.index.to_numpy() for tk, g in x.groupby("ticker", sort=False)}
B = {}
for tk, ix in pos.items():
    sub = x.loc[ix]
    B[tk] = ({k: sub[k].to_numpy() for k in ("open","high","low","close","sma10","sma20","sma50","date")}, {v: n for n, v in enumerate(ix)})

def m2(b, i0, k_long=None, slow="sma50"):
    o,h,l,c,s10,s20,d = (b[k] for k in ("open","high","low","close","sma10","sma20","date"))
    sl = b[slow]
    entry, stop = c[i0], s10[i0]; risk = entry - stop
    rem, rs, ps, r1, tp1, fills, mx = 1.0, 0.0, 0.0, False, False, [], entry
    def fill(i, f, px, tag):
        nonlocal rem, rs, ps
        f = min(f, rem)
        if f > 0:
            rem -= f; rs += f*(px-entry)/risk; ps += f*100*(px-entry)/entry; fills.append((d[i], f, px, tag))
    last = i0
    for i in range(i0+1, len(c)):
        last = i; mx = max(mx, h[i])
        if l[i] <= stop: fill(i, rem, min(o[i], stop), "SL")
        else:
            if not r1 and h[i] >= entry + risk: r1 = True
            if not r1 and c[i] < s10[i]: fill(i, rem, c[i], "Fail")
            else:
                if not tp1 and c[i] < s10[i]: fill(i, .4, c[i], "TP1"); tp1 = True
                longrun = k_long is not None and mx >= entry + k_long*risk
                if longrun:
                    if c[i] < sl[i]: fill(i, rem, c[i], "TP2L")
                elif c[i] < s20[i]: fill(i, rem, c[i], "TP2")
        if rem <= 1e-9: break
    op = rem > 1e-9
    if op: rs += rem*(c[last]-entry)/risk; ps += rem*100*(c[last]-entry)/entry
    return dict(R=rs, pct=ps, open=op, fills=fills, entry=entry, stop=stop, risk=risk, exit_date=d[last])

def resim(k_long, slow="sma50"):
    rows = {}
    for k, r in t.iterrows():
        b, loc = B[r.ticker]; rows[k] = m2(b, loc[k], k_long, slow)
    v = pd.DataFrame.from_dict(rows, orient="index")
    return t[["date","ticker","atr_mult","rw","half"]].join(v)

def run(tr, risk_pct=1, cap_pct=15, per_day=3, start_eq=10000, max_pos=None, max_sec=None, dd_brake=None, irx=None):
    tr = tr.sort_values(["date","rw","atr_mult"], ascending=[True,False,True])
    by_day = {d: g for d, g in tr.groupby("date")}
    cash = eq = start_eq; peak = start_eq; held, curve, n_tr = {}, [], 0
    prev = None
    for d in closes.index:
        if irx is not None and prev is not None:
            days = (d - prev).days
            cash *= (1 + max(irx.get(prev, 0), 0)/100 * days/365)
        prev = d
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
        peak = max(peak, eq)
        brake = dd_brake is not None and eq/peak - 1 <= dd_brake/100
        n = 0
        for k, r in (by_day[d].iterrows() if d in by_day else []):
            if n >= per_day: break
            if max_pos and len(held) >= max_pos: break
            if any(p["t"] == r.ticker for p in held.values()): continue
            if max_sec and sum(p["s"] == SECT.get(r.ticker) for p in held.values()) >= max_sec: continue
            rw = r.rw * (0.5 if brake else 1)
            sh = np.floor(min(eq*risk_pct*rw/100/r.risk, eq*cap_pct/100/r.entry)*10)/10
            sh = min(sh, np.floor(cash/r.entry*10)/10)
            if sh <= 0: continue
            cash -= sh*r.entry
            held[k] = dict(t=r.ticker, s=SECT.get(r.ticker), sh=sh, sh0=sh, fills=list(r.fills), last=r.entry); n += 1; n_tr += 1
        curve.append((d, cash + sum(p["sh"]*p["last"] for p in held.values())))
    return pd.DataFrame(curve, columns=["date","eq"]).set_index("date")["eq"], n_tr

def st(c, n):
    yrs = (c.index[-1]-c.index[0]).days/365.25; cg = (c.iloc[-1]/c.iloc[0])**(1/yrs)-1
    r = c.pct_change().dropna(); dd = (c/c.cummax()-1).min()
    ye = c.resample("YE").last(); yr = (ye/pd.concat([pd.Series([c.iloc[0]]), ye.iloc[:-1]]).to_numpy()-1)*100
    a = c[c.index < "2023"]; b2 = c[c.index >= "2023"]
    cg_a = (a.iloc[-1]/a.iloc[0])**(365.25/(a.index[-1]-a.index[0]).days)-1
    cg_b = (b2.iloc[-1]/b2.iloc[0])**(365.25/(b2.index[-1]-b2.index[0]).days)-1
    return dict(CAGR=round(100*cg,1), MaxDD=round(100*dd,1), Sharpe=round(r.mean()/r.std()*np.sqrt(252),2),
                Sortino=round(r.mean()/r[r<0].std()*np.sqrt(252),2), Calmar=round(cg/-dd,2), Schlechtestes_Jahr=round(yr.min(),1),
                Trades_Jahr=round(n/yrs), CAGR_19_22=round(100*cg_a,1), CAGR_23_26=round(100*cg_b,1))
