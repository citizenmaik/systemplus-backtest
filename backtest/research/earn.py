import sys
GRP = sys.argv[1] if len(sys.argv) > 1 else "stats"
exec(open('impr.py').read())
E = pd.read_csv("/home/claude/systemplus-backtest/data/earnings.csv", keep_default_na=False, na_values=[""])
E["d"] = pd.to_datetime(E["date"])
cal = closes.index
# Reaktionstag: vor 12 Uhr = selber Tag, sonst nächster Handelstag
E["hh"] = E["time"].str.slice(0, 2).astype(int)
def react(d, hh):
    i = cal.searchsorted(d)
    if i < len(cal) and cal[i] == d and hh >= 12: i += 1
    return i
E["ri"] = [react(d, h) for d, h in zip(E.d, E.hh)]
EV = {tk: np.sort(g.ri.unique()) for tk, g in E.groupby("ticker")}
pos_of = {d: i for i, d in enumerate(cal)}
def feat(r):
    ev = EV.get(r.ticker)
    if ev is None: return pd.Series(dict(to_next=np.nan, since_last=np.nan, inside=np.nan))
    i0 = pos_of[r.date]; ie = pos_of.get(pd.Timestamp(r.exit_date), len(cal)-1)
    nx = ev[ev > i0]; pv = ev[ev <= i0]
    return pd.Series(dict(to_next=(nx[0]-i0) if len(nx) else np.nan, since_last=(i0-pv[-1]) if len(pv) else np.nan,
                          inside=float(len(nx) > 0 and nx[0] <= ie)))
F = t.apply(feat, axis=1)
t2 = t.join(F)
cl = t2[~t2.open]
print("mit Earnings-Daten:", round(100*t2.to_next.notna().mean()), "%")
b = pd.cut(cl.to_next, [0,5,10,20,40,70,999], labels=["1-5","6-10","11-20","21-40","41-70",">70"])
print(cl.groupby(b, observed=True).R.agg(["size","mean","median", lambda s: 100*(s>0).mean()]).round(2))
print("Earnings in Haltedauer:", cl.groupby("inside").R.agg(["size","mean"]).round(2).to_dict("index"))
b2 = pd.cut(cl.since_last, [-1,5,10,20,40,70,999], labels=["0-5","6-10","11-20","21-40","41-70",">70"])
print(cl.groupby(b2, observed=True).R.agg(["size","mean", lambda s: 100*(s>0).mean()]).round(2))
for h in "AB":
    z = cl[cl.half==h]; print(h, "to_next<=5: %.2f (n=%d) | >5: %.2f" % (z[z.to_next<=5].R.mean(), (z.to_next<=5).sum(), z[z.to_next>5].R.mean()),
        "| inside: %.2f | not: %.2f" % (z[z.inside==1].R.mean(), z[z.inside==0].R.mean()))
for p, lab in [((cl.date<"2023"), "19-22"), ((cl.date>="2023"), "23-26")]:
    z = cl[p]; print(lab, "to_next<=5: %.2f | >5: %.2f | inside %.2f | not %.2f" % (z[z.to_next<=5].R.mean(), z[z.to_next>5].R.mean(), z[z.inside==1].R.mean(), z[z.inside==0].R.mean()))

# Verkauf am Close vor dem Reaktionstag
def sell_before(r):
    if r.inside != 1: return r
    ev = pos_of[r.date] + int(r.to_next); dprev = cal[ev-1]
    if dprev <= r.date: return r
    keep = [f for f in r.fills if pd.Timestamp(f[0]) < dprev]
    rem = 1 - sum(f[1] for f in keep); px = closes.at[dprev, r.ticker]
    fills = keep + ([(np.datetime64(dprev), rem, px, "Earn")] if rem > 1e-9 else [])
    R = sum(f*(p-r.entry)/r.risk for _, f, p, _ in fills); pct = sum(f*100*(p-r.entry)/r.entry for _, f, p, _ in fills)
    r = r.copy(); r["fills"] = fills; r["R"] = R; r["pct"] = pct; r["open"] = False; r["exit_date"] = dprev
    return r
tS = t2.apply(sell_before, axis=1)
c2 = tS[~tS.open]; print("Verkauf vor Earnings: Ø R %.3f vs Basis %.3f" % (c2.R.mean(), cl.R.mean()))

irx = pd.read_csv("/home/claude/systemplus-backtest/data/irx.csv", parse_dates=["date"]).set_index("date").irx
irx = irx.reindex(cal).ffill()
irx_ib = (irx - 0.5).clip(lower=0)   # IBKR: Benchmark minus 0,5 %
V = {"Basis": (t2, {}), "Zins auf Cash (T-Bill)": (t2, dict(irx=irx)), "Zins auf Cash (T-Bill - 0,5 %, IBKR)": (t2, dict(irx=irx_ib)),
     "kein Entry <= 5 Tage vor Earnings": (t2[~(t2.to_next <= 5)], {}),
     "kein Entry <= 10 Tage vor Earnings": (t2[~(t2.to_next <= 10)], {}),
     "Verkauf am Tag vor Earnings": (tS, {}),
     "Verkauf vor Earnings + Zins IBKR": (tS, dict(irx=irx_ib)),
     "kein Entry <= 5 T + Zins IBKR": (t2[~(t2.to_next <= 5)], dict(irx=irx_ib))}
rows = []
for name, (tt, kw) in V.items():
    res = {"Variante": name}; c, n = run(tt, **kw); res.update(st(c, n))
    for h in "AB":
        ch, nh = run(tt[tt.half == h], **kw); s2 = st(ch, nh); res[f"{h}_CAGR"] = s2["CAGR"]; res[f"{h}_Sharpe"] = s2["Sharpe"]
    rows.append(res); print(res, flush=True)
pd.DataFrame(rows).to_csv("earn.csv", index=False)
