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
t2 = t.join(F); cl = t2[~t2.open]
for lab, m in [("0-5 Tage nach Earnings", cl.since_last <= 5), ("0-2", cl.since_last <= 2), ("3-5", cl.since_last.between(3,5)), ("Rest", cl.since_last > 5)]:
    z = cl[m]; print(lab, "n", len(z), "ØR %.2f" % z.R.mean(), "A %.2f B %.2f" % (z[z.half=="A"].R.mean(), z[z.half=="B"].R.mean()),
        "19-22 %.2f 23-26 %.2f" % (z[z.date<"2023"].R.mean(), z[z.date>="2023"].R.mean()), "Win %.0f" % (100*(z.R>0).mean()))
irx = pd.read_csv("/home/claude/systemplus-backtest/data/irx.csv", parse_dates=["date"]).set_index("date").irx.reindex(cal).ffill()
irx_ib = (irx - 0.5).clip(lower=0)
for name, tt in [("kein Entry 0-5 T nach Earnings + Zins", t2[~(t2.since_last <= 5)])]:
    res = {"Variante": name}; c, n = run(tt, irx=irx_ib); res.update(st(c, n))
    for h in "AB":
        ch, nh = run(tt[tt.half == h], irx=irx_ib); s2 = st(ch, nh); res[f"{h}_CAGR"] = s2["CAGR"]; res[f"{h}_Sharpe"] = s2["Sharpe"]
    print(res)
