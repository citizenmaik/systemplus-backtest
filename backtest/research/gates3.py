import sys, zlib
GRP = sys.argv[1]
exec(open('impr.py').read())
E = pd.read_pickle("v4_base_ext.pkl")
E = E[(E.date <= "2026-10-23") & (E.ticker != "NXH")].copy()
E["half"] = np.where(E.ticker.map(lambda s: zlib.crc32(s.encode()) % 2) == 0, "A", "B")
nbon = (E.q_ema.fillna(True) | (E.q_dd_val <= -10) | E.q_slope.fillna(True)).to_numpy()
YRS = (E.date.max() - pd.Timestamp("2019-01-02")).days / 365.25
REV = {"yoy_accel": lambda e: e.rev_accel > 0, "none": lambda e: pd.Series(True, index=e.index),
       "yoy_gt0": lambda e: e.rev_yoy > 0, "yoy_gt10": lambda e: e.rev_yoy > 10,
       "yoy_accel_gt0": lambda e: (e.rev_accel > 0) & (e.rev_yoy > 0),
       "yoy_accel_lt25": lambda e: (e.rev_accel > 0) & (e.rev_yoy < 25),
       "qoq_gt0": lambda e: e.qoq > 0, "qoq_accel": lambda e: e.qoq_accel > 0,
       "yoy_accel_qoq_gt0": lambda e: (e.rev_accel > 0) & (e.qoq > 0),
       "yoy_accel_or_missing": lambda e: (e.rev_accel > 0) | e.rev_accel.isna()}
def pick(price=5, mcap=500e6, mcap_known=False, vol=1e6, dvol=None, atr=(4.5,10), w52=(7.5,30), box=3.0, rev="yoy_accel", nb="half"):
    m = ((E.raw_close > price) & (E.avgvol30 > vol) & E.atr_pct.between(*atr) & E.off52.between(-w52[1], -w52[0])
         & (E.box_h_atr <= box) & REV[rev](E).fillna(False))
    m &= (E.mcap > mcap) if mcap_known else (E.mcap.isna() | (E.mcap > mcap))
    if dvol: m &= E.dollarvol > dvol
    tt = E[m].copy(); on = nbon[m.to_numpy()]
    if nb == "half": tt["rw"] = np.where(on, 0.5, 1.0)
    elif nb == "block": tt = tt[~on]; tt["rw"] = 1.0
    else: tt["rw"] = 1.0
    return tt
G0 = {
 "kurs": {"V4.1 (korrigiert)": {}, "Kurs > 10": dict(price=10), "Kurs > 20": dict(price=20),
          "Mcap > 1 Mrd": dict(mcap=1e9), "Mcap > 2 Mrd": dict(mcap=2e9), "Mcap > 5 Mrd": dict(mcap=5e9), "Mcap bekannt > 500 Mio": dict(mcap_known=True),
          "Volumen > 2 Mio": dict(vol=2e6), "Volumen > 3 Mio": dict(vol=3e6), "Dollar-Volumen > 20 Mio": dict(dvol=20e6), "Dollar-Volumen > 50 Mio": dict(dvol=50e6),
          "Box <= 2,5 ATR": dict(box=2.5), "Box <= 2,0 ATR": dict(box=2.0)},
 "atr52": {"ATR ab 4,0": dict(atr=(4.0,10)), "ATR ab 5,0": dict(atr=(5.0,10)), "ATR ab 5,5": dict(atr=(5.5,10)), "ATR bis 8": dict(atr=(4.5,8)), "ATR bis 12": dict(atr=(4.5,12)),
          "52W 5-30": dict(w52=(5,30)), "52W 10-30": dict(w52=(10,30)), "52W 7,5-25": dict(w52=(7.5,25)), "52W 7,5-35": dict(w52=(7.5,35)), "52W 7,5-40": dict(w52=(7.5,40)),
          "Notbremse aus": dict(nb="off"), "Notbremse: kein Entry": dict(nb="block")},
 "umsatz": {f"Umsatz: {k}": dict(rev=k) for k in REV if k != "yoy_accel"},
}
G = {"kombi2": {"_": {}}, "kombi": {"V4.1 (korrigiert)": {}, "52W 10-30": dict(w52=(10,30)),
     "52W 10-30 + Umsatz beschl. & YoY < 25": dict(w52=(10,30), rev="yoy_accel_lt25"),
     "Umsatz beschl. & YoY < 25": dict(rev="yoy_accel_lt25")}}
rows = []
for name, kw in ([] if GRP=="kombi2" else G[GRP].items()):
    tt = pick(**kw); cl = tt[~tt.open]; w = cl.R > 0
    c, n = run(tt); s = st(c, n)
    rows.append(dict(Variante=name, Signale_Jahr=round(len(tt)/YRS), Avg_R=round(cl.R.mean(),3), Win=round(100*w.mean(),1),
        PF=round(cl.R[w].sum()/-cl.R[~w].sum(),2), R_A=round(cl[cl.half=="A"].R.mean(),2), R_B=round(cl[cl.half=="B"].R.mean(),2),
        R_19_22=round(cl[cl.date<"2023"].R.mean(),2), R_23_26=round(cl[cl.date>="2023"].R.mean(),2),
        CAGR=s["CAGR"], MaxDD=s["MaxDD"], Sharpe=s["Sharpe"], Schlechtestes_Jahr=s["Schlechtestes_Jahr"], Trades_Jahr=s["Trades_Jahr"]))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"gates_{GRP}.csv", index=False)
if GRP == "kombi2":
    tt = pick(w52=(10,30), rev="yoy_accel_lt25"); grid = []
    for rp, cp in [(1,20),(1.25,20),(1.5,20),(1.5,15)]:
        c, n = run(tt, risk_pct=rp, cap_pct=cp); s = st(c, n); grid.append(dict(Risiko=rp, Deckel=cp, **s)); print(grid[-1], flush=True)
    pd.DataFrame(grid).to_csv("gates_grid3.csv", index=False)
