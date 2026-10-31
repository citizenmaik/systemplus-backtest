"""V4.2 – Varianten: kein Entry 0..N Handelstage nach Earnings (N = 3, 4, 5).

Tag 0 = Reaktionstag (Meldung vor 12 Uhr: selber Tag, sonst nächster Handelstag) – wie
sinceEarn im Pine-Skript V4.2. Signale werden vor der Kontosimulation entfernt, d. h. frei
werdende Slots/Cash gehen an andere Signale. Ausgabe: backtest/out/v42_earnings_varianten.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtest")); sys.path.insert(0, str(ROOT / "scripts"))
import v42_report as R

tr, closes = R.signals()
cal = closes.index
E = pd.read_csv(ROOT / "data" / "earnings.csv", keep_default_na=False, na_values=[""])
E = E[E["date"].ne("")]
E["d"] = pd.to_datetime(E["date"]); E["hh"] = E["time"].fillna("16:00").str.slice(0, 2).replace("", "16").astype(int)
ri = cal.searchsorted(E["d"].to_numpy())
same = (ri < len(cal)) & (cal[np.minimum(ri, len(cal) - 1)] == E["d"].to_numpy())
E["ri"] = ri + (same & (E["hh"].to_numpy() >= 12))
EV = {t: np.sort(g["ri"].unique()) for t, g in E.groupby("ticker")}
pos = {d: i for i, d in enumerate(cal)}
def since(r):
    ev = EV.get(r.ticker)
    if ev is None: return np.nan
    i0 = pos[r.date]; pv = ev[ev <= i0]
    return i0 - pv[-1] if len(pv) else np.nan
tr["since"] = [since(r) for r in tr.itertuples()]

def met(c):
    r = c.pct_change().dropna(); yrs = (c.index[-1] - c.index[0]).days / 365.25
    return 100 * ((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1), 100 * (c / c.cummax() - 1).min(), r.mean() / r.std() * np.sqrt(252)

rows = []
base_taken = None
for N in [None, 3, 4, 5]:
    t = tr if N is None else tr[~tr["since"].between(0, N)]
    c, taken, log, e = R.account(t.copy(), closes)
    k = [x[0] for x in taken]; tt = tr.loc[k]
    if N is None: base_taken = tt
    cg, dd, sh = met(c); c1 = c[c.index < "2023-01-01"]; c2 = c[c.index >= "2023-01-01"]
    yr = {f"Rendite {y}": round(100 * (g.iloc[-1] / c[c.index < f"{y}-01-01"].iloc[-1] - 1), 1) if y > 2019 else round(100 * (g.iloc[-1] / 10000 - 1), 1)
          for y, g in c.groupby(c.index.year)}
    rows.append({"Variante": "Basis" if N is None else f"ohne 0–{N} T nach Earnings", "CAGR": round(cg, 1), "MaxDD": round(dd, 1),
                 "Sharpe": round(sh, 2), "CAGR 19–22": round(met(c1)[0], 1), "CAGR 23–26": round(met(c2)[0], 1),
                 "Ende": round(c.iloc[-1]), "Trades": len(tt), "Ø R": round(tt.R.mean(), 2), "Gewinnquote": round(100 * (tt.R > 0).mean()),
                 "Signale gesperrt": 0 if N is None else int(tr["since"].between(0, N).sum()), **yr})
    print(rows[-1], flush=True)
b = base_taken
print("Basis-Trades ohne Earnings-Daten:", int(b["since"].isna().sum()), "von", len(b))
for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 2), (3, 5), (6, 10), (11, 999)]:
    z = b[b["since"].between(lo, hi)]
    print(f"Tag {lo}–{hi}: n {len(z)}  Ø R {z.R.mean():+.2f}  Gewinnquote {100*(z.R>0).mean():.0f} %")
pd.DataFrame(rows).to_csv(ROOT / "backtest" / "out" / "v42_earnings_varianten.csv", index=False)
