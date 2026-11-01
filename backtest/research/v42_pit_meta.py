"""V4.2 – Market Cap/Float heutiger Stand (meta.csv) gegen damaligen Stand (SEC, pit_meta.py).
Ausgabe: backtest/out/v42_pit_meta.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtest")); sys.path.insert(0, str(ROOT / "scripts"))
import v42_report as R
from signals import build
from pit_meta import add_pit

x = add_pit(build(start="2019-01-01", verbose=False))
s = x[x.entry_sig]
print("Breakout-Tage mit SEC-Float:", round(100 * s.float_pit.notna().mean(), 1), "%  mit SEC-Market-Cap:",
      round(100 * s.mcap_pit.notna().mean(), 1), "%")
b = s.dropna(subset=["float_pit", "float_pct"])
print("Float > 70 % heute / damals gleich:", round(100 * ((b.float_pct > 70) == (b.float_pit > 70)).mean(), 1), "%",
      " heute ≤70 aber damals >70:", int(((b.float_pct <= 70) & (b.float_pit > 70)).sum()),
      " heute >70 aber damals ≤70:", int(((b.float_pct > 70) & (b.float_pit <= 70)).sum()))
r = (s.mcap_pit / s.mcap).dropna()
print("Market Cap damals/heute-Rechnung Median", round(r.median(), 2), " Anteil < 0,5 oder > 2:", round(100 * ((r < .5) | (r > 2)).mean(), 1), "%")

def met(c):
    q = c.pct_change().dropna(); yrs = (c.index[-1] - c.index[0]).days / 365.25
    return 100 * ((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1), 100 * (c / c.cummax() - 1).min(), q.mean() / q.std() * np.sqrt(252)

rows, sets = [], {}
for name, pit, pf in [("heutiger Stand (meta.csv)", False, False), ("nur Market Cap damals", True, False), ("Market Cap und Float damals (SEC)", True, True)]:
    tr, closes = R.signals(x=x, pit=pit, pit_float=pf)
    c, taken, log, e = R.account(tr, closes)
    tt = tr.loc[[k for k, _, _ in taken]]; sets[(pit, pf)] = tt
    cg, dd, sh = met(c); c1 = c[c.index < "2023-01-01"]; c2 = c[c.index >= "2023-01-01"]
    ye = c.resample("YE").last(); yb = pd.concat([pd.Series([10000.0]), ye.iloc[:-1]]).to_numpy()
    rows.append({"Variante": name, "CAGR": round(cg, 1), "MaxDD": round(dd, 1), "Sharpe": round(sh, 2),
                 "CAGR 19–22": round(met(c1)[0], 1), "CAGR 23–26": round(met(c2)[0], 1), "Ende": round(c.iloc[-1]),
                 "Trades": len(tt), "Ø R": round(tt.R.mean(), 2),
                 **{f"Rendite {d.year}": round(100 * (v / yb[i] - 1), 1) for i, (d, v) in enumerate(ye.items())}})
    print(rows[-1], flush=True)
a, b2 = sets[(False, False)], sets[(True, False)]
ka, kb = set(zip(a.date, a.ticker)), set(zip(b2.date, b2.ticker))
f = lambda d: " ".join(f"{t}({str(dt.date())},{r:+.1f}R)" for dt, t, r in zip(d.date, d.ticker, d.R))
print("nur heute (vs. nur Market Cap damals):", f(a[[k in ka - kb for k in zip(a.date, a.ticker)]]))
print("nur Market Cap damals:", f(b2[[k in kb - ka for k in zip(b2.date, b2.ticker)]]))
pd.DataFrame(rows).to_csv(ROOT / "backtest" / "out" / "v42_pit_meta.csv", index=False)
