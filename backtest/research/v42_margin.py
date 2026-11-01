"""V4.2 – Margin-Varianten: Positionswert bis lev × Konto, wenn das Cash nicht reicht.

Regeln sonst wie v42_report (inkl. Earnings-Sperre, max. 3 Käufe/Tag, 1 % Risiko, 15 % Deckel).
Zins auf negatives Cash: 13-Wochen-T-Bill (data/irx.csv) + Aufschlag, täglich /252.
Ausgabe: backtest/out/v42_margin.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtest")); sys.path.insert(0, str(ROOT / "scripts"))
import v42_report as R

tr, closes = R.signals()
irx = pd.read_csv(ROOT / "data" / "irx.csv", parse_dates=["date"]).set_index("date")["irx"].reindex(closes.index).ffill().bfill()

def met(c):
    r = c.pct_change().dropna(); yrs = (c.index[-1] - c.index[0]).days / 365.25
    return 100 * ((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1), 100 * (c / c.cummax() - 1).min(), r.mean() / r.std() * np.sqrt(252)

rows = []
for lev, spread in [(1.0, None), (1.25, 1.5), (1.5, 1.5), (2.0, 1.5), (1.5, 0.0), (1.5, 3.0)]:
    st = {}
    c, taken, log, e = R.account(tr.copy(), closes, st, lev=lev, rate=None if spread is None else irx + spread)
    tt = tr.loc[[k for k, _, _ in taken]]
    cg, dd, sh = met(c); c1 = c[c.index < "2023-01-01"]; c2 = c[c.index >= "2023-01-01"]
    sk = pd.Series([g for _, g, *_ in log]).value_counts()
    ye = c.resample("YE").last(); yb = pd.concat([pd.Series([10000.0]), ye.iloc[:-1]]).to_numpy()
    rows.append({"Variante": "ohne Margin" if lev == 1 else f"bis {lev:.2f}x, Zins T-Bill+{spread:.1f} %",
                 "CAGR": round(cg, 1), "MaxDD": round(dd, 1), "Sharpe": round(sh, 2), "CAGR 19–22": round(met(c1)[0], 1),
                 "CAGR 23–26": round(met(c2)[0], 1), "Ende": round(c.iloc[-1]), "Trades": len(tt), "Ø R": round(tt.R.mean(), 2),
                 "Ø investiert %": round(100 * e.mean()), "Tage > 100 %": int((e > 1.0001).sum()), "Max. investiert %": round(100 * e.max()),
                 "Zins USD": round(-st["interest"]), "kein Cash": int(sk.get("kein Cash", 0)), "gekürzt": int(sk.get("gekauft, gekürzt", 0)),
                 **{f"Rendite {d.year}": round(100 * (v / yb[i] - 1), 1) for i, (d, v) in enumerate(ye.items())}})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "backtest" / "out" / "v42_margin.csv", index=False)
