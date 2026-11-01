"""V4.2 – Float-Filter (> 70 %, heutiger Stand aus meta.csv) über die ganze Historie,
nur in den letzten 12 Monaten oder gar nicht. Sonst Regeln wie v42_report.
Ausgabe: backtest/out/v42_float.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtest")); sys.path.insert(0, str(ROOT / "scripts"))
import v42_report as R
from signals import build

x = build(start="2019-01-01", verbose=False)

def met(c):
    r = c.pct_change().dropna(); yrs = (c.index[-1] - c.index[0]).days / 365.25
    return 100 * ((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1), 100 * (c / c.cummax() - 1).min(), r.mean() / r.std() * np.sqrt(252)

rows = []
for name, fd in [("Float-Filter ganze Historie", None), ("Float-Filter nur letzte 12 Monate", 365), ("ohne Float-Filter", 0)]:
    tr, closes = R.signals(float_days=fd, x=x)
    c, taken, log, e = R.account(tr, closes)
    tt = tr.loc[[k for k, _, _ in taken]]
    cg, dd, sh = met(c); c1 = c[c.index < "2023-01-01"]; c2 = c[c.index >= "2023-01-01"]
    ye = c.resample("YE").last(); yb = pd.concat([pd.Series([10000.0]), ye.iloc[:-1]]).to_numpy()
    low = tt[tt.ticker.map(x.groupby("ticker").float_pct.first()) <= 70]
    rows.append({"Variante": name, "CAGR": round(cg, 1), "MaxDD": round(dd, 1), "Sharpe": round(sh, 2),
                 "CAGR 19–22": round(met(c1)[0], 1), "CAGR 23–26": round(met(c2)[0], 1), "Ende": round(c.iloc[-1]),
                 "Trades": len(tt), "Ø R": round(tt.R.mean(), 2), "Trades Float ≤ 70": len(low),
                 "Ø R Float ≤ 70": round(low.R.mean(), 2) if len(low) else None,
                 **{f"Rendite {d.year}": round(100 * (v / yb[i] - 1), 1) for i, (d, v) in enumerate(ye.items())}})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "backtest" / "out" / "v42_float.csv", index=False)
