"""V4.2 – SL = SMA10 vom Entry-Tag (aktuell) gegen SL = Tief der Entry-Kerze.
Übrige Regeln wie v42_report (Fail/TP1/TP2 an SMA10/SMA20, 1 % Risiko, 15 % Deckel, max. 3/Tag, Zins).
Ausgabe: backtest/out/v42_stop_low.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtest")); sys.path.insert(0, str(ROOT / "scripts"))
import v42_report as R, simulate as S
from signals import build
from pit_meta import add_pit

x = add_pit(build(start="2019-01-01", verbose=False))

def met(c):
    r = c.pct_change().dropna(); yrs = (c.index[-1] - c.index[0]).days / 365.25
    return 100 * ((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1), 100 * (c / c.cummax() - 1).min(), r.mean() / r.std() * np.sqrt(252)

rows = []
for name, mode in [("SL = SMA10 (aktuell)", "sma10"), ("SL = Tief Entry-Tag", "low")]:
    S.STOP_MODE = mode
    tr, closes = R.signals(x=x)
    c, taken, log, e = R.account(tr, closes, cash_rate=R.cash_rate(closes.index))
    tt = tr.loc[[k for k, _, _ in taken]]
    tags = pd.Series([f[3] for fl in tt.fills for f in fl[-1:]]).value_counts(normalize=True).mul(100).round()
    cg, dd, sh = met(c); c1 = c[c.index < "2023-01-01"]; c2 = c[c.index >= "2023-01-01"]
    ye = c.resample("YE").last(); yb = pd.concat([pd.Series([10000.0]), ye.iloc[:-1]]).to_numpy()
    rows.append({"Variante": name, "CAGR": round(cg, 1), "MaxDD": round(dd, 1), "Sharpe": round(sh, 2),
                 "CAGR 19–22": round(met(c1)[0], 1), "CAGR 23–26": round(met(c2)[0], 1), "Ende": round(c.iloc[-1]),
                 "Trades": len(tt), "Gewinnquote": round(100 * (tt.R > 0).mean()), "Ø R": round(tt.R.mean(), 2),
                 "Ø Ergebnis %": round(tt.pct.mean(), 2), "Ø SL-Abstand %": round(tt.sl_dist.mean(), 1),
                 "Median SL-Abstand %": round(tt.sl_dist.median(), 1), "1R erreicht %": round(100 * tt.r1.mean()),
                 "Ø Haltedauer": round(tt.bars_held.mean(), 1), "Ø investiert %": round(100 * e.mean()),
                 "SL-Exit %": tags.get("SL", 0), "Fail-Exit %": tags.get("Fail", 0),
                 **{f"Rendite {d.year}": round(100 * (v / yb[i] - 1), 1) for i, (d, v) in enumerate(ye.items())}})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "backtest" / "out" / "v42_stop_low.csv", index=False)
