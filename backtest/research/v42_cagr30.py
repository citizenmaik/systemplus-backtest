"""V4.2 – Hebel für mehr CAGR: Zins auf Cash, Risiko je Trade, Positionsdeckel, Margin, Tageslimit.
Basis = aktuelle Regeln aus v42_report. Ausgabe: backtest/out/v42_cagr30.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtest")); sys.path.insert(0, str(ROOT / "scripts"))
import v42_report as R

tr, closes = R.signals()
irx = pd.read_csv(ROOT / "data" / "irx.csv", parse_dates=["date"]).set_index("date")["irx"].reindex(closes.index).ffill().bfill()
cash_z = (irx - 0.5).clip(lower=0)   # Guthabenzins wie IBKR: Benchmark − 0,5 %
debit_z = irx + 1.5                  # Sollzins Margin

def met(c):
    r = c.pct_change().dropna(); yrs = (c.index[-1] - c.index[0]).days / 365.25
    return 100 * ((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1), 100 * (c / c.cummax() - 1).min(), r.mean() / r.std() * np.sqrt(252)

V = [  # Name, RISK, CAP, PER_DAY, lev, Guthabenzins
    ("Basis", 1.0, 15.0, None, 1.0, False),
    ("Zins auf Cash", 1.0, 15.0, None, 1.0, True),
    ("Risiko 1,25 %", 1.25, 15.0, None, 1.0, False),
    ("Risiko 1,5 %", 1.5, 15.0, None, 1.0, False),
    ("Deckel 20 %", 1.0, 20.0, None, 1.0, False),
    ("Margin bis 1,25x", 1.0, 15.0, None, 1.25, False),
    ("max. 3/Tag", 1.0, 15.0, 3, 1.0, False),
    ("Zins + Risiko 1,25 %", 1.25, 15.0, None, 1.0, True),
    ("Zins + Margin 1,25x", 1.0, 15.0, None, 1.25, True),
    ("Zins + Risiko 1,25 % + Deckel 20 %", 1.25, 20.0, None, 1.0, True),
    ("Zins + Deckel 20 %", 1.0, 20.0, None, 1.0, True),
    ("Zins + Deckel 20 % + max. 3/Tag", 1.0, 20.0, 3, 1.0, True),
]
rows = []
for name, risk, cap, pd_, lev, cz in V:
    R.RISK, R.CAP, R.PER_DAY = risk, cap, pd_
    st = {}
    c, taken, log, e = R.account(tr.copy(), closes, st, lev=lev, rate=debit_z, cash_rate=cash_z if cz else None)
    tt = tr.loc[[k for k, _, _ in taken]]
    cg, dd, sh = met(c); c1 = c[c.index < "2023-01-01"]; c2 = c[c.index >= "2023-01-01"]
    rows.append({"Variante": name, "CAGR": round(cg, 1), "MaxDD": round(dd, 1), "Sharpe": round(sh, 2),
                 "CAGR 19–22": round(met(c1)[0], 1), "CAGR 23–26": round(met(c2)[0], 1), "Ende": round(c.iloc[-1]),
                 "Trades": len(tt), "Ø investiert %": round(100 * e.mean()), "Zins USD": round(st["interest"])})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "backtest" / "out" / "v42_cagr30.csv", index=False)
