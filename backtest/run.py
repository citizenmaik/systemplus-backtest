"""System+ v3.8 Backtest auf dem Marktdatensatz – reproduziert alle Kennzahlen.

    python backtest/run.py            # Ergebnisse nach backtest/out/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import signals
from simulate import account, acc_stats, no_overlap, simulate_trades, stats

OUT = Path(__file__).resolve().parent / "out"
LO, HI = "2021-01-01", "2026-10-09"  # Zeitraum des 518-Titel-Backtests


def trade_row(name, tr, m, lo=LO, hi=HI):
    t = tr[m & (tr["date"] >= lo) & (tr["date"] <= hi)]
    t = t[no_overlap(t)]
    s = stats(t)
    closed = t[~t["open"]]
    g = closed.groupby(closed["date"].dt.to_period("M"))["R"].agg(["sum", "count"])
    rng = np.random.default_rng(0)
    bs = [g.iloc[rng.integers(0, len(g), len(g))].pipe(lambda z: z["sum"].sum() / z["count"].sum())
          for _ in range(2000)]
    lo5, hi95 = np.percentile(bs, [5, 95])
    return dict(Szenario=name, Trades=s["n"], Gewinnquote=round(s["win"], 3), ØR=round(s["avgR"], 2),
                Band90=f"{lo5:+.2f} bis {hi95:+.2f}", MedianR=round(s["medR"], 2), PF=round(s["PF"], 2),
                SL=round(s["sl"], 1), Gewinn=round(s["gain"], 1), Verlust=round(s["loss"], 1))


def main():
    OUT.mkdir(exist_ok=True)
    x = signals.build(start="2019-01-01")
    core = ["g_atr", "g_sl", "g_52w", "g_cb"]
    sig = x[x["entry_sig"] & x[core].all(axis=1)]
    tr = simulate_trades(x, sig).join(sig[["atr_mult", "g_screen", "g_lead", "g_rs", "g_sec", "cohort",
                                           "rs_rating", "sector_rank", "sector_etf"]])
    tr.drop(columns=["fills"]).to_csv(OUT / "trades.csv")

    o518 = tr["cohort"].isin(["handverlesen", "zufall"])
    rows = [
        trade_row("518 Titel · alle Gates (wie alter Backtest)", tr, o518 & tr["g_lead"]),
        trade_row("518 Titel · alle Gates + Screener am Entry-Tag", tr, o518 & tr["g_lead"] & tr["g_screen"]),
        trade_row("Zufall (291) · alle Gates", tr, (tr["cohort"] == "zufall") & tr["g_lead"]),
        trade_row("Handverlesen (231) · alle Gates", tr, (tr["cohort"] == "handverlesen") & tr["g_lead"]),
        trade_row("Übriger Markt · alle Gates + Screener", tr, (tr["cohort"] == "neu") & tr["g_lead"] & tr["g_screen"]),
        trade_row("Gesamtmarkt · alle Gates + Screener", tr, tr["g_lead"] & tr["g_screen"]),
        trade_row("Gesamtmarkt · ohne Führung + Screener", tr, tr["g_screen"]),
    ]
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "trades_summary.csv", index=False)
    print(T.to_string(index=False))

    xs = x[["date", "ticker", "close"]]
    acc = []
    for name, m in [("518 Titel · alle Gates", o518 & tr["g_lead"]),
                    ("Gesamtmarkt · alle Gates + Screener", tr["g_lead"] & tr["g_screen"])]:
        for cap in (10, 20):
            t = tr[m & (tr["date"] >= LO)]
            cur, tk = account(xs[xs["date"] >= LO], t, cap_pct=cap)
            a = acc_stats(cur)
            acc.append(dict(Szenario=name, Deckel=cap, Trades=len(tk), CAGR=round(a["cagr"], 3),
                            MaxDD=round(a["maxdd"], 3), Endwert=round(a["end"])))
            cur.to_csv(OUT / f"equity_{'518' if '518' in name else 'markt'}_{cap}.csv")
    A = pd.DataFrame(acc)
    A.to_csv(OUT / "account_summary.csv", index=False)
    print(A.to_string(index=False))


if __name__ == "__main__":
    main()
