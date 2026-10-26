"""Historische Earnings-Termine (Yahoo) und 13-Wochen-T-Bill-Zins (^IRX) für Research.

-> data/earnings.csv  (ticker, date, time, eps_est, eps_act)
-> data/irx.csv       (date, irx = Zins in % p. a.)
Ticker aus universe/earnings_tickers.txt (alle Titel mit V4-Signalen).
"""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from common import DATA, ROOT

TICKERS = ROOT / "universe" / "earnings_tickers.txt"


def main() -> None:
    irx = yf.download("^IRX", start="2018-01-01", progress=False, auto_adjust=False)
    irx = irx["Close"].squeeze().rename("irx").dropna()
    irx.index.name = "date"
    irx.round(4).to_csv(DATA / "irx.csv")
    print(f"IRX: {len(irx)} Tage")

    tickers = [t.strip() for t in TICKERS.read_text().splitlines() if t.strip()]
    rows, miss = [], []
    for i, t in enumerate(tickers, 1):
        for attempt in range(3):
            try:
                e = yf.Ticker(t).get_earnings_dates(limit=40)
                break
            except Exception as ex:  # noqa: BLE001
                e = None
                if attempt == 2:
                    print(f"  {t}: {ex}")
                time.sleep(2 * (attempt + 1))
        if e is None or e.empty:
            miss.append(t)
        else:
            for ts, r in e.iterrows():
                ts = pd.Timestamp(ts)
                rows.append((t, ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M"),
                             r.get("EPS Estimate"), r.get("Reported EPS")))
        time.sleep(0.3)
        if i % 200 == 0:
            print(f"  {i}/{len(tickers)}, ohne Daten {len(miss)}")
    df = pd.DataFrame(rows, columns=["ticker", "date", "time", "eps_est", "eps_act"])
    df.to_csv(DATA / "earnings.csv", index=False)
    print(f"Earnings: {df.ticker.nunique()} Titel, {len(df)} Termine, ohne Daten {len(miss)}")


if __name__ == "__main__":
    main()
