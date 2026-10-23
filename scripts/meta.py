"""Sektor, Branche, Market Cap und Float je Ticker (Stand: Abrufdatum, NICHT point-in-time).

Lädt nur fehlende oder über 30 Tage alte Einträge, höchstens --max pro Lauf,
und speichert Zwischenstände, falls Yahoo drosselt."""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from common import BENCHMARKS, META, load_universe

FIELDS = ["sector", "industry", "quoteType", "marketCap", "sharesOutstanding", "floatShares"]
MAX_AGE_DAYS = 30


def main(max_n: int, pause: float) -> None:
    uni = load_universe()
    meta = pd.read_csv(META, dtype={"ticker": str}) if META.exists() else pd.DataFrame(columns=["ticker", "fetched"] + FIELDS)
    meta = meta.set_index("ticker")
    cutoff = (date.today() - timedelta(days=MAX_AGE_DAYS)).isoformat()
    todo = [t for t in uni["ticker"] if t not in BENCHMARKS
            and (t not in meta.index or str(meta.at[t, "fetched"]) < cutoff)][:max_n]
    print(f"Metadaten für {len(todo)} Ticker")
    fails = 0
    for i, t in enumerate(todo, 1):
        try:
            info = yf.Ticker(t).get_info() or {}
            meta.loc[t, FIELDS] = [info.get(f) for f in FIELDS]
            meta.loc[t, "fetched"] = date.today().isoformat()
            fails = 0
        except Exception as e:
            fails += 1
            print(f"  {t}: {type(e).__name__}")
            if "RateLimit" in type(e).__name__ or fails >= 5:
                print("  Gedrosselt – Abbruch, Rest im nächsten Lauf")
                break
        if i % 100 == 0:
            meta.reset_index().to_csv(META, index=False)
            print(f"  {i}/{len(todo)}")
        time.sleep(pause)
    META.parent.mkdir(parents=True, exist_ok=True)
    meta.index.name = "ticker"
    meta.reset_index().sort_values("ticker").to_csv(META, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1500)
    ap.add_argument("--pause", type=float, default=0.5)
    a = ap.parse_args()
    main(a.max, a.pause)
