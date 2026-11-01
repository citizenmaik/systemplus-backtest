"""Market Cap und Float zum damaligen Stand (point-in-time) aus SEC-Daten.

Aktienzahl: dei:EntityCommonStockSharesOutstanding (Deckblatt 10-Q/10-K), sonst
us-gaap:CommonStockSharesOutstanding (Bilanz). Bekannt ab dem Tag nach `filed`.
  mcap_pit = Kurs (nicht splitbereinigt) × letzte bekannte Aktienzahl
Public Float: dei:EntityPublicFloat (USD, Stichtag Ende Q2 des Geschäftsjahres, gemeldet im 10-K).
  float_pit = Public Float / (Kurs am Stichtag × Aktienzahl am Stichtag), höchstens 100 %,
  bekannt ab dem Tag nach `filed`, gültig bis 400 Tage danach.
Ohne SEC-Wert bleiben die Spalten leer; der Aufrufer nimmt dann den heutigen Stand (meta.csv).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _first(f: pd.DataFrame, k: str, prio: list[str]) -> pd.DataFrame:
    g = f[f["kennzahl"] == k].dropna(subset=["end", "filed"]).copy()
    g["p"] = g["konzept"].map({c: i for i, c in enumerate(prio)}).fillna(len(prio))
    g = g[g["wert"] > 0].sort_values(["p", "filed"]).drop_duplicates(["ticker", "end"], keep="first")
    g["known"] = (g["filed"] + pd.Timedelta(days=1)).astype("datetime64[ns]")
    g["end"] = g["end"].astype("datetime64[ns]")
    return g[["ticker", "end", "known", "wert"]]


def add_pit(x: pd.DataFrame, fund_path: Path | None = None) -> pd.DataFrame:
    """x: Spalten date, ticker, raw_close. Ergänzt shares_pit, mcap_pit, float_pit (Index bleibt)."""
    f = pd.read_parquet(fund_path or ROOT / "data" / "fundamentals.parquet",
                        filters=[("kennzahl", "in", ["shares", "public_float"])])
    out = pd.DataFrame(index=x.index, columns=["shares_pit", "mcap_pit", "float_pit"], dtype=float)
    if f.empty:
        return x.join(out)
    sh = _first(f, "shares", ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"])
    pf = _first(f, "public_float", ["EntityPublicFloat"])
    t = x[["date", "ticker", "raw_close"]].copy()
    t["date"] = t["date"].astype("datetime64[ns]")
    t["_i"] = t.index
    t = t.sort_values("date")
    m = pd.merge_asof(t, sh.sort_values("known")[["ticker", "known", "wert"]], left_on="date", right_on="known",
                      by="ticker", direction="backward", tolerance=pd.Timedelta(days=200)).set_index("_i")
    out["shares_pit"] = m["wert"]
    out["mcap_pit"] = m["raw_close"] * m["wert"]
    # Float-Anteil am Stichtag: Kurs und Aktienzahl um den Stichtag
    px = t[["date", "ticker", "raw_close"]].rename(columns={"date": "d"}).sort_values("d")
    p = pf.sort_values("end")
    p = pd.merge_asof(p, px, left_on="end", right_on="d", by="ticker", direction="backward",
                      tolerance=pd.Timedelta(days=7))
    p = pd.merge_asof(p.sort_values("end"), sh.rename(columns={"end": "e2", "wert": "sh", "known": "k2"}).sort_values("e2"),
                      left_on="end", right_on="e2", by="ticker", direction="nearest", tolerance=pd.Timedelta(days=120))
    p["fpct"] = np.minimum(100 * p["wert"] / (p["raw_close"] * p["sh"]), 100)
    p = p.dropna(subset=["fpct"]).sort_values("known")
    m2 = pd.merge_asof(t, p[["ticker", "known", "fpct"]], left_on="date", right_on="known", by="ticker",
                       direction="backward", tolerance=pd.Timedelta(days=400)).set_index("_i")
    out["float_pit"] = m2["fpct"]
    return x.join(out)
