"""Datensatz für den Backtest laden – mit Filtern, die Tag für Tag (point-in-time) gelten.

    from load import load
    df = load()                      # Langformat, alle Ticker
    df = load(cohorts=["zufall"])    # nur eine Kohorte
    panel = {t: g for t, g in df.groupby("ticker")}

open/high/low/close/volume sind split-bereinigt (für Indikatoren, Renditen, R).
Zusätzlich:
    raw_close, raw_volume  wie damals gehandelt (für die Grundfilter)
    bar_no      Handelstag seit erstem Bar (IPO-Proxy)
    avgvol30    Ø-Rohvolumen 30 Tage
    dollarvol30 Ø-Dollar-Umsatz 30 Tage (Ersatz für Market Cap, die es nicht point-in-time gibt)
    atr_pct     ATR14 (Wilder) in % vom Schlusskurs
    base_ok     Rohkurs > 5, avgvol30 >= 1 Mio., bar_no >= 200, Datenqualität nicht 'bad'
    atr_ok      2,5 % <= atr_pct <= 10 %
    cohort      handverlesen | zufall | neu (aus universe/cohorts.csv, Standard 'neu')
    sector, industry, float_pct   Stand Abrufdatum, NICHT point-in-time
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import BENCHMARKS, COHORTS, META, QUALITY, read_csv, read_splits, read_store, to_adjusted

MIN_BARS = 200
ATR_BAND = (2.5, 10.0)


def _wilder_atr_pct(g: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = g["close"].shift()
    tr = np.maximum(g["high"] - g["low"],
                    np.maximum((g["high"] - prev).abs(), (g["low"] - prev).abs()))
    return 100 * tr.ewm(alpha=1 / n, adjust=False).mean() / g["close"]


def load(start: str | None = None, tickers: list[str] | None = None,
         cohorts: list[str] | None = None, include_benchmarks: bool = True) -> pd.DataFrame:
    df = read_store()
    if tickers is not None:
        keep = set(tickers) | (set(BENCHMARKS) if include_benchmarks else set())
        df = df[df["ticker"].isin(keep)]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    raw_close, raw_volume = df["close"].copy(), df["volume"].copy()
    df = to_adjusted(df, read_splits())
    df["raw_close"], df["raw_volume"] = raw_close.values, raw_volume.values

    g = df.groupby("ticker", sort=False)
    df["bar_no"] = g.cumcount()
    df["avgvol30"] = df.groupby("ticker")["raw_volume"].transform(
        lambda v: v.rolling(30, min_periods=30).mean())
    df["dollarvol30"] = (df["raw_close"] * df["raw_volume"]).groupby(df["ticker"]).transform(
        lambda v: v.rolling(30, min_periods=30).mean())
    df["atr_pct"] = g[["high", "low", "close"]].apply(_wilder_atr_pct).reset_index(level=0, drop=True)

    # Kohorten
    coh = read_csv(COHORTS, dtype=str) if COHORTS.exists() else pd.DataFrame(columns=["ticker", "cohort"])
    df = df.merge(coh[["ticker", "cohort"]].drop_duplicates("ticker"), on="ticker", how="left")
    df["cohort"] = df["cohort"].fillna("neu")
    df.loc[df["ticker"].isin(BENCHMARKS), "cohort"] = "benchmark"

    # Datenqualität
    if QUALITY.exists():
        q = read_csv(QUALITY)[["ticker", "status"]]
        df = df.merge(q.rename(columns={"status": "quality"}), on="ticker", how="left")
    else:
        df["quality"] = "ok"

    # Metadaten (Stand heute)
    if META.exists():
        m = read_csv(META)
        m["float_pct"] = (100 * m["floatShares"] / m["sharesOutstanding"]).round(1)
        df = df.merge(m[["ticker", "sector", "industry", "float_pct"]], on="ticker", how="left")

    df["base_ok"] = ((df["raw_close"] > 5) & (df["avgvol30"] >= 1_000_000)
                     & (df["bar_no"] >= MIN_BARS) & (df["quality"] != "bad"))
    df["atr_ok"] = df["atr_pct"].between(*ATR_BAND)

    if cohorts is not None:
        df = df[df["cohort"].isin(set(cohorts) | ({"benchmark"} if include_benchmarks else set()))]
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]  # erst nach Indikatorberechnung schneiden
    return df.reset_index(drop=True)


if __name__ == "__main__":
    d = load()
    last = d[d["date"] == d["date"].max()]
    print(f"{d['ticker'].nunique()} Ticker, {len(d):,} Zeilen, {d['date'].min().date()} bis {d['date'].max().date()}")
    print(f"Am letzten Tag: {int(last['base_ok'].sum())} erfüllen die Grundfilter, "
          f"{int((last['base_ok'] & last['atr_ok']).sum())} davon auch das ATR-Band")
    print(d.drop_duplicates("ticker")["cohort"].value_counts().to_string())
