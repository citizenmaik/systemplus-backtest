"""Datenprüfung je Ticker -> data/quality.csv. Markiert nur, löscht nichts.

status: ok | warn | bad
  bad  = ungültige Kerzen (High < Low, Kurs <= 0) gehäuft oder > 5 % fehlende Handelstage
  warn = Tagesbewegungen > 50 %, Null-Volumen-Tage, fehlende Tage, veraltet (delistet?)"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import DATA, QUALITY, read_splits, read_store, to_adjusted


def check(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    cal = np.sort(df.loc[df["ticker"] == "SPY", "date"].unique())
    if len(cal) == 0:
        cal = np.sort(df["date"].unique())

    tol = 1e-3
    bad_ohlc = ((df["high"] < df["low"])
                | (df["close"] > df["high"] * (1 + tol)) | (df["close"] < df["low"] * (1 - tol))
                | (df["open"] > df["high"] * (1 + tol)) | (df["open"] < df["low"] * (1 - tol)))
    nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    ret = df.groupby("ticker")["close"].pct_change()

    g = df.assign(bad_ohlc=bad_ohlc, nonpos=nonpos, big_move=ret.abs() > 0.5,
                  zero_vol=df["volume"] <= 0).groupby("ticker")
    q = pd.DataFrame({
        "first_date": g["date"].min(),
        "last_date": g["date"].max(),
        "bars": g.size(),
        "bad_ohlc": g["bad_ohlc"].sum(),
        "nonpos": g["nonpos"].sum(),
        "big_moves": g["big_move"].sum(),
        "zero_vol_days": g["zero_vol"].sum(),
    })
    lo = np.searchsorted(cal, q["first_date"].values, side="left")
    hi = np.searchsorted(cal, q["last_date"].values, side="right")
    q["missing_days"] = np.maximum((hi - lo) - q["bars"].values, 0)
    q["missing_pct"] = (100 * q["missing_days"] / np.maximum(hi - lo, 1)).round(2)
    stale_cut = cal[-6] if len(cal) >= 6 else cal[0]
    q["stale"] = q["last_date"] < stale_cut

    bad = (q["bad_ohlc"] > 5) | (q["nonpos"] > 0) | (q["missing_pct"] > 5)
    warn = (q["big_moves"] > 0) | (q["zero_vol_days"] > 0) | (q["missing_days"] > 0) | q["stale"]
    q["status"] = np.where(bad, "bad", np.where(warn, "warn", "ok"))
    q["first_date"] = q["first_date"].dt.date
    q["last_date"] = q["last_date"].dt.date
    return q.reset_index()


if __name__ == "__main__":
    q = check(to_adjusted(read_store(), read_splits()))  # bereinigt, sonst zählen Splits als Kurssprung
    q.to_csv(QUALITY, index=False)
    summary = {"tickers": len(q), **q["status"].value_counts().to_dict(),
               "stale": int(q["stale"].sum())}
    (DATA / "quality_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
