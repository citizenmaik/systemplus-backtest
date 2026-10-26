"""Wachstumsmerkmale aus SEC-Quartalszahlen, point-in-time zum Entry-Tag.

Je Titel und Kennzahl wird das Konzept mit den meisten Quartalswerten genommen. Quartale sind
Zeiträume von 80–100 Tagen; fehlt Q4, wird es aus dem Geschäftsjahr minus Q1–Q3 abgeleitet.
Je Quartal zählt die erste Meldung (filed), spätere Korrekturen nicht. Ein Wert ist ab dem
Tag nach `filed` bekannt.

Merkmale: rev_yoy, rev_yoy_prev, rev_accel (Beschleunigung in Prozentpunkten),
eps_yoy, eps_pos (letztes Quartal EPS > 0), ni_pos (Nettogewinn > 0), fund_age (Tage seit Meldung).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def quarterly(f: pd.DataFrame) -> pd.DataFrame:
    f = f.dropna(subset=["end", "filed"]).copy()
    f["dur"] = (f["end"] - f["start"]).dt.days
    out = []
    for (t, k), g in f.groupby(["ticker", "kennzahl"], sort=False):
        q = g[g["dur"].between(80, 100)]
        best = q["konzept"].value_counts().index[0] if len(q) else g["konzept"].value_counts().index[0]
        g = g[g["konzept"] == best]
        q = g[g["dur"].between(80, 100)].sort_values("filed").drop_duplicates("end", keep="first")
        fy = g[g["dur"].between(350, 380)].sort_values("filed").drop_duplicates("end", keep="first")
        rows = {e: (v, fl) for e, v, fl in zip(q["end"], q["wert"], q["filed"])}
        if k != "eps":  # Q4 = Geschäftsjahr − Q1..Q3 (EPS ist nicht additiv)
            for e, v, fl, s in zip(fy["end"], fy["wert"], fy["filed"], fy["start"]):
                if any(abs((e - x).days) <= 10 for x in rows):
                    continue
                three = [rows[x][0] for x in rows if s - pd.Timedelta(days=10) <= x < e - pd.Timedelta(days=20)]
                if len(three) == 3:
                    rows[e] = (v - sum(three), fl)
        if not rows:
            continue
        d = pd.DataFrame([(t, k, e, v, fl) for e, (v, fl) in rows.items()],
                         columns=["ticker", "kennzahl", "end", "wert", "filed"]).sort_values("end")
        out.append(d)
    return pd.concat(out, ignore_index=True)


def yoy(q: pd.DataFrame) -> pd.DataFrame:
    res = []
    for (t, k), g in q.groupby(["ticker", "kennzahl"], sort=False):
        g = g.sort_values("end").reset_index(drop=True)
        ends = g["end"].to_numpy()
        prev = np.full(len(g), np.nan)
        for i, e in enumerate(ends):
            j = np.where(np.abs((ends - (e - np.timedelta64(365, "D"))).astype("timedelta64[D]").astype(int)) <= 20)[0]
            if len(j):
                prev[i] = g.loc[j[0], "wert"]
        g["prev"] = prev
        base = np.abs(g["prev"])
        g["yoy"] = np.where(base > (0.01 if k == "eps" else 1e5), (g["wert"] - g["prev"]) / base * 100, np.nan)
        res.append(g)
    return pd.concat(res, ignore_index=True)


def asof_features(trades: pd.DataFrame, fund_path: Path | None = None) -> pd.DataFrame:
    """trades: Index = Trade-ID, Spalten date, ticker. Gibt die Merkmale je Trade zurück."""
    f = pd.read_parquet(fund_path or ROOT / "data" / "fundamentals.parquet")
    Y = yoy(quarterly(f))
    Y["known"] = Y["filed"] + pd.Timedelta(days=1)
    out = pd.DataFrame(index=trades.index)
    t = trades[["date", "ticker"]].copy()
    t["date"] = pd.to_datetime(t["date"]).astype("datetime64[ns]")
    Y["known"] = Y["known"].astype("datetime64[ns]")
    t["_i"] = t.index
    t = t.sort_values("date")
    for k in ("revenue", "eps", "net_income"):
        y = Y[Y["kennzahl"] == k].sort_values("known")
        # je Meldetag das jüngste Quartal
        y = y.sort_values(["ticker", "known", "end"]).drop_duplicates(["ticker", "known"], keep="last")
        y["yoy_prev"] = y.groupby("ticker")["yoy"].shift(1)
        m = pd.merge_asof(t, y[["ticker", "known", "wert", "yoy", "yoy_prev", "end"]].sort_values("known"),
                          left_on="date", right_on="known", by="ticker", direction="backward",
                          tolerance=pd.Timedelta(days=200)).set_index("_i")
        if k == "revenue":
            out["rev_yoy"] = m["yoy"]
            out["rev_yoy_prev"] = m["yoy_prev"]
            out["rev_accel"] = m["yoy"] - m["yoy_prev"]
            out["fund_age"] = (m["date"] - m["known"]).dt.days
        elif k == "eps":
            out["eps_yoy"] = m["yoy"]
            out["eps_pos"] = (m["wert"] > 0).astype(float).where(m["wert"].notna())
        else:
            out["ni_pos"] = (m["wert"] > 0).astype(float).where(m["wert"].notna())
    return out
