"""Merkmale am Entry-Tag (point-in-time, nur Daten bis einschließlich Entry-Close).

build_features(x) ergänzt die Signal-Tabelle aus signals.build() um:
  Momentum   perf_1m, perf_3m, perf_6m, perf_12m (%), rs_rating
  Trend      dist_sma50, dist_sma200 (%), sma200_slope (% über 20 T), above_sma200
  Volumen    vol_ratio (Breakout-Volumen ÷ Ø 50 T), dollar_vol (Ø 30 T, Mio. USD)
  Kerze      close_loc (0 = Tief, 1 = Hoch), gap_pct, day_ret, brk_atr ((Close − Oberkante)/ATR)
  Box        box_h_atr, box_age (Bars seit Swing-High), box_depth ((Ober − Unterkante)/Oberkante %)
  Risiko     atr_pct, atr_mult, sl_pct, off52
  Größe      raw_close, mcap (Mio.), bar_no (Handelstage seit Datenbeginn)
  Markt      qqqe_dist_ema200, qqqe_dd60, sector_rank
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals import NON_STOCK, load


def build_features(x: pd.DataFrame) -> pd.DataFrame:
    df = load()
    df = df[~df["ticker"].isin(NON_STOCK - {"QQQE"})].sort_values(["ticker", "date"])
    g = df.groupby("ticker", sort=False)
    c = df["close"]
    f = pd.DataFrame({"date": df["date"], "ticker": df["ticker"]})
    for n, nm in [(21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")]:
        f[f"perf_{nm}"] = 100 * (c / g["close"].shift(n) - 1)
    sma50 = g["close"].transform(lambda s: s.rolling(50).mean())
    sma200 = g["close"].transform(lambda s: s.rolling(200).mean())
    f["dist_sma50"] = 100 * (c / sma50 - 1)
    f["dist_sma200"] = 100 * (c / sma200 - 1)
    f["sma200_slope"] = 100 * (sma200 / sma200.groupby(df["ticker"]).shift(20) - 1)
    vol50 = g["volume"].transform(lambda s: s.shift(1).rolling(50).mean())
    f["vol_ratio"] = df["volume"] / vol50
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    f["close_loc"] = (c - df["low"]) / rng
    prev = g["close"].shift(1)
    f["gap_pct"] = 100 * (df["open"] / prev - 1)
    f["day_ret"] = 100 * (c / prev - 1)
    f["bar_no"] = df["bar_no"]
    f["dollar_vol"] = df["dollarvol30"] / 1e6

    q = f[f["ticker"] == "QQQE"].set_index("date")
    qc = df[df["ticker"] == "QQQE"].set_index("date")["close"]
    qm = pd.DataFrame({"qqqe_dist_ema200": 100 * (qc / qc.ewm(span=200, adjust=False).mean() - 1),
                       "qqqe_dd60": 100 * (qc / qc.rolling(60).max() - 1)})
    del q
    f = f[f["ticker"] != "QQQE"]

    out = x.merge(f, on=["date", "ticker"], how="left")
    out = out.join(qm, on="date")
    out["brk_atr"] = (out["close"] - out["box_top"]) / out["atr"]
    out["box_depth"] = 100 * (out["box_top"] - out["box_bot"]) / out["box_top"]
    out["above_sma200"] = out["dist_sma200"] > 0
    out["mcap_m"] = out["mcap"] / 1e6
    return out


def box_age(x: pd.DataFrame, piv: int = 3) -> pd.Series:
    """Bars seit dem Swing-High, das die aktuelle Oberkante bildet."""
    def per(g):
        h = g["high"].to_numpy()
        top = g["box_top"].to_numpy()
        n = len(h)
        age = np.full(n, np.nan)
        last_change = np.nan
        for i in range(n):
            if i > 0 and top[i] != top[i - 1] and not np.isnan(top[i]):
                last_change = i - piv
            age[i] = i - last_change
        return pd.Series(age, index=g.index)
    return x.groupby("ticker", sort=False, group_keys=False).apply(per)
