"""Zusätzliche Merkmale für die Hypothesen 5–7 (Stand 26.10.2026), alle point-in-time.

H5 Chart/Trend
  tt_ok         Trend-Template: Close > SMA50 > SMA150 > SMA200, SMA200 steigt (20 T), Close ≥ 1,3 × 52W-Tief
  prior_bases   Anzahl früherer Pivot-Breakouts (Close > SMA10) in den letzten 126 Tagen
  higher_low    aktuelle Box-Unterkante > vorherige Box-Unterkante (steigende Tiefs)
  adv_6m        Anstieg vom 6-Monats-Tief bis zum Close in %
H6 Führung
  ind_rank      Perzentil (0–100) der Branche nach gleichgewichteter 63-T-Rendite (Yahoo-Branche, ≥ 5 Titel)
  rsl_high      RS-Linie (Close ÷ SPY) auf 63-Tage-Hoch
H3/H4 Näherung für Zahlen/Prognosen
  egap_63       in den letzten 63 Tagen ein Gap ≥ 5 % mit Volumen ≥ 2,5 × Ø50 und Close über dem Vortagesclose
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import read_csv  # noqa: E402
from signals import NON_STOCK, load  # noqa: E402


def build(x: pd.DataFrame) -> pd.DataFrame:
    df = load()
    spy = df[df["ticker"] == "SPY"].set_index("date")["close"]
    df = df[~df["ticker"].isin(NON_STOCK)].sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker", sort=False)
    c = df["close"]
    roll = lambda s, n, f: g[s].transform(lambda v: getattr(v.rolling(n), f)())  # noqa: E731
    sma50, sma150, sma200 = (g["close"].transform(lambda v, n=n: v.rolling(n).mean()) for n in (50, 150, 200))
    low252 = roll("low", 252, "min")
    s200_prev = sma200.groupby(df["ticker"]).shift(20)
    out = pd.DataFrame({"date": df["date"], "ticker": df["ticker"]})
    out["tt_ok"] = (c > sma50) & (sma50 > sma150) & (sma150 > sma200) & (sma200 > s200_prev) & (c >= 1.3 * low252)
    out["adv_6m"] = 100 * (c / roll("low", 126, "min") - 1)
    rsl = c / df["date"].map(spy)
    out["rsl_high"] = rsl >= rsl.groupby(df["ticker"]).transform(lambda v: v.rolling(63).max()) * 0.999

    prev = g["close"].shift(1)
    vol50 = g["volume"].transform(lambda v: v.shift(1).rolling(50).mean())
    gap = (df["open"] / prev - 1 >= 0.05) & (df["volume"] >= 2.5 * vol50) & (c > prev)
    out["egap_63"] = gap.astype(int).groupby(df["ticker"]).transform(lambda v: v.rolling(63, min_periods=1).max()).astype(bool)

    # Branchen-Rang: gleichgewichtete 63-T-Rendite je Yahoo-Branche
    meta = read_csv(ROOT / "data" / "meta.csv").set_index("ticker")
    df["industry"] = df["ticker"].map(meta["industry"])
    df["r63"] = c / g["close"].shift(63) - 1
    ind = df.dropna(subset=["industry", "r63"]).groupby(["date", "industry"])["r63"].agg(["mean", "size"])
    ind = ind[ind["size"] >= 5]
    ind["ind_rank"] = ind.groupby(level="date")["mean"].rank(pct=True) * 100
    out = out.join(df[["industry"]])
    out = out.merge(ind["ind_rank"].reset_index(), on=["date", "industry"], how="left")

    # frühere Basen und steigende Tiefs aus der Signaltabelle
    xs = x[["date", "ticker", "entry_sig", "box_bot"]].sort_values(["ticker", "date"]).copy()
    gx = xs.groupby("ticker", sort=False)
    xs["prior_bases"] = gx["entry_sig"].transform(lambda v: v.astype(int).shift(1).rolling(126, min_periods=1).sum())

    def prev_distinct(v: pd.Series) -> pd.Series:
        a = v.to_numpy()
        out_ = np.full(len(a), np.nan)
        last, before = np.nan, np.nan
        for i, val in enumerate(a):
            if not np.isnan(val) and val != last:
                before, last = last, val
            out_[i] = before
        return pd.Series(out_, index=v.index)
    xs["bot_prev"] = gx["box_bot"].transform(prev_distinct)
    xs["higher_low"] = xs["box_bot"] > xs["bot_prev"]
    out = out.merge(xs[["date", "ticker", "prior_bases", "higher_low"]], on=["date", "ticker"], how="left")
    return out[["date", "ticker", "tt_ok", "adv_6m", "rsl_high", "egap_63", "ind_rank", "prior_bases", "higher_low"]]


def pullback_entries(x: pd.DataFrame, sig: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """Kauf bei Schwäche: nach dem Breakout-Signal ein Limit auf Höhe der Box-Oberkante,
    gültig `days` Tage. Fill zu min(Open, Oberkante), Stop = SMA10 am Fill-Tag (muss < Fill sein).
    Gibt je Signal den Fill-Index (in x) und den Fill-Kurs zurück, sonst NaN."""
    res = []
    pos = {t: g.index.to_numpy() for t, g in x.groupby("ticker", sort=False)}
    for t, gs in sig.groupby("ticker", sort=False):
        ix = pos[t]
        sub = x.loc[ix]
        o, lo, s10 = sub["open"].to_numpy(), sub["low"].to_numpy(), sub["sma10"].to_numpy()
        loc = {v: n for n, v in enumerate(ix)}
        for r in gs.itertuples():
            i0, lvl = loc[r.Index], r.box_top
            fill = np.nan
            for j in range(i0 + 1, min(i0 + 1 + days, len(ix))):
                if lo[j] <= lvl:
                    px = min(o[j], lvl)
                    if px > s10[j]:
                        fill = (ix[j], px)
                    break
            res.append((r.Index, *(fill if isinstance(fill, tuple) else (np.nan, np.nan))))
    return pd.DataFrame(res, columns=["sig_idx", "fill_idx", "fill_px"]).set_index("sig_idx")
