"""System+ v3.8 – Signale (1:1 zum Pine-Indikator „Darvas Lines/Box + Trade Mgmt“, Profil System+).

Pro Ticker und Tag:
  Pivot-Box   Oberkante = letztes bestätigtes Swing-High, Unterkante = letztes Swing-Low
              (je 3 Kerzen links/rechts, Dochte), Höhe ≤ 3 × ATR(14) vom Vortag.
              Breakout = erster Close > Oberkante (Vortag ≤ Oberkante); eine Oberkante löst
              nur einmal aus (verbraucht, sobald Close > Oberkante UND Close > SMA10).
  Gates       ATR-Multiple (Close − SMA10) / ATR ≤ 2,5 · SL-Abstand ≤ 10 % ·
              Close 10–30 % unter höchstem Hoch 252 · Notbremse QQQE aus ·
              Führung: RS-Rating ≥ 80 oder Sektor Platz 1
  Screener    Kurs > 5 · Market Cap > 500 Mio. · Ø Vol 30 T > 1 Mio. · ATR14 2,5–10 % ·
              Float > 70 % (Branchenauswahl nicht nachgebildet)

Abweichungen vom Indikator (Daten bedingt):
  * RS-Rating = Perzentil des Fred6724-Scores über alle Titel im Datensatz am selben Tag
    (Indikator: 7 feste Stützwerte). Referenz SPY statt SPX – kürzt sich im Perzentil heraus.
  * Market Cap = bereinigter Kurs × heutige Aktienzahl, Float = heutiger Stand.
  * Sektor: TradingView-Sektor aus universe/tv_sectors.csv, sonst Yahoo-Sektor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import read_csv  # noqa: E402
from load import load  # noqa: E402

SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLU"]
NON_STOCK = set(SECTOR_ETFS) | {"SPY", "QQQ", "QQQE", "RSP", "IWM"}

TV_MAP = {
    "Electronic technology": "XLK", "Technology services": "XLK", "Communications": "XLC",
    "Consumer durables": "XLY", "Consumer services": "XLY", "Retail trade": "XLY",
    "Consumer non-durables": "XLP", "Energy minerals": "XLE", "Finance": "XLF",
    "Health services": "XLV", "Health technology": "XLV", "Industrial services": "XLI",
    "Distribution services": "XLI", "Commercial services": "XLI", "Producer manufacturing": "XLI",
    "Transportation": "XLI", "Non-energy minerals": "XLB", "Process industries": "XLB",
    "Utilities": "XLU",
}
YAHOO_MAP = {
    "Technology": "XLK", "Communication Services": "XLC", "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP", "Energy": "XLE", "Financial Services": "XLF",
    "Real Estate": "XLF",  # TradingView führt REITs unter Finance
    "Healthcare": "XLV", "Industrials": "XLI", "Basic Materials": "XLB", "Utilities": "XLU",
}

P = dict(piv=3, atr_len=14, box_atr=3.0, fast=10, slow=20, atr_mult=2.5, sl_pct=10.0,
         w52=(10.0, 30.0), rs_min=80, sec_max=1,
         price=5.0, mcap=500e6, avgvol=1e6, atr_band=(2.5, 10.0), float_min=70.0, w52_screen=30.0)


# ---------------------------------------------------------------------------
def _rma(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ta.rma: Start mit SMA der ersten n Werte, dann alpha = 1/n."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.nanmean(x[:n])
    a = 1.0 / n
    for i in range(n, len(x)):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def _pivots(h: np.ndarray, l: np.ndarray, k: int):
    """Pine ta.pivothigh/-low: Wert an Bar t, wenn Bar t−k höher (tiefer) ist als k Bars links
    (strikt) und k Bars rechts (≥ bzw. ≤ zählt nicht als höher/tiefer)."""
    n = len(h)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    if n < 2 * k + 1:
        return ph, pl
    from numpy.lib.stride_tricks import sliding_window_view as sw
    wh, wl = sw(h, 2 * k + 1), sw(l, 2 * k + 1)          # Fenster endet bei t
    ch, cl = wh[:, k], wl[:, k]
    okh = (ch > wh[:, :k].max(axis=1)) & (ch >= wh[:, k + 1:].max(axis=1))
    okl = (cl < wl[:, :k].min(axis=1)) & (cl <= wl[:, k + 1:].min(axis=1))
    ph[2 * k:] = np.where(okh, ch, np.nan)
    pl[2 * k:] = np.where(okl, cl, np.nan)
    return ph, pl


def per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (g[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(c)
    k = P["piv"]
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr[0] = h[0] - l[0]
    atr = _rma(tr, P["atr_len"])
    sma10 = pd.Series(c).rolling(P["fast"]).mean().to_numpy()
    sma20 = pd.Series(c).rolling(P["slow"]).mean().to_numpy()
    hi252 = pd.Series(h).rolling(252).max().to_numpy()

    ph, pl = _pivots(h, l, k)
    idx = np.arange(n)
    top = pd.Series(ph).ffill().to_numpy()
    top_bar = pd.Series(np.where(~np.isnan(ph), idx - k, np.nan)).ffill().to_numpy()
    bot = pd.Series(pl).ffill().to_numpy()
    atr_prev = np.r_[np.nan, atr[:-1]]
    height = (top - bot) / atr_prev
    has_box = ~np.isnan(top) & ~np.isnan(bot) & (bot < top)
    cross = (c > top) & (prev_c <= top)
    cand = has_box & (height <= P["box_atr"]) & cross            # pivBreak ohne „verbraucht“
    consume = cand & (c > sma10)                                  # verbraucht die Oberkante
    # je Oberkante (top_bar) zählt nur der erste Breakout mit Close > SMA10
    seg = pd.Series(top_bar)
    first = consume & (pd.Series(consume.astype(int)).groupby(seg).cumsum().to_numpy() == 1)

    out = pd.DataFrame({
        "date": g["date"].to_numpy(), "ticker": g["ticker"].iloc[0],
        "open": o, "high": h, "low": l, "close": c, "sma10": sma10, "sma20": sma20,
        "atr": atr, "atr_pct": 100 * atr / c, "box_top": top, "box_bot": bot, "box_h_atr": height,
        "off52": 100 * (c / hi252 - 1), "entry_sig": first,
        "raw_close": g["raw_close"].to_numpy(), "avgvol30": g["avgvol30"].to_numpy(),
    })
    extra = {"atr_mult": (c - sma10) / atr, "sl_pct": 100 * (c - sma10) / c}
    # RS-Score Fred6724 (Lookback = min(bar_index, n) wie im Indikator)
    for nn in (63, 126, 189, 252):
        j = np.maximum(idx - np.minimum(idx, nn), 0)
        extra[f"r{nn}"] = c / c[j]
    return pd.concat([out, pd.DataFrame(extra)], axis=1)


def notbremse(q: pd.DataFrame) -> pd.Series:
    c = q.set_index("date")["close"]
    ema200 = c.ewm(span=200, adjust=False).mean()
    dd = 100 * (c / c.rolling(60).max() - 1)
    s200 = c.rolling(200).mean()
    return (c < ema200) | (dd <= -10) | (s200 < s200.shift(20))


def sector_rank(px: pd.DataFrame) -> pd.DataFrame:
    """Rang 1–10 je SPDR-Sektor (1 = stärkster), wie im Indikator."""
    c = px.pivot(index="date", columns="ticker", values="close")

    def w(s):
        return sum(wt * (s / s.shift(n) - 1) * 100 for wt, n in
                   [(.05, 1), (.15, 5), (.20, 10), (.35, 20), (.25, 63)])
    wspy = w(c["SPY"])
    sc = pd.DataFrame({e: (50 + 4.5 * w(c[e]).sub(wspy)).clip(0, 100).round()
                       for e in SECTOR_ETFS if e in c})
    rank = sc.rank(axis=1, ascending=False, method="min")
    return rank


def sector_of(tickers) -> pd.Series:
    tv = read_csv(ROOT / "universe" / "tv_sectors.csv").set_index("ticker")["tv_sector"].map(TV_MAP)
    meta = read_csv(ROOT / "data" / "meta.csv").set_index("ticker")["sector"].map(YAHOO_MAP)
    s = pd.Series(index=pd.Index(sorted(set(tickers)), name="ticker"), dtype=object)
    s.update(meta)
    s.update(tv)  # TradingView hat Vorrang
    return s


def build(start: str = "2019-01-01", verbose: bool = True) -> pd.DataFrame:
    """Alle Bars mit Indikatoren und Gate-Spalten (ab start, Indikatoren mit voller Historie)."""
    df = load()
    if verbose:
        print(f"geladen: {df['ticker'].nunique()} Ticker, {len(df):,} Zeilen")
    bm = df[df["ticker"].isin(NON_STOCK)][["date", "ticker", "close"]]
    stocks = df[~df["ticker"].isin(NON_STOCK)]

    parts = [per_ticker(g) for _, g in stocks.groupby("ticker", sort=False)]
    x = pd.concat(parts, ignore_index=True)
    if verbose:
        print("Indikatoren fertig")

    # RS-Rating: Perzentil des Scores über alle Titel des Tages
    spy = bm[bm["ticker"] == "SPY"].set_index("date")["close"].sort_index()
    sidx = np.arange(len(spy))
    ref = {}
    for nn in (63, 126, 189, 252):
        ref[nn] = pd.Series(spy.to_numpy() / spy.to_numpy()[np.maximum(sidx - nn, 0)], index=spy.index)
    rs_ref = 0.4 * ref[63] + 0.2 * ref[126] + 0.2 * ref[189] + 0.2 * ref[252]
    x["rs_score"] = (0.4 * x["r63"] + 0.2 * x["r126"] + 0.2 * x["r189"] + 0.2 * x["r252"]) \
        / x["date"].map(rs_ref) * 100
    x["rs_rating"] = (x.groupby("date")["rs_score"].rank(pct=True) * 100).clip(1, 99)
    x = x.drop(columns=["r63", "r126", "r189", "r252"])

    # Sektor-Rang
    sec = sector_of(x["ticker"].unique())
    x["sector_etf"] = x["ticker"].map(sec)
    rank = sector_rank(bm).stack().rename("sector_rank").reset_index()
    rank.columns = ["date", "sector_etf", "sector_rank"]
    x = x.merge(rank, on=["date", "sector_etf"], how="left")

    # Notbremse
    cb = notbremse(bm[bm["ticker"] == "QQQE"])
    x["notbremse"] = x["date"].map(cb).fillna(True).astype(bool)

    # Metadaten (heutiger Stand) für Market Cap und Float
    meta = read_csv(ROOT / "data" / "meta.csv").set_index("ticker")
    x["mcap"] = x["close"] * x["ticker"].map(meta["sharesOutstanding"])
    flt = 100 * meta["floatShares"] / meta["sharesOutstanding"]
    x["float_pct"] = x["ticker"].map(flt)
    coh = read_csv(ROOT / "universe" / "cohorts.csv", dtype=str).set_index("ticker")["cohort"]
    x["cohort"] = x["ticker"].map(coh).fillna("neu")

    x = x[x["date"] >= pd.Timestamp(start)].reset_index(drop=True)

    # Gates
    lo, hi = P["w52"]
    x["g_screen"] = ((x["raw_close"] > P["price"]) & (x["avgvol30"] > P["avgvol"])
                     & x["atr_pct"].between(*P["atr_band"])
                     & (x["mcap"].isna() | (x["mcap"] > P["mcap"]))
                     & (x["float_pct"].isna() | (x["float_pct"] > P["float_min"]))
                     & (x["off52"] >= -P["w52_screen"]))
    x["g_atr"] = x["atr_mult"] <= P["atr_mult"]
    x["g_sl"] = x["sl_pct"] <= P["sl_pct"]
    x["g_52w"] = (x["off52"] <= -lo) & (x["off52"] >= -hi)
    x["g_rs"] = x["rs_rating"].round() >= P["rs_min"]
    x["g_sec"] = x["sector_rank"] <= P["sec_max"]
    x["g_lead"] = x["g_rs"] | x["g_sec"]
    x["g_cb"] = ~x["notbremse"]
    return x


if __name__ == "__main__":
    x = build()
    s = x[x["entry_sig"]]
    print(f"Breakouts mit Close > SMA10: {len(s):,}")
    for col in ["g_screen", "g_atr", "g_sl", "g_52w", "g_cb", "g_lead"]:
        print(f"  {col}: {s[col].mean():.1%}")
    full = s[s[["g_screen", "g_atr", "g_sl", "g_52w", "g_cb", "g_lead"]].all(axis=1)]
    print(f"alle Gates: {len(full):,}")
    print(full.groupby("cohort").size())
