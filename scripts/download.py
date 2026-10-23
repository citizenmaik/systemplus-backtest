"""Tageskerzen per yfinance in Blöcken laden, mit Wiederholung und Fehlerliste.

Yahoo liefert split-bereinigte, nicht dividendenbereinigte Kurse (auto_adjust=False,
Spalte Close) plus die Splits (actions=True). update.py rechnet daraus Rohkurse."""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"}


def to_long(raw: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """yfinance-Breitformat -> Langformat date,ticker,open,high,low,close,volume,stock_splits."""
    cols = ["date", "ticker", "open", "high", "low", "close", "volume", "stock_splits"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)
    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({requested[0]: raw}, axis=1)
    if set(map(str, raw.columns.get_level_values(0))) & FIELDS:
        raw = raw.swaplevel(0, 1, axis=1)  # (Feld, Ticker) -> (Ticker, Feld)
    raw = raw.sort_index(axis=1)
    try:
        long = raw.stack(level=0, future_stack=True)
    except TypeError:  # ältere pandas-Versionen
        long = raw.stack(level=0, dropna=False)
    long.index = long.index.set_names(["date", "ticker"])
    long = long.reset_index()
    long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
    for c in cols:
        if c not in long.columns:
            long[c] = pd.NA
    long = long[cols].dropna(subset=["close"])
    long["stock_splits"] = pd.to_numeric(long["stock_splits"], errors="coerce").fillna(0.0)
    d = pd.to_datetime(long["date"])
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    long["date"] = d.dt.normalize()
    long["ticker"] = long["ticker"].astype(str)
    return long.reset_index(drop=True)


def _download(tickers: list[str], start: str | None, period: str | None, tries: int = 4) -> pd.DataFrame:
    kw = dict(interval="1d", auto_adjust=False, actions=True, group_by="ticker",
              threads=True, progress=False)
    kw.update({"start": start} if start else {"period": period or "1mo"})
    for attempt in range(1, tries + 1):
        try:
            return yf.download(tickers=tickers, **kw)
        except Exception as e:  # Rate-Limit, Netz
            wait = 30 * attempt
            print(f"  Fehler ({type(e).__name__}: {e}) – neuer Versuch in {wait}s")
            time.sleep(wait)
    return pd.DataFrame()


def fetch(tickers: list[str], *, start: str | None = None, period: str | None = None,
          chunk: int = 100, pause: float = 2.0) -> tuple[pd.DataFrame, list[str]]:
    """Gibt (Langformat-Daten, Ticker ohne Daten) zurück."""
    tickers = sorted(set(tickers))
    frames, missing = [], []

    def run(batch_list: list[str], size: int) -> None:
        for i in range(0, len(batch_list), size):
            part = batch_list[i:i + size]
            long = to_long(_download(part, start, period), part)
            got = set(long["ticker"]) if len(long) else set()
            frames.append(long)
            missing.extend(t for t in part if t not in got)
            print(f"  {min(i + size, len(batch_list))}/{len(batch_list)} geladen, {len(got)} mit Daten")
            time.sleep(pause)

    run(tickers, chunk)
    if missing:  # zweiter Durchgang in kleinen Blöcken
        retry, missing[:] = list(missing), []
        print(f"  Zweiter Versuch für {len(retry)} Ticker")
        run(retry, 20)
    data = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) \
        else to_long(pd.DataFrame(), [])
    return data, sorted(set(missing))
