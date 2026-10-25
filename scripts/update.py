"""Datensatz aktualisieren.

  --mode full         Kandidaten neu holen, komplette Historie ab START laden, Universum erweitern
  --mode incremental  Letzten Monat für das Universum laden und anhängen

Gespeichert werden Rohkurse (data/ohlcv) und Splits (data/splits.csv). Ein Split
ändert deshalb nur splits.csv, nicht die Historie."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from common import (AVGVOL_WIN, BENCHMARKS, COHORTS, DATA, MIN_AVGVOL, MIN_ELIGIBLE_DAYS, MIN_PRICE,
                    START, load_universe, read_splits, read_store, save_universe, splits_from_download,
                    to_raw, upsert, write_splits, write_store)
from download import fetch

MISMATCH_TOL = 0.005  # Abweichung Rohkurs alt/neu, ab der die Historie neu geladen wird


def cohort_tickers() -> set[str]:
    if not COHORTS.exists():
        return set()
    c = pd.read_csv(COHORTS, dtype=str, comment="#")
    return set(c["ticker"].dropna().str.strip())


def eligible_days(raw: pd.DataFrame) -> pd.Series:
    """Tage mit Rohkurs > 5 und Ø-Volumen(30) >= 1 Mio. – mit den damals echten Werten."""
    raw = raw.sort_values(["ticker", "date"])
    avgvol = raw.groupby("ticker")["volume"].transform(
        lambda v: v.rolling(AVGVOL_WIN, min_periods=AVGVOL_WIN).mean())
    ok = (raw["close"] > MIN_PRICE) & (avgvol >= MIN_AVGVOL)
    return ok.groupby(raw["ticker"]).sum().astype(int)


def full() -> dict:
    from symbols import fetch_candidates

    today = datetime.now(timezone.utc).date().isoformat()
    cand = fetch_candidates()
    uni = load_universe()
    old = read_store()
    old_splits = read_splits()
    print(f"Kandidaten: {len(cand)}, Universum bisher: {len(uni)}")

    cohort = cohort_tickers()
    tickers = sorted(set(cand["ticker"]) | set(uni["ticker"]) | set(BENCHMARKS) | cohort)
    adj, missing = fetch(tickers, start=START)
    got = set(adj["ticker"])

    # Splits: für neu geladene Ticker die vollständige Yahoo-Liste, sonst die gespeicherte
    splits = pd.concat([old_splits[~old_splits["ticker"].isin(got)], splits_from_download(adj)],
                       ignore_index=True)
    raw = to_raw(adj, splits)

    elig = eligible_days(raw)
    # Kohorten-Titel immer behalten, damit der Vergleich handverlesen/zufall vollständig ist
    keep = (set(elig[elig >= MIN_ELIGIBLE_DAYS].index) | set(uni["ticker"]) | set(BENCHMARKS)
            | (cohort & set(raw["ticker"])))
    raw = raw[raw["ticker"].isin(keep)]
    splits = splits[splits["ticker"].isin(keep)]
    # Ticker ohne neue Daten (z. B. delistet) behalten ihre alte Historie
    store = upsert(old, raw, replace_tickers=set(raw["ticker"]))

    info = cand.set_index("ticker")
    uni = uni.set_index("ticker")
    added = sorted(keep - set(uni.index))
    for t in sorted(keep):
        if t not in uni.index:
            uni.loc[t, "added"] = today
        if t in info.index:
            uni.loc[t, "name"] = info.at[t, "name"]
            uni.loc[t, "exchange"] = info.at[t, "exchange"]
            uni.loc[t, "last_listed"] = today
        elif t in BENCHMARKS:
            uni.loc[t, ["name", "exchange", "last_listed"]] = [t, "ETF", today]
        if t in elig.index:
            uni.loc[t, "eligible_days"] = int(elig[t])
    uni.index.name = "ticker"
    save_universe(uni.reset_index())
    write_splits(splits)

    res = write_store(store)
    res.update(mode="full", candidates=len(cand), universe=len(uni), new_in_universe=len(added),
               no_data=len(missing), splits=len(splits))
    return res


def incremental() -> dict:
    uni = load_universe()
    old = read_store()
    if uni.empty or old.empty:
        print("Noch kein Datensatz – starte Vollaufbau.")
        return full()

    tickers = sorted(set(uni["ticker"]) | set(BENCHMARKS))
    adj, missing = fetch(tickers, period="1mo")
    splits = pd.concat([read_splits(), splits_from_download(adj)], ignore_index=True)
    splits = splits.drop_duplicates(["ticker", "date"], keep="last")
    raw = to_raw(adj, splits)

    # Kontrolle: Rohkurse der Überlappungstage müssen zu den gespeicherten passen.
    # Wenn nicht (verpasster Split, Datenkorrektur bei Yahoo), Historie komplett neu laden.
    m = raw.merge(old[["date", "ticker", "close"]], on=["date", "ticker"], suffixes=("", "_old"))
    dev = (m["close"] / m["close_old"] - 1).abs()
    redo = sorted(m.loc[dev > MISMATCH_TOL, "ticker"].unique())
    replace: set[str] = set()
    if redo:
        print(f"Historie neu laden (Abweichung > {MISMATCH_TOL:.1%}): {', '.join(redo)}")
        hist_adj, _ = fetch(redo, start=START, chunk=50)
        replace = set(hist_adj["ticker"])
        splits = pd.concat([splits[~splits["ticker"].isin(replace)], splits_from_download(hist_adj)],
                           ignore_index=True)
        raw = pd.concat([raw[~raw["ticker"].isin(replace)], to_raw(hist_adj, splits)], ignore_index=True)

    store = upsert(old, raw, replace_tickers=replace)
    write_splits(splits)
    res = write_store(store)
    res.update(mode="incremental", refetched=redo, no_data=len(missing),
               last_date=str(store["date"].max().date()))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    args = ap.parse_args()
    result = full() if args.mode == "full" else incremental()
    result["run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    DATA.mkdir(exist_ok=True)
    (DATA / "last_run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
