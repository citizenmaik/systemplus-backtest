"""Datensatz aktualisieren.

  --mode incremental  Letzten Monat für alle Universum-Titel laden und anhängen
  --mode full         wie incremental, zusätzlich: Kandidatenliste neu holen, für alle
                      Kandidaten außerhalb des Universums die Historie ab START laden und
                      neue Titel aufnehmen, die die Grundfilter erfüllt haben

Gespeichert werden Rohkurse (data/ohlcv/<Aufnahmedatum>/YYYY-MM.parquet) und Splits
(data/splits.csv). Ein Split ändert deshalb nur splits.csv, nicht die Historie. Neue
Titel landen in einem eigenen Ordner, bestehende Monatsdateien bleiben unberührt."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from common import (AVGVOL_WIN, BENCHMARKS, COHORTS, DATA, MIN_AVGVOL, MIN_ELIGIBLE_DAYS, MIN_PRICE,
                    START, load_universe, read_csv, read_splits, read_store, save_universe,
                    splits_from_download, to_raw, upsert, write_splits, write_store)
from download import fetch

MISMATCH_TOL = 0.005  # Abweichung Rohkurs alt/neu, ab der die Historie neu geladen wird


def cohort_tickers() -> set[str]:
    if not COHORTS.exists():
        return set()
    c = read_csv(COHORTS, dtype=str)
    return set(c["ticker"].dropna().str.strip())


def eligible_days(raw: pd.DataFrame) -> pd.Series:
    """Tage mit Rohkurs > 5 und Ø-Volumen(30) >= 1 Mio. – mit den damals echten Werten."""
    raw = raw.sort_values(["ticker", "date"])
    avgvol = raw.groupby("ticker")["volume"].transform(
        lambda v: v.rolling(AVGVOL_WIN, min_periods=AVGVOL_WIN).mean())
    ok = (raw["close"] > MIN_PRICE) & (avgvol >= MIN_AVGVOL)
    return ok.groupby(raw["ticker"]).sum().astype(int)


def refresh(tickers: list[str], old: pd.DataFrame, splits: pd.DataFrame):
    """Letzten Monat laden und anhängen. Passen die Rohkurse der Überlappungstage nicht zu
    den gespeicherten (verpasster Split, Datenkorrektur bei Yahoo), wird die Historie
    dieses Titels komplett neu geladen."""
    adj, missing = fetch(tickers, period="1mo")
    splits = pd.concat([splits, splits_from_download(adj)], ignore_index=True)
    splits = splits.drop_duplicates(["ticker", "date"], keep="last")
    raw = to_raw(adj, splits)

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
    return store, splits, redo, missing


def incremental() -> dict:
    uni = load_universe()
    old = read_store()
    if uni.empty or old.empty:
        print("Noch kein Datensatz – starte Vollaufbau.")
        return full()
    tickers = sorted(set(uni["ticker"]) | set(BENCHMARKS))
    store, splits, redo, missing = refresh(tickers, old, read_splits())
    write_splits(splits)
    res = write_store(store)
    res.update(mode="incremental", refetched=redo, no_data=len(missing),
               last_date=str(store["date"].max().date()))
    return res


def full() -> dict:
    from symbols import BAD_NAME, fetch_candidates

    today = datetime.now(timezone.utc).date().isoformat()
    cand = fetch_candidates()
    uni = load_universe()
    old = read_store()
    splits = read_splits()
    cohort = cohort_tickers()
    print(f"Kandidaten: {len(cand)}, Universum bisher: {len(uni)}")

    # 1. Fonds, SPACs usw. entfernen (auch rückwirkend, falls der Namensfilter verschärft wurde)
    excluded = {t for t, n in zip(uni["ticker"], uni["name"].fillna(""))
                if BAD_NAME.search(n)} - cohort - set(BENCHMARKS)
    uni = uni[~uni["ticker"].isin(excluded)]
    old = old[~old["ticker"].isin(excluded)]
    splits = splits[~splits["ticker"].isin(excluded)]

    # 2. Bestehende Titel: nur anhängen
    have = set(uni["ticker"])
    store, splits, redo, missing = (refresh(sorted(have | set(BENCHMARKS)), old, splits)
                                    if have else (old, splits, [], []))

    # 3. Neue Kandidaten: volle Historie, Aufnahme nach Grundfiltern
    new_t = sorted((set(cand["ticker"]) | cohort | set(BENCHMARKS)) - have)
    added: list[str] = []
    elig = pd.Series(dtype=int)
    if new_t:
        print(f"Prüfe {len(new_t)} Titel außerhalb des Universums")
        adj, _ = fetch(new_t, start=START)
        new_splits = splits_from_download(adj)
        raw = to_raw(adj, new_splits)
        elig = eligible_days(raw)
        take = set(elig[elig >= MIN_ELIGIBLE_DAYS].index) | ((cohort | set(BENCHMARKS)) & set(raw["ticker"]))
        added = sorted(take)
        store = upsert(store, raw[raw["ticker"].isin(take)])
        splits = pd.concat([splits, new_splits[new_splits["ticker"].isin(take)]], ignore_index=True)

    # 4. Universum fortschreiben
    info = cand.set_index("ticker")
    uni = uni.set_index("ticker")
    for t in added:
        uni.loc[t, "added"] = today
        uni.loc[t, "eligible_days"] = int(elig.get(t, 0))
    listed = uni.index.intersection(info.index)
    uni.loc[listed, "name"] = info.loc[listed, "name"]
    uni.loc[listed, "exchange"] = info.loc[listed, "exchange"]
    uni.loc[listed, "last_listed"] = today
    bm = uni.index.intersection(BENCHMARKS)
    uni.loc[bm, ["name", "exchange", "last_listed"]] = [[t, "ETF", today] for t in bm]
    rest = uni.index.difference(listed).difference(bm).intersection(list(cohort))
    uni.loc[rest, "exchange"] = uni.loc[rest, "exchange"].fillna("Kohorte")
    uni.index.name = "ticker"
    save_universe(uni.reset_index())
    write_splits(splits)

    res = write_store(store)
    res.update(mode="full", candidates=len(cand), universe=len(uni), new_in_universe=len(added),
               excluded=len(excluded), refetched=redo, no_data=len(missing),
               splits=len(splits), last_date=str(store["date"].max().date()))
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
