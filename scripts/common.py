"""Gemeinsame Pfade, Parameter und der Parquet-Speicher für systemplus-backtest."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OHLCV = DATA / "ohlcv"
UNIVERSE = ROOT / "universe" / "universe.csv"
COHORTS = ROOT / "universe" / "cohorts.csv"
META = DATA / "meta.csv"
QUALITY = DATA / "quality.csv"
SPLITS = DATA / "splits.csv"

# Historie ab 2018, damit EMA/SMA200 ab Anfang 2019 eingeschwungen sind
START = "2018-01-01"

# Immer mitladen (ETFs fallen sonst aus der Kandidatenliste)
BENCHMARKS = ["SPY", "QQQ", "QQQE", "RSP", "IWM"]

# Aufnahme ins Universum: an mindestens MIN_ELIGIBLE_DAYS Tagen seit START
# Kurs > MIN_PRICE und Ø-Volumen(30) >= MIN_AVGVOL. Ein Titel, der einmal
# drin ist, bleibt drin – auch wenn er später delistet wird.
MIN_PRICE = 5.0
MIN_AVGVOL = 1_000_000
AVGVOL_WIN = 30
MIN_ELIGIBLE_DAYS = 20

COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]
UNI_COLS = ["ticker", "name", "exchange", "added", "last_listed", "eligible_days"]


def read_csv(path, **kw) -> pd.DataFrame:
    """CSV lesen, ohne dass Ticker wie NA, NAN oder NULL zu fehlenden Werten werden."""
    kw.setdefault("dtype", {"ticker": str})
    return pd.read_csv(path, keep_default_na=False, na_values=[""], comment="#", **kw)


def empty_store() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in COLS})
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df["volume"] = df["volume"].astype("int64")
    return df


def read_store() -> pd.DataFrame:
    files = sorted(OHLCV.glob("**/*.parquet"))
    if not files:
        return empty_store()
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df[COLS]


def batch_map() -> dict[str, str]:
    """Ticker -> Speicherordner (= Aufnahmedatum ins Universum). Neue Titel landen so in
    einem eigenen kleinen Ordner, statt alle bestehenden Monatsdateien zu verändern."""
    uni = load_universe()
    return dict(zip(uni["ticker"], uni["added"].fillna("unbekannt").astype(str)))


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Typen, Rundung und Sortierung festlegen, damit gleiche Daten byte-gleiche Dateien ergeben."""
    if df.empty:
        return empty_store()
    df = df[COLS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype("float64").round(4)
    df["volume"] = df["volume"].fillna(0).round().astype("int64")
    df = df.dropna(subset=["close"])
    df = df.drop_duplicates(["ticker", "date"], keep="last")
    return df.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)


def write_store(df: pd.DataFrame, batches: dict[str, str] | None = None) -> dict:
    """Schreibt data/ohlcv/<Aufnahmedatum>/YYYY-MM.parquet. Unveränderte Dateien werden
    nicht angefasst, verwaiste gelöscht – so bleibt die Git-Historie klein."""
    OHLCV.mkdir(parents=True, exist_ok=True)
    df = normalize(df)
    batches = batches if batches is not None else batch_map()
    batch = df["ticker"].map(batches).fillna("unbekannt")
    months = df["date"].dt.strftime("%Y-%m")
    written, unchanged, produced = 0, 0, set()
    for (b, month), part in df.groupby([batch, months], sort=True):
        table = pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False)
        table = table.replace_schema_metadata(None)  # keine Versions-Metadaten -> stabile Bytes
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="zstd", compression_level=9)
        data = buf.getvalue()
        path = OHLCV / b / f"{month}.parquet"
        produced.add(path)
        if path.exists() and path.read_bytes() == data:
            unchanged += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        written += 1
    removed = 0
    for f in OHLCV.glob("**/*.parquet"):
        if f not in produced:
            f.unlink()
            removed += 1
    for d in sorted(OHLCV.glob("*/"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    return {"files_written": written, "files_unchanged": unchanged, "files_removed": removed,
            "rows": len(df), "tickers": int(df["ticker"].nunique())}


def upsert(old: pd.DataFrame, new: pd.DataFrame, replace_tickers: set[str] | None = None) -> pd.DataFrame:
    """Neue Zeilen gewinnen. Für replace_tickers wird die alte Historie komplett ersetzt
    (nötig nach Splits, weil sich dann die ganze Vergangenheit ändert)."""
    if replace_tickers:
        old = old[~old["ticker"].isin(replace_tickers)]
    both = pd.concat([old, new], ignore_index=True)
    return normalize(both)


# ---------------------------------------------------------------------------
# Splits. Gespeichert werden ROHKURSE (wie damals gehandelt). Bereinigt wird erst
# beim Laden. So ändert ein Split nie die gespeicherte Historie.

def read_splits() -> pd.DataFrame:
    if SPLITS.exists():
        s = read_csv(SPLITS, parse_dates=["date"])
        return s[["ticker", "date", "ratio"]]
    return pd.DataFrame({"ticker": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                         "ratio": pd.Series(dtype="float64")})


def write_splits(s: pd.DataFrame) -> None:
    s = s[s["ratio"] > 0].copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    s["ratio"] = s["ratio"].astype(float).round(6)
    s = s.drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"])
    SPLITS.parent.mkdir(parents=True, exist_ok=True)
    s.assign(date=s["date"].dt.strftime("%Y-%m-%d")).to_csv(SPLITS, index=False)


def splits_from_download(df: pd.DataFrame) -> pd.DataFrame:
    if "stock_splits" not in df.columns:
        return read_splits().iloc[0:0]
    s = df.loc[df["stock_splits"].fillna(0) > 0, ["ticker", "date", "stock_splits"]]
    return s.rename(columns={"stock_splits": "ratio"}).reset_index(drop=True)


def future_factor(df: pd.DataFrame, splits: pd.DataFrame) -> np.ndarray:
    """Produkt aller Split-Verhältnisse NACH dem jeweiligen Datum (1.0 ohne spätere Splits)."""
    f = np.ones(len(df))
    if splits.empty or df.empty:
        return f
    pos = df.groupby("ticker", sort=False).indices
    dates = df["date"].values
    for t, s in splits.groupby("ticker"):
        if t not in pos:
            continue
        idx = pos[t]
        d = dates[idx]
        for sd, r in zip(s["date"].values, s["ratio"].values):
            f[idx[d < sd]] *= r
    return f


def to_raw(adj: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Split-bereinigte Yahoo-Kurse -> Rohkurse."""
    out = adj.reset_index(drop=True).copy()
    f = future_factor(out, splits)
    for c in ("open", "high", "low", "close"):
        out[c] = out[c] * f
    out["volume"] = out["volume"] / f
    return out[COLS]


def to_adjusted(raw: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Rohkurse -> split-bereinigt (für Indikatoren und Renditen)."""
    out = raw.reset_index(drop=True).copy()
    f = future_factor(out, splits)
    for c in ("open", "high", "low", "close"):
        out[c] = out[c] / f
    out["volume"] = out["volume"] * f
    return out


def load_universe() -> pd.DataFrame:
    if UNIVERSE.exists():
        uni = read_csv(UNIVERSE)
        for c in UNI_COLS:
            if c not in uni.columns:
                uni[c] = pd.NA
        return uni[UNI_COLS]
    return pd.DataFrame(columns=UNI_COLS)


def save_universe(uni: pd.DataFrame) -> None:
    UNIVERSE.parent.mkdir(parents=True, exist_ok=True)
    uni.sort_values("ticker").to_csv(UNIVERSE, index=False)
