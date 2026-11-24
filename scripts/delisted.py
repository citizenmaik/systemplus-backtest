"""Delistete US-Aktien von einem Datenanbieter nachladen (Survivorship-Lücke vor dem ersten Lauf).

yfinance liefert für delistete Titel keine Kurse mehr. Was zwischen START (2018) und dem ersten
Lauf dieses Repos delistet wurde, fehlt deshalb im Datensatz. Dieses Skript lädt diese Titel
einmalig von Sharadar (sharadar.com, Plan "Prices", 10 Jahre) oder EODHD:

  python delisted.py --provider sharadar            # API-Key in SHARADAR_API_KEY
  python delisted.py --provider sharadar --sample   # Probelauf mit Sharadars Gratis-Schlüssel
  python delisted.py --provider eodhd               # API-Key in EODHD_API_KEY

--sample braucht kein Abo: Sharadars "test-api-key" liefert die 30 Dow-Titel der letzten 5 Jahre.
Der Lauf prüft damit Abruf, Rohkurs-Rückrechnung und Splits gegen die yfinance-Daten im Repo
und zählt, wie viele delistete Titel der Import ungefähr bringt. Ergebnis in data/delisted_sample/.

Gleiche Aufnahmeregel wie das Universum (update.py): Stammaktien an NYSE und Nasdaq, ohne Fonds,
SPACs, Optionsscheine usw. (symbols.BAD_NAME), an mindestens 20 Tagen seit START Rohkurs > 5
und Ø-Volumen(30) >= 1 Mio. Titel, die schon im Datensatz stecken (gleiche Kurse vor dem ersten
Lauf), fliegen raus.

Sharadar führt Firmen, die nach dem Delisting außerbörslich weiterhandeln (Pleiten mit Q-Suffix,
z. B. SQBGQ), als delistet mit Kursen bis heute. Die bleiben drin: Genau diese Verlierer fehlen
sonst. Wird ein Symbol neu vergeben, hängt Sharadar an das alte eine Zahl an (INOV2).

Ablage in data/delisted/ – NICHT im Git (.gitignore). Die Rohdaten der Anbieter dürfen mit einer
privaten Lizenz nicht weitergegeben werden, und dieses Repo ist öffentlich.
  ohlcv.parquet  Rohkurse wie damals gehandelt: date, ticker, open, high, low, close, volume
  splits.csv     ticker, date, ratio (neue/alte Aktien, wie data/splits.csv)
  tickers.csv    ticker, symbol, name, exchange, first_date, last_date, eligible_days, cik, …
  info.json      Zählungen des Laufs

ticker = Symbol + Monat des letzten Handelstags (z. B. TWTR-202210). So kollidieren
wiederverwendete Symbole nicht mit heutigen Titeln.

Weiter geht es mit delisted_sec.py (Quartalszahlen und Earnings-Termine von der SEC) und
backtest/research/v42_delisted.py (V4.2 mit und ohne delistete Titel).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction

import numpy as np
import pandas as pd
import requests

from common import (COLS, DATA, MIN_ELIGIBLE_DAYS, START, future_factor, load_universe, normalize, read_splits,
                    read_store)
from symbols import BAD_NAME
from update import eligible_days

OUT = DATA / "delisted"
OUT_SAMPLE = DATA / "delisted_sample"
KEY_ENV = {"sharadar": "SHARADAR_API_KEY", "eodhd": "EODHD_API_KEY"}
SHARADAR = "https://api.sharadar.com/v1.0/data"
SHARADAR_TEST_KEY = "test-api-key"  # öffentlich, liefert nur die Dow-30-Stichprobe (5 Jahre)
SHARADAR_SAMPLE = ["AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GOOGL", "GS", "HD",
                   "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW",
                   "TRV", "UNH", "V", "WMT"]
LIMIT = 10000  # Zeilen je Abfrage (Sharadar-Standard)
EOD = "https://eodhd.com/api"

# Nur NYSE und Nasdaq wie das Universum (symbols.OTHER_EXCHANGES). Unbekannte Werte bleiben drin,
# weil Anbieter delistete Titel teils ohne Börse oder als "DELISTED" führen.
OTHER_VENUE = re.compile(r"OTC|PINK|GREY|NMFQS|NYSE ?MKT|NYSE ?AMERICAN|AMEX|ARCA|BATS|CBOE|IEX|INDEX", re.I)

SPLIT_JUMP = 0.005      # Faktorsprung ab 0,5 % gilt als Split/Aktiendividende (bei Pennystocks
                        # höher, siehe splits_from_factor: Kurse haben nur 3 Nachkommastellen)
SHARADAR_MAX_ERR = 0.02  # Rückrechnung Rohkurs + Splits -> Sharadar-Close, max. Abweichung
EODHD_MAX_JUMP = 0.25    # Sprung zwischen Split- und Gesamtbereinigung (fehlender Split?)
DUP_SHARE = 0.8          # Anteil gleicher Schlusskurse (letzte 30 Tage) für "schon im Datensatz"


class ApiError(RuntimeError):
    pass


class AuthError(ApiError):
    pass


class Client:
    """GET mit Wiederholung und einfachem Rate-Limit. Der API-Key taucht nie in Meldungen auf."""

    def __init__(self, key: str, min_interval: float, headers: dict | None = None, key_param: str | None = None):
        self.key = key
        self.key_param = key_param  # Key zusätzlich als Query-Parameter (Sharadar dokumentiert api_key)
        self.min_interval = min_interval
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "systemplus-backtest"
        self.s.headers.update(headers or {})
        self._lock = threading.Lock()
        self._next = 0.0

    def _redact(self, text: str) -> str:
        return text.replace(self.key, "***") if self.key else text

    def get(self, url: str, params: dict, tries: int = 5, follow: bool = True):
        """Antwort bei 200, None bei 404. follow=False: Weiterleitung (Bulk-Download) zurückgeben."""
        if self.key_param:
            params = {**params, self.key_param: self.key}
        err = ""
        for i in range(tries):
            with self._lock:
                wait = self._next - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._next = time.monotonic() + self.min_interval
            try:
                r = self.s.get(url, params=params, timeout=120, allow_redirects=follow)
            except requests.RequestException as e:
                err = self._redact(str(e))
                time.sleep(2 * (i + 1))
                continue
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code == 429:  # Rate-Limit: bis zum Reset warten (Sharadar: x-ratelimit-reset, Unix-Zeit)
                reset = r.headers.get("x-ratelimit-reset") or ""
                retry = r.headers.get("retry-after") or ""
                wait = float(reset) - time.time() if reset.replace(".", "").isdigit() else \
                    float(retry) if retry.replace(".", "").isdigit() else 60.0
                wait = min(max(wait, 5.0), 3600.0) + 1
                print(f"  Rate-Limit erreicht, warte {wait:.0f} s", flush=True)
                time.sleep(wait)
                err = "HTTP 429"
                continue
            if not follow and r.status_code in (301, 302, 303, 307, 308):
                return r  # Bulk-Download: Weiterleitung auf die Download-URL
            if r.status_code in (401, 403):
                hint = " (Gratis-Stufe: Abo nicht aktiv oder Daten außerhalb der Stichprobe)" \
                    if "free tier" in r.text.lower() else ""
                raise AuthError(f"HTTP {r.status_code} – API-Key oder Abo prüfen{hint}: {self._redact(r.text[:300])}")
            if r.status_code in (400, 422):  # falsche Anfrage, Wiederholen hilft nicht
                raise ApiError(f"{url}: HTTP {r.status_code} {self._redact(r.text[:300])}")
            err = f"HTTP {r.status_code}"
            time.sleep(min(60, 2 ** (i + 1)))
        raise ApiError(f"{url}: {err}")

    def get_json(self, url: str, params: dict, tries: int = 5):
        r = self.get(url, params, tries)
        return None if r is None else r.json()


def _col(t: pd.DataFrame, name: str, default="") -> pd.Series:
    return t[name] if name in t else pd.Series(default, index=t.index)


def is_common(cat: str) -> bool:
    c = str(cat or "").lower()
    return ("common" in c or c == "") and not any(w in c for w in ("warrant", "preferred", "right", "unit"))


# ---------------------------------------------------------------------------
# Splits aus dem Verhältnis Rohkurs / split-bereinigter Kurs

def _nice(r: float) -> float:
    """Auf ein übliches Verhältnis runden (2, 3/2, 1/10 …), wenn es auf 0,05 % passt."""
    fr = Fraction(r).limit_denominator(20)
    return float(fr) if fr and abs(float(fr) / r - 1) < 0.0005 else round(r, 6)


def splits_from_factor(code: pd.Series, date: pd.Series, f: pd.Series, adj: pd.Series | None = None,
                       win: int = 5) -> pd.DataFrame:
    """f = Rohkurs / split-bereinigter Kurs = Produkt aller späteren Splits (bei Sharadar inkl.
    Aktiendividenden). Ein Split ist ein dauerhafter Sprung von f; ratio = f vorher / f nachher
    (2 = 2-für-1), Datum = erster Tag danach. adj = bereinigter Kurs: Rundung auf 0,001 USD rauscht
    bei Pennystocks, die Schwelle steigt dort (0,002 / Kurs)."""
    rows = []
    d = pd.DataFrame({"code": code.to_numpy(), "date": date.to_numpy(), "f": f.to_numpy(float),
                      "p": (adj if adj is not None else pd.Series(np.inf, index=f.index)).to_numpy(float)})
    for t, g in d.groupby("code", sort=False):
        F, D, P = g["f"].to_numpy(), g["date"].to_numpy(), g["p"].to_numpy()
        if len(F) < 2:
            continue
        thr = np.log1p(np.maximum(SPLIT_JUMP, 0.002 / np.maximum(P, 1e-6)))
        cand = np.where(np.abs(np.diff(np.log(F))) > thr[1:])[0] + 1
        last = -win
        for j in cand:
            if j - last < win:
                continue
            r = np.median(F[max(0, j - win):j]) / np.median(F[j:j + win])
            if abs(np.log(r)) <= thr[j]:
                continue  # einzelner Ausreißer oder Rundung, kein dauerhafter Sprung
            rows.append((t, D[j], _nice(float(r))))
            last = j
    return pd.DataFrame(rows, columns=["code", "date", "ratio"])


def _factor(raw: pd.DataFrame, splits: pd.DataFrame) -> np.ndarray:
    return future_factor(raw.rename(columns={"code": "ticker"}).reset_index(drop=True),
                         splits.rename(columns={"code": "ticker"}))


# ---------------------------------------------------------------------------
# Sharadar direkt (api.sharadar.com/v1.0/data/{table}): tickers + stocks
# Abfrage: ticker (kommagetrennt), from/to, fields, sort, limit (10000), skip, format=csv.
# Der Key geht als api_key und Header x-api-key mit; Fehlermeldungen schwärzen ihn.

def sh_query(c: Client, table: str, params: dict, limit: int = LIMIT) -> pd.DataFrame:
    """CSV-Abfrage, blättert mit skip. Nur mit eindeutiger Sortierung aufrufen (permaticker,
    oder date bei genau einem Ticker), sonst können sich Seiten überschneiden. CSV verträgt
    größere limits (JSON höchstens 10.000)."""
    out, skip = [], 0
    while True:
        r = c.get(f"{SHARADAR}/{table}", {**params, "format": "csv", "limit": limit, "skip": skip})
        if r is None or not r.text.strip():
            break
        df = pd.read_csv(io.StringIO(r.text), keep_default_na=False, na_values=["", "N/A"],
                         dtype={"ticker": str, "permaticker": str, "contraticker": str})
        out.append(df)
        if len(df) < limit:
            break
        skip += limit
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def cik_from_url(u) -> float:
    m = re.search(r"CIK=0*(\d+)", str(u or ""), re.I)
    return float(m.group(1)) if m else np.nan


TICKER_FIELDS = "table,permaticker,ticker,name,exchange,isdelisted,category,sector,industry,lastpricedate,secfilings"


def _sh_frame(t: pd.DataFrame) -> pd.DataFrame:
    if "table" in t:
        t = t[t["table"] == "stocks"]
    t = t.drop_duplicates("permaticker").reset_index(drop=True)
    code = t["ticker"].astype(str)
    return pd.DataFrame({
        "provider_id": t["permaticker"].astype(str), "code": code,
        # wiederverwendetes Symbol: Sharadar hängt beim delisteten Titel eine Zahl an (INOV2 -> INOV)
        "symbol": code.str.replace(r"(?<=[A-Z])\d+$", "", regex=True),
        "name": t["name"].fillna("").astype(str), "exchange": t["exchange"].fillna("").astype(str),
        "category": t["category"].fillna("").astype(str), "sector": t["sector"], "industry": t["industry"],
        "last_hint": pd.to_datetime(t["lastpricedate"], errors="coerce"),
        "cik": t["secfilings"].map(cik_from_url)})


def sharadar_candidates(c: Client, start: str) -> pd.DataFrame:
    """Alle delisteten Titel der Kurstabelle mit letztem Kurs ab start. from begrenzt lastpricedate."""
    t = sh_query(c, "tickers", {"table": "stocks", "from": start, "sort": "permaticker.asc", "fields": TICKER_FIELDS},
                 limit=100000)
    if t.empty:
        raise ApiError("Sharadar: Tickertabelle leer")
    return _sh_frame(t[t["isdelisted"] == "Y"])


def sharadar_sample(c: Client) -> pd.DataFrame:
    t = sh_query(c, "tickers", {"ticker": ",".join(SHARADAR_SAMPLE), "from": "1990-01-01", "fields": TICKER_FIELDS})
    return _sh_frame(t)


def sharadar_to_raw(p: pd.DataFrame):
    """stocks: open/high/low/close/volume split-bereinigt, closeunadj roh.
    Rohwerte wie von Sharadar beschrieben: OpenUnadj = Open × CloseUnadj / Close,
    VolumeUnadj = Volume × Close / CloseUnadj."""
    p = p.rename(columns={"ticker": "code"}).copy()
    for col in ("open", "high", "low", "close", "volume", "closeunadj"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p["date"] = pd.to_datetime(p["date"])
    p = p[(p["close"] > 0) & (p["closeunadj"] > 0)].sort_values(["code", "date"]).reset_index(drop=True)
    f = p["closeunadj"] / p["close"]
    raw = pd.DataFrame({"date": p["date"], "code": p["code"], "open": p["open"] * f, "high": p["high"] * f,
                        "low": p["low"] * f, "close": p["closeunadj"], "volume": p["volume"] / f})
    splits = splits_from_factor(p["code"], p["date"], f, adj=p["close"])
    # Prüfung: Rohkurse mit den abgeleiteten Splits wieder bereinigen und mit Sharadar vergleichen
    err = (raw["close"] / _factor(raw, splits) / p["close"] - 1).abs()
    check = err.groupby(p["code"]).max()
    return raw, splits, check, check <= SHARADAR_MAX_ERR


PRICE_FIELDS = "ticker,date,open,high,low,close,volume,closeunadj"
DELIST_ACTIONS = {  # Rangfolge: erster Treffer gewinnt, wenn mehrere am selben Tag stehen
    "bankruptcyliquidation": "Konkurs", "acquisitionby": "Übernahme", "mergerto": "Fusion",
    "regulatorydelisting": "Delisting (Regulierung)", "voluntarydelisting": "Delisting (freiwillig)",
    "delisted": "Delisting"}


def sharadar_bulk(c: Client, codes: list[str], start: str, years: str = "10") -> pd.DataFrame | None:
    """Ganze stocks-Tabelle als ZIP (eine CSV, 10 Jahre ≈ 325 MB), hier auf codes ab start gefiltert.
    Von Sharadar für große Abrufe empfohlen, schont das Rate-Limit. None, wenn das nicht klappt."""
    import zipfile
    from urllib.parse import urljoin
    path = OUT / "stocks_bulk.zip"
    try:
        r = c.get(f"{SHARADAR}/stocks", {"years": years}, follow=False)
        loc = r.headers.get("location") if r is not None else None
        if not loc:
            print("  Bulk: keine Download-Adresse erhalten")
            return None
        OUT.mkdir(parents=True, exist_ok=True)
        # vorsignierte Adresse: ohne API-Key abrufen
        with requests.get(urljoin(f"{SHARADAR}/stocks", loc), stream=True, timeout=900) as d:
            d.raise_for_status()
            with open(path, "wb") as fh:
                for chunk in d.iter_content(1 << 20):
                    fh.write(chunk)
        print(f"  Bulk geladen: {path.stat().st_size / 1e6:.0f} MB", flush=True)
        want, parts = set(codes), []
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(name) as fh:
                for ch in pd.read_csv(fh, usecols=PRICE_FIELDS.split(","), dtype={"ticker": str}, chunksize=1_000_000,
                                      keep_default_na=False, na_values=["", "N/A"]):
                    ch = ch[ch["ticker"].isin(want) & (ch["date"].astype(str) >= start)]
                    if len(ch):
                        parts.append(ch)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PRICE_FIELDS.split(","))
    except AuthError as e:
        print(f"  Bulk nicht freigeschaltet: {e}")
        return None
    except (ApiError, requests.RequestException, OSError, ValueError, StopIteration, KeyError) as e:
        print(f"  Bulk fehlgeschlagen ({type(e).__name__}: {c._redact(str(e))[:200]}), nehme Einzelabfragen")
        return None
    finally:
        path.unlink(missing_ok=True)  # Rohdatei nicht liegen lassen


def sharadar_actions(c: Client, codes: list[str], start: str) -> pd.DataFrame:
    """Splits und Delisting-Gründe aus actions (50 Ticker je Abfrage, auch Dividenden kommen mit)."""
    parts = []
    for i in range(0, len(codes), 50):
        parts.append(sh_query(c, "actions", {"ticker": ",".join(codes[i:i + 50]), "from": start, "sort": "date.asc",
                                             "fields": "date,action,ticker,value,contraticker"}, limit=100000))
    parts = [x for x in parts if len(x)]
    if not parts:
        return pd.DataFrame(columns=["date", "action", "ticker", "value", "contraticker"])
    a = pd.concat(parts, ignore_index=True)
    return a[a["action"].isin(set(DELIST_ACTIONS) | {"split"})].reset_index(drop=True)


def delist_info(a: pd.DataFrame) -> pd.DataFrame:
    """Je Ticker: Datum und Grund des Delistings (frühestes Ereignis, bei Gleichstand nach Rangfolge)."""
    d = a[a["action"].isin(DELIST_ACTIONS)].copy()
    if d.empty:
        return pd.DataFrame(columns=["delist_date", "delist_reason", "contraticker"])
    d["rank"] = d["action"].map({k: i for i, k in enumerate(DELIST_ACTIONS)})
    d = d.sort_values(["ticker", "date", "rank"]).drop_duplicates("ticker")
    return pd.DataFrame({"delist_date": d["date"].astype(str).to_numpy(),
                         "delist_reason": d["action"].map(DELIST_ACTIONS).to_numpy(),
                         "contraticker": d["contraticker"].to_numpy()}, index=d["ticker"].to_numpy())


def split_mismatch(splits: pd.DataFrame, a: pd.DataFrame, codes) -> list[str]:
    """Ticker, bei denen die aus den Kursen abgeleiteten Splits nicht zu actions passen (nur Info)."""
    s = a[a["action"] == "split"]
    out = []
    for t in codes:
        mine = {(str(pd.Timestamp(d).date()), round(float(r), 3)) for d, r in
                splits.loc[splits["code"] == t, ["date", "ratio"]].values}
        theirs = {(str(d), round(float(v), 3)) for d, v in s.loc[s["ticker"] == t, ["date", "value"]].values}
        if mine != theirs:
            out.append(t)
    return out


def _sh_batch(c: Client, batch: list[str], start: str) -> pd.DataFrame:
    """Mehrere Ticker in einer Abfrage, solange sie sicher unter LIMIT Zeilen bleibt. Voll -> einzeln
    nachladen (bei genau einem Ticker ist date.asc eindeutig, dann darf geblättert werden)."""
    p = {"ticker": ",".join(batch), "from": start, "fields": PRICE_FIELDS, "sort": "date.asc"}
    if len(batch) == 1:
        return sh_query(c, "stocks", p)
    r = c.get(f"{SHARADAR}/stocks", {**p, "format": "csv", "limit": LIMIT, "skip": 0})
    df = pd.DataFrame() if r is None or not r.text.strip() else \
        pd.read_csv(io.StringIO(r.text), keep_default_na=False, na_values=["", "N/A"], dtype={"ticker": str})
    if len(df) < LIMIT:
        return df
    return pd.concat([_sh_batch(c, [t], start) for t in batch], ignore_index=True)


def sharadar_prices(c: Client, codes: list[str], start: str, workers: int, mode: str = "auto"):
    """mode: bulk (ZIP), api (Einzelabfragen) oder auto (erst Bulk, sonst Einzelabfragen)."""
    if mode in ("auto", "bulk"):
        p = sharadar_bulk(c, codes, start)
        if p is not None:
            raw, splits, check, ok = sharadar_to_raw(p)
            return raw, splits, check, ok, []
        if mode == "bulk":
            raise ApiError("Bulk-Download nicht möglich")
    days = max(1, int(np.busday_count(start, datetime.now(timezone.utc).date().isoformat())))
    size = max(1, int(0.9 * LIMIT) // days)  # z. B. 4 Ticker à ~2.250 Tage seit 2018
    batches = [codes[i:i + size] for i in range(0, len(codes), size)]
    parts, failed = [], []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_sh_batch, c, b, start): b for b in batches}
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                parts.append(fu.result())
            except AuthError:
                raise
            except ApiError as e:
                failed += futs[fu]
                print(f"  {','.join(futs[fu])}: {e}")
            if n % 100 == 0 or n == len(batches):
                print(f"  Kurse {n}/{len(batches)} Abfragen ({size} Ticker je Abfrage)", flush=True)
    cols = PRICE_FIELDS.split(",")
    parts = [x for x in parts if len(x)]
    p = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    raw, splits, check, ok = sharadar_to_raw(p)
    return raw, splits, check, ok, failed


# ---------------------------------------------------------------------------
# EODHD: exchange-symbol-list (delisted=1), eod, splits

def eodhd_candidates(c: Client) -> pd.DataFrame:
    j = c.get_json(f"{EOD}/exchange-symbol-list/US",
                   {"delisted": 1, "type": "common_stock", "fmt": "json", "api_token": c.key}) or []
    t = pd.DataFrame(j)
    if t.empty:
        raise ApiError("EODHD: leere Liste delisteter Titel")
    code = t["Code"].astype(str)
    return pd.DataFrame({
        "provider_id": code, "code": code,
        "symbol": code.str.replace(r"(?i)[_.\-]old\d*$", "", regex=True),
        "name": _col(t, "Name").fillna("").astype(str), "exchange": _col(t, "Exchange").fillna("").astype(str),
        "category": _col(t, "Type").fillna("").astype(str), "sector": np.nan, "industry": np.nan,
        "last_hint": pd.NaT, "cik": np.nan})


def eodhd_one(c: Client, code: str, start: str):
    e = c.get_json(f"{EOD}/eod/{code}.US", {"from": start, "period": "d", "fmt": "json", "api_token": c.key})
    if not e:
        return code, None, []
    s = c.get_json(f"{EOD}/splits/{code}.US", {"from": start, "fmt": "json", "api_token": c.key}) or []
    return code, pd.DataFrame(e), s


def _ratio(s: str) -> float:
    """EODHD-Split "neu/alt", z. B. "4.000000/1.000000" -> 4.0 (beide Zahlen teilen, nicht nur die erste)."""
    a, _, b = str(s).partition("/")
    try:
        return float(a) / float(b) if b else float(a)
    except (ValueError, ZeroDivisionError):
        return np.nan


def eodhd_to_raw(parts: list[pd.DataFrame], sp_rows: list[tuple]):
    """EODHD: open/high/low/close roh, volume split-bereinigt, adjusted_close split- und
    dividendenbereinigt. Rohvolumen = volume / Produkt der späteren Splits."""
    cols = ["date", "code", "open", "high", "low", "close", "volume", "adjusted_close"]
    p = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    for col in cols[2:]:
        p[col] = pd.to_numeric(_col(p, col, np.nan), errors="coerce")
    p["date"] = pd.to_datetime(p["date"])
    p = p[(p["close"] > 0)].sort_values(["code", "date"]).reset_index(drop=True)
    splits = pd.DataFrame(sp_rows, columns=["code", "date", "ratio"])
    splits["date"] = pd.to_datetime(splits["date"])
    splits = splits[splits["ratio"] > 0].drop_duplicates(["code", "date"])
    ff = _factor(p, splits)
    raw = p[["date", "code", "open", "high", "low", "close"]].copy()
    raw["volume"] = p["volume"] / ff
    # Prüfung: (Close / Splitfaktor) / adjusted_close darf nur durch Dividenden leicht springen
    q = np.log((p["close"] / ff) / p["adjusted_close"])
    jump = q.groupby(p["code"]).diff().abs()
    check = jump.groupby(p["code"]).max().fillna(0)
    return raw, splits, check, check <= EODHD_MAX_JUMP


def eodhd_prices(c: Client, codes: list[str], start: str, workers: int):
    parts, sp_rows, failed = [], [], []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(eodhd_one, c, code, start): code for code in codes}
        for n, fu in enumerate(as_completed(futs), 1):
            try:
                code, p, s = fu.result()
            except AuthError:
                raise
            except ApiError as e:
                failed.append(futs[fu])
                print(f"  {futs[fu]}: {e}")
                continue
            if p is not None and not p.empty and "close" in p:
                parts.append(p.assign(code=code))
                sp_rows += [(code, x.get("date"), _ratio(x.get("split"))) for x in s]
            if n % 250 == 0 or n == len(codes):
                print(f"  Kurse {n}/{len(codes)}, mit Daten {len(parts)}", flush=True)
    raw, splits, check, ok = eodhd_to_raw(parts, sp_rows)
    return raw, splits, check, ok, failed


# ---------------------------------------------------------------------------

def duplicates(raw: pd.DataFrame, store: pd.DataFrame, n: int = 30) -> dict[str, str]:
    """code -> Ticker im Datensatz, wenn die letzten n Schlusskurse (vor dem ersten Lauf, also
    `raw` entsprechend abschneiden) fast alle gleich sind: Titel, die beim ersten Lauf noch
    gelistet waren und deshalb schon über yfinance im Datensatz stecken."""
    if store.empty or raw.empty:
        return {}
    px = store.pivot_table(index="date", columns="ticker", values="close")
    pos = {d: i for i, d in enumerate(px.index)}
    M, names = px.to_numpy(), px.columns
    out = {}
    for code, g in raw.sort_values("date").groupby("code"):
        g = g.tail(n)
        idx = np.array([pos.get(d, -1) for d in g["date"]])
        m = idx >= 0
        if m.sum() < 10:
            continue
        v = g["close"].to_numpy()[m][:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            hit = (np.abs(M[idx[m]] / v - 1) < 0.005).sum(axis=0)
        j = int(hit.argmax())
        if hit[j] >= DUP_SHARE * m.sum():
            out[code] = str(names[j])
    return out


def candidate_filters(cand: pd.DataFrame, start: pd.Timestamp, cutoff: pd.Timestamp, upper: bool):
    """Börse, Wertpapierart, Name, Zeitraum. upper=False (Sharadar): kein Höchstdatum, weil Titel nach
    dem Delisting außerbörslich weiterlaufen können; dort entscheidet später die Dublettenprüfung."""
    f_venue = ~cand["exchange"].str.contains(OTHER_VENUE)
    f_common = cand["category"].map(is_common)
    f_name = ~cand["name"].map(lambda n: bool(BAD_NAME.search(n)))
    lh = cand["last_hint"]
    f_time = lh.isna() | ((lh >= start) & ((lh < cutoff) if upper else True))
    counts = dict(ohne_nyse_nasdaq=int((~f_venue).sum()), keine_stammaktie=int((f_venue & ~f_common).sum()),
                  fonds_spac_usw=int((f_venue & f_common & ~f_name).sum()),
                  ausserhalb_zeitraum=int((f_venue & f_common & f_name & ~f_time).sum()))
    return cand[f_venue & f_common & f_name & f_time].drop_duplicates("code").reset_index(drop=True), counts


def compare_with_store(raw: pd.DataFrame, splits: pd.DataFrame, check: pd.Series) -> pd.DataFrame:
    """Probelauf: Sharadar-Rohkurse und abgeleitete Splits gegen die yfinance-Daten im Repo."""
    store, sp = read_store(), read_splits()
    rows = []
    for code, g in raw.groupby("code"):
        s = store[store["ticker"] == code].set_index("date")
        g = g.set_index("date")
        common = g.index.intersection(s.index)
        if len(common) == 0:
            rows.append({"ticker": code, "tage": 0})
            continue
        a, b = g.loc[common], s.loc[common]
        rel = {c: float((a[c] / b[c] - 1).abs().quantile(0.99)) for c in ("open", "high", "low", "close")}
        lo, hi = g.index.min(), g.index.max()
        mine = {(pd.Timestamp(d).date(), round(r, 4)) for d, r in splits.loc[splits["code"] == code, ["date", "ratio"]].values}
        repo = {(pd.Timestamp(d).date(), round(r, 4)) for d, r in sp.loc[(sp["ticker"] == code) & sp["date"].between(
            lo + pd.Timedelta(days=1), hi), ["date", "ratio"]].values}
        rows.append({"ticker": code, "tage": len(common),
                     **{f"{c}_abw_p99_%": round(100 * v, 3) for c, v in rel.items()},
                     "volumen_sharadar/yahoo": round(float((a["volume"] / b["volume"]).median()), 3),
                     "splits_sharadar": ";".join(f"{d}:{r:g}" for d, r in sorted(mine)),
                     "splits_repo": ";".join(f"{d}:{r:g}" for d, r in sorted(repo)),
                     "splits_gleich": mine == repo, "rueckrechnung_max_%": round(100 * float(check.get(code, np.nan)), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", choices=sorted(KEY_ENV), required=True)
    ap.add_argument("--sample", action="store_true", help="Sharadar-Probelauf mit dem Gratis-Schlüssel (Dow 30)")
    ap.add_argument("--start", default=START)
    ap.add_argument("--cutoff", help="erster Lauf (Standard: frühestes 'added' in universe.csv)")
    ap.add_argument("--limit", type=int, help="nur die ersten n Kandidaten (Test)")
    ap.add_argument("--workers", type=int, default=4, help="parallele Abrufe")
    ap.add_argument("--mode", choices=["auto", "bulk", "api"], default="auto",
                    help="Sharadar-Kurse: Bulk-ZIP, Einzelabfragen oder erst Bulk, sonst Einzelabfragen")
    args = ap.parse_args()
    if args.sample and args.provider != "sharadar":
        raise SystemExit("--sample gibt es nur für sharadar")

    key = SHARADAR_TEST_KEY if args.sample else os.environ.get(KEY_ENV[args.provider], "").strip()
    if not key:
        raise SystemExit(f"API-Key fehlt: Umgebungsvariable {KEY_ENV[args.provider]} setzen")
    uni = load_universe()
    added = uni["added"].dropna().astype(str)
    added = added[added.str.match(r"\d{4}-\d{2}-\d{2}")]
    cutoff = pd.Timestamp(args.cutoff or (added.min() if len(added) else datetime.now(timezone.utc).date()))
    start = pd.Timestamp(args.start)
    if args.sample:  # Gratis-Stufe: nur die letzten 5 Jahre
        start = max(start, pd.Timestamp(datetime.now(timezone.utc).date()) - pd.DateOffset(years=5) + pd.Timedelta(days=3))
    print(f"Anbieter {args.provider}{' (Probelauf)' if args.sample else ''}: ab {start.date()}, erster Lauf {cutoff.date()}")

    if args.provider == "sharadar":
        c = Client(key, min_interval=0.1, headers={"x-api-key": key}, key_param="api_key")
    else:
        c = Client(key, min_interval=0.07)  # EODHD: < 1000/min
    info: dict = {"provider": args.provider, "start": str(start.date()), "cutoff": str(cutoff.date())}

    if args.sample:
        # Wie viele delistete Titel sieht schon die Gratis-Stufe (5 Jahre)? Gibt eine Größenordnung.
        dl = sharadar_candidates(c, str(start.date()))
        dl_ok, counts = candidate_filters(dl, start, cutoff, upper=False)
        info["delistet_5j_vor_kursfiltern"] = {"gesamt": len(dl), "nach_filtern": len(dl_ok), **counts,
                                               "boersen": dl["exchange"].value_counts().head(10).to_dict()}
        cand = sharadar_sample(c)
    else:
        cand = sharadar_candidates(c, str(start.date())) if args.provider == "sharadar" else eodhd_candidates(c)
    info.update(kandidaten=len(cand), boersen=cand["exchange"].replace("", "(leer)").value_counts().head(15).to_dict())
    cand, counts = candidate_filters(cand, start, cutoff, upper=args.provider == "eodhd")
    info.update(counts)
    if args.limit:
        cand = cand.head(args.limit)
    print(f"{info['kandidaten']} Titel beim Anbieter, {len(cand)} nach Börse/Art/Name/Zeitraum")

    codes = cand["code"].tolist()
    if args.provider == "sharadar":
        raw, splits, check, ok, failed = sharadar_prices(c, codes, str(start.date()), args.workers,
                                                         mode="api" if args.sample else args.mode)
    else:
        raw, splits, check, ok, failed = eodhd_prices(c, codes, str(start.date()), args.workers)
    raw = raw[raw["date"] >= start]

    if args.sample:
        # Ausführliche Zahlen nur lokal in data/delisted_sample/ (nicht im Git). Sharadars Lizenz (§8)
        # erlaubt Bewertungen der Datenqualität nur mit Erlaubnis zu veröffentlichen – Actions-Logs
        # öffentlicher Repos sind öffentlich. In der Konsole stehen deshalb nur Prüfergebnisse.
        OUT_SAMPLE.mkdir(parents=True, exist_ok=True)
        acts = sharadar_actions(c, codes, str(start.date()))
        cmp = compare_with_store(raw, splits, check)
        cmp.to_csv(OUT_SAMPLE / "vergleich.csv", index=False)
        dup = duplicates(raw[raw["date"] < cutoff], read_store())
        n = int(raw["code"].nunique())
        mism = split_mismatch(splits, acts, sorted(raw["code"].unique()))
        both = cmp[cmp["tage"] > 0] if "tage" in cmp else cmp.iloc[0:0]  # Titel, die auch im Repo stecken
        res = {
            "Abruf tickers/stocks/actions": not failed and n > 0,
            f"Kurse für {n}/{len(SHARADAR_SAMPLE)} Dow-Titel": n >= len(SHARADAR_SAMPLE) - 2,
            f"Rohkurse = yfinance-Rohkurse (Close, {len(both)} Titel im Repo)":
                len(both) >= 15 and bool((both["close_abw_p99_%"] < 1).all()),
            "Splits aus Kursen = Splits im Repo": len(both) >= 15 and bool(both["splits_gleich"].astype(bool).all()),
            "Splits aus Kursen = Sharadar actions": not mism,
            "Rückrechnung Rohkurs + Splits = Sharadar": bool(ok.reindex(raw["code"].unique()).fillna(False).all()),
            "Dublettenprüfung erkennt alle Titel im Repo":
                len(both) > 0 and sum(dup.get(t) == t for t in both["ticker"]) == len(both),
        }
        info.update(pruefungen=res, abruf_fehler=failed, titel_mit_kursen=n, zeilen=len(raw), splits_abweichend=mism,
                    run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        (OUT_SAMPLE / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        print("\nProbelauf:")
        for k, v in res.items():
            print(f"  {'OK    ' if v else 'FEHLER'} {k}")
        dl = info["delistet_5j_vor_kursfiltern"]
        print(f"  Delistete Kandidaten der letzten 5 Jahre nach Börse/Art/Name: {dl['nach_filtern']} "
              f"(10 Jahre ab 2018: deutlich mehr; Kursfilter kommen erst beim Import)")
        print(f"Details lokal: {OUT_SAMPLE / 'vergleich.csv'}")
        if not all(res.values()):
            raise SystemExit(1)
        return

    span = raw.groupby("code")["date"].agg(["min", "max", "size"])
    # EODHD: Kurse enden mit dem Delisting -> nur Titel, die vor dem ersten Lauf endeten.
    # Sharadar: alle (auch außerbörslich weiterlaufende); schon Erfasstes fängt die Dublettenprüfung.
    in_time = span.index[span["max"] < cutoff] if args.provider == "eodhd" else span.index
    elig = eligible_days(raw[raw["code"].isin(in_time)].rename(columns={"code": "ticker"}))
    take = set(elig[elig >= MIN_ELIGIBLE_DAYS].index)
    bad_split = sorted(t for t in take if not bool(ok.get(t, False)))
    take -= set(bad_split)
    pre = raw[(raw["date"] < cutoff) & raw["code"].isin(take)]
    dup = duplicates(pre, read_store())
    take -= set(dup)
    info.update(mit_kursen=int(len(span)), abruf_fehler=failed, nach_erstem_lauf=int(len(span) - len(in_time)),
                grundfilter_nicht_erfuellt=int(len(in_time) - len(elig[elig >= MIN_ELIGIBLE_DAYS])),
                split_pruefung_fehlgeschlagen=bad_split, schon_im_datensatz=dup)

    meta = cand.set_index("code").loc[sorted(take)].reset_index()
    meta["first_date"] = meta["code"].map(span["min"])
    meta["last_date"] = meta["code"].map(span["max"])
    meta["bars"] = meta["code"].map(span["size"]).astype(int)
    # Delisting-Datum und -Grund (Sharadar actions): Übernahme, Fusion, Konkurs, Delisting
    meta["delist_date"], meta["delist_reason"], meta["contraticker"] = pd.NaT, None, None
    if args.provider == "sharadar" and len(meta):
        acts = sharadar_actions(c, meta["code"].tolist(), str(start.date()))
        di = delist_info(acts)
        meta["delist_date"] = pd.to_datetime(meta["code"].map(di["delist_date"]), errors="coerce")
        meta["delist_reason"] = meta["code"].map(di["delist_reason"])
        meta["contraticker"] = meta["code"].map(di["contraticker"])
        info.update(splits_abweichend_von_actions=split_mismatch(splits, acts, meta["code"].tolist()),
                    delisting_gruende=meta["delist_reason"].fillna("unbekannt").value_counts().to_dict())
    # Schlüssel: Symbol + Monat des Delistings (sonst des letzten Kurses)
    month = meta["delist_date"].fillna(meta["last_date"]).dt.strftime("%Y%m")
    meta["ticker"] = meta["symbol"].str.upper().str.replace(".", "-", regex=False) + "-" + month
    clash = meta["ticker"].duplicated(keep=False)
    meta.loc[clash, "ticker"] = meta.loc[clash, "ticker"] + "-" + meta.loc[clash, "provider_id"].astype(str)
    meta["eligible_days"] = meta["code"].map(elig).astype(int)
    meta["split_check"] = meta["code"].map(check).round(4)
    meta["provider"] = args.provider
    key_of = dict(zip(meta["code"], meta["ticker"]))

    OUT.mkdir(parents=True, exist_ok=True)
    px = raw[raw["code"].isin(take)].assign(ticker=lambda d: d["code"].map(key_of))
    px = normalize(px[COLS])
    px.to_parquet(OUT / "ohlcv.parquet", index=False, compression="zstd")
    sp = splits[splits["code"].isin(take)].assign(ticker=lambda d: d["code"].map(key_of))
    sp = sp[["ticker", "date", "ratio"]].sort_values(["ticker", "date"])
    sp.assign(date=pd.to_datetime(sp["date"]).dt.strftime("%Y-%m-%d")).to_csv(OUT / "splits.csv", index=False)
    for col in ("first_date", "last_date", "delist_date"):
        meta[col] = meta[col].dt.strftime("%Y-%m-%d")
    meta["cik"] = meta["cik"].astype("Int64")
    meta[["ticker", "symbol", "name", "exchange", "category", "first_date", "last_date", "delist_date",
          "delist_reason", "contraticker", "bars", "eligible_days", "cik", "sector", "industry", "provider",
          "provider_id", "split_check"]].sort_values("ticker").to_csv(OUT / "tickers.csv", index=False)

    info.update(aufgenommen=len(meta), zeilen=len(px), splits=len(sp),
                mit_cik=int(meta["cik"].notna().sum()),
                run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (OUT / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: (v if not isinstance(v, (list, dict)) or len(v) <= 10 else f"{len(v)} Einträge")
                      for k, v in info.items()}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
