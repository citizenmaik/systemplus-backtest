"""Quartalszahlen und Earnings-Termine der SEC für die delisteten Titel aus delisted.py.

Ohne Quartalszahlen fallen delistete Trades am Umsatz-Gate (rev_accel > 0) heraus, der Vergleich
mit/ohne delistete Titel wäre dann wertlos. Ablauf je Titel:
  1. CIK: Sharadar liefert sie mit (tickers.secfilings). Sonst (EODHD) Suche über den Firmennamen in
     SEC cik-lookup-data.txt.
  2. Prüfung über die Einreichungen (data.sec.gov/submissions): Die Firma muss im Zeitraum der
     Kurse 10-Q/10-K/20-F/40-F eingereicht haben. Bei mehreren Namenstreffern gewinnt die CIK mit
     den meisten Berichten im Zeitraum. Namenstreffer ohne Berichte im Zeitraum werden verworfen.
  3. companyfacts nach denselben Regeln wie fundamentals.py.
  4. Earnings-Termine: 8-K mit Item 2.02 (Results of Operations), Zeitpunkt = acceptanceDateTime.
     Die Uhrzeit entscheidet in v42_report.earn_since über den Reaktionstag (vor 12 Uhr = selber
     Tag). EDGAR führt die Zeit vermutlich in US-Ostküstenzeit, obwohl das JSON ein "Z" anhängt.
     --check prüft das an heutigen Titeln gegen die Yahoo-Termine in data/earnings.csv und nimmt
     die Lesart mit mehr Übereinstimmung.

-> data/delisted/fundamentals.parquet  (Spalten wie data/fundamentals.parquet)
-> data/delisted/earnings.csv          (Spalten wie data/earnings.csv, EPS leer)
-> data/delisted/sec.csv               (ticker, cik, Quelle, Berichte im Zeitraum, Earnings, Status)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

from common import DATA, read_csv, read_store
from fundamentals import UA, facts_rows, get, to_frame

OUT = DATA / "delisted"
SUB = "https://data.sec.gov/submissions/"
FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{:010d}.json"
LOOKUP = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
TICKERS = "https://www.sec.gov/files/company_tickers.json"
REPORTS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "20-F", "40-F"}
PAUSE = 0.12  # SEC: max. 10 Anfragen/s

LEGAL = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "LLC",
         "LP", "SA", "NV", "AG", "SE", "THE", "COMMON", "STOCK", "SHARES", "ORDINARY", "ADR", "ADS", "NEW"}


def sec(url: str):
    time.sleep(PAUSE)
    try:
        return get(url)
    except (requests.RequestException, ValueError) as e:
        print(f"  {url}: {e}")
        return None


def norm(name: str) -> str:
    """Firmenname vergleichbar machen: SEC "TWITTER, INC." und Anbieter "Twitter Inc" -> "TWITTER"."""
    n = str(name or "").upper().replace("&", " AND ")
    n = re.sub(r"/[A-Z]{2,4}/?", " ", n)                     # SEC-Zusätze wie /DE/, /NEW/
    n = re.sub(r"\bCL(ASS)?\s+[A-Z]\b", " ", n)              # Class A
    toks = re.sub(r"[^A-Z0-9 ]", " ", n).split()
    merged: list[str] = []
    for t in toks:                                             # U S -> US
        if len(t) == 1 and merged and len(merged[-1]) == 1 and merged[-1].isalpha() and t.isalpha():
            merged[-1] += t
        else:
            merged.append(t)
    core = [t for t in merged if t not in LEGAL]
    return " ".join(core or merged)


def name_index() -> dict[str, set[int]]:
    time.sleep(PAUSE)
    r = requests.get(LOOKUP, headers=UA, timeout=300)
    r.raise_for_status()
    idx: dict[str, set[int]] = defaultdict(set)
    for line in r.content.decode("latin-1").splitlines():
        parts = line.rsplit(":", 2)
        if len(parts) == 3 and parts[1].isdigit():
            idx[norm(parts[0])].add(int(parts[1]))
    return idx


def filings(cik: int, oldest: str) -> pd.DataFrame:
    """Alle Einreichungen ab `oldest` (recent + ältere Seiten)."""
    cols = ["form", "filingDate", "acceptanceDateTime", "items"]
    j = sec(f"{SUB}CIK{cik:010d}.json")
    if not j:
        return pd.DataFrame(columns=cols)
    frames = [pd.DataFrame(j.get("filings", {}).get("recent", {}))]
    for f in j.get("filings", {}).get("files", []):
        if str(f.get("filingTo", "9999")) >= oldest:
            k = sec(SUB + f["name"])
            if k:
                frames.append(pd.DataFrame(k))
    df = pd.concat(frames, ignore_index=True)
    for c in cols:
        if c not in df:
            df[c] = ""
    df = df[cols].fillna("")
    return df[df["filingDate"].astype(str) >= oldest].reset_index(drop=True)


def n_reports(fl: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> int:
    d = pd.to_datetime(fl["filingDate"], errors="coerce")
    return int((fl["form"].isin(REPORTS) & d.between(first - pd.Timedelta(days=120),
                                                       last + pd.Timedelta(days=120))).sum())


def earnings_events(fl: pd.DataFrame, tz: str) -> pd.Series:
    """8-K mit Item 2.02 -> Zeitpunkte in US-Ostküstenzeit (ohne Zeitzone)."""
    m = (fl["form"] == "8-K") & fl["items"].astype(str).str.contains(r"(?:^|,)\s*2\.02")
    ts = pd.to_datetime(fl.loc[m, "acceptanceDateTime"].astype(str).str.slice(0, 19), errors="coerce").dropna()
    if tz == "utc":
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)
    return ts.sort_values().reset_index(drop=True)


def reaction_idx(cal: np.ndarray, ts: pd.Series) -> np.ndarray:
    """Wie v42_report.earn_since: vor 12 Uhr selber Handelstag, sonst nächster."""
    d = ts.dt.normalize().to_numpy()
    ri = cal.searchsorted(d)
    same = (ri < len(cal)) & (cal[np.minimum(ri, len(cal) - 1)] == d)
    return ri + (same & (ts.dt.hour.to_numpy() >= 12))


def check_tz(n: int, cal: np.ndarray) -> tuple[str, dict]:
    """Earnings-Reaktionstag aus EDGAR (beide Lesarten) gegen Yahoo bei n heutigen Titeln."""
    E = pd.read_csv(DATA / "earnings.csv", keep_default_na=False, na_values=[""])
    E = E[E["date"].between("2021-01-01", "2025-12-31") & E["time"].notna()]
    E["ts"] = pd.to_datetime(E["date"] + " " + E["time"], errors="coerce")
    E = E.dropna(subset=["ts"])
    m = get(TICKERS) or {}
    cik = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in m.values()}
    pick = sorted(t for t, g in E.groupby("ticker") if len(g) >= 8 and t in cik)[:: 7][:n]
    hits = {"et": 0, "utc": 0}
    total = 0
    for t in pick:
        fl = filings(cik[t], "2020-06-01")
        y = E[E["ticker"] == t]["ts"].sort_values().reset_index(drop=True)
        ry = reaction_idx(cal, y)
        for tz in hits:
            ev = earnings_events(fl, tz)
            if ev.empty:
                continue
            re_ = reaction_idx(cal, ev)
            # je Yahoo-Termin der nächste EDGAR-Termin innerhalb von 3 Tagen
            for yi, yt in zip(ry, y):
                j = (ev - yt).abs().idxmin()
                if abs((ev[j] - yt).days) <= 3 and re_[j] == yi:
                    hits[tz] += 1
        total += len(y)
    tz = "et" if hits["et"] >= hits["utc"] else "utc"
    res = {"titel": len(pick), "yahoo_termine": total, "gleicher_reaktionstag_ostkueste": hits["et"],
           "gleicher_reaktionstag_utc": hits["utc"], "gewaehlt": tz}
    print("Zeitzonen-Prüfung:", json.dumps(res, ensure_ascii=False))
    return tz, res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", type=int, default=40, help="heutige Titel für die Zeitzonen-Prüfung (0 = aus)")
    ap.add_argument("--tz", choices=["et", "utc"], help="Lesart von acceptanceDateTime fest vorgeben")
    args = ap.parse_args()

    tk = read_csv(OUT / "tickers.csv", parse_dates=["first_date", "last_date"])
    print(f"{len(tk)} delistete Titel")
    store = read_store()
    cal = np.sort(store.loc[store["ticker"] == "SPY", "date"].unique())
    del store
    tz, tzinfo = (args.tz, {"gewaehlt": args.tz}) if args.tz else (check_tz(args.check, cal) if args.check else ("et", {}))

    need_names = tk["cik"].isna().any()
    idx = name_index() if need_names else {}
    rows, earn, log = [], [], []
    for i, r in enumerate(tk.itertuples(index=False), 1):
        oldest = str((r.first_date - pd.Timedelta(days=500)).date())
        if pd.notna(r.cik):
            cands, source = [int(r.cik)], "anbieter"
        else:
            cands, source = sorted(idx.get(norm(r.name), set()))[:5], "name"
        best, best_n, best_fl = None, -1, None
        for c in cands:
            fl = filings(c, oldest)
            n = n_reports(fl, r.first_date, r.last_date)
            if n > best_n:
                best, best_n, best_fl = c, n, fl
        status = "ok"
        if best is None:
            status = "keine CIK"
        elif best_n < 2 and source == "name":
            status, best = "Name passt, keine Berichte im Zeitraum", None
        elif best_n < 2:
            status = "keine Berichte im Zeitraum"
        n_earn = 0
        if best is not None:
            ev = earnings_events(best_fl, tz)
            ev = ev[(ev >= r.first_date - pd.Timedelta(days=30)) & (ev <= r.last_date + pd.Timedelta(days=5))]
            earn += [(r.ticker, t.strftime("%Y-%m-%d"), t.strftime("%H:%M"), None, None) for t in ev]
            n_earn = len(ev)
            facts = sec(FACTS.format(best))
            if facts:
                rows += facts_rows(r.ticker, facts)
            else:
                status = "keine XBRL-Zahlen"
        log.append((r.ticker, best, source, len(cands), best_n, n_earn, status))
        if i % 100 == 0 or i == len(tk):
            print(f"  {i}/{len(tk)}", flush=True)

    f = to_frame(rows)
    f.to_parquet(OUT / "fundamentals.parquet", index=False, compression="zstd")
    pd.DataFrame(earn, columns=["ticker", "date", "time", "eps_est", "eps_act"]).to_csv(OUT / "earnings.csv", index=False)
    lg = pd.DataFrame(log, columns=["ticker", "cik", "cik_quelle", "cik_kandidaten", "berichte_im_zeitraum",
                                    "earnings_termine", "status"])
    lg["cik"] = lg["cik"].astype("Int64")
    lg.to_csv(OUT / "sec.csv", index=False)
    info = json.loads((OUT / "info.json").read_text()) if (OUT / "info.json").exists() else {}
    info["sec"] = {"titel": len(tk), "mit_cik": int(lg["cik"].notna().sum()),
                   "mit_umsatz": int(f.loc[f["kennzahl"] == "revenue", "ticker"].nunique()) if len(f) else 0,
                   "mit_earnings_terminen": int((lg["earnings_termine"] > 0).sum()),
                   "status": lg["status"].value_counts().to_dict(), "zeitzone": tzinfo}
    (OUT / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(info["sec"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
