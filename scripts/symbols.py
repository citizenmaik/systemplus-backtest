"""Kandidatenliste: alle US-Stammaktien an Nasdaq und NYSE (Nasdaq-Trader-Symbolverzeichnisse)."""
from __future__ import annotations

import io
import re

import pandas as pd
import requests

URL_NASDAQ = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
URL_OTHER = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# otherlisted: N = NYSE, A = NYSE American, P = NYSE Arca, Z = Cboe BZX, V = IEX
OTHER_EXCHANGES = {"N": "NYSE"}

# Keine Optionsscheine, Units, Rechte, Vorzugsaktien, Anleihen
BAD_NAME = re.compile(
    r"warrant|\bunits?\b|\brights?\b|preferred|\bpfd\b|\bnotes? due\b|debenture|"
    r"\d+(\.\d+)?\s*%|subordinated|when issued",
    re.I,
)
BAD_SUFFIX = ("-U", "-WS", "-W", "-WT", "-R", "-RT", "-WI")


def _read(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60, headers={"User-Agent": "systemplus-backtest"})
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False)


def to_yahoo(symbol: str) -> str | None:
    s = symbol.strip().upper()
    if not s or any(ch in s for ch in "$^=+ "):
        return None
    s = s.replace(".", "-").replace("/", "-")
    if s.endswith(BAD_SUFFIX):
        return None
    return s


def parse(nasdaq: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in nasdaq.iterrows():
        if r.get("Test Issue") == "Y" or r.get("ETF") == "Y" or r.get("NextShares") == "Y":
            continue
        rows.append((r["Symbol"], r["Security Name"], "NASDAQ"))
    for _, r in other.iterrows():
        if r.get("Test Issue") == "Y" or r.get("ETF") == "Y":
            continue
        exch = OTHER_EXCHANGES.get(r.get("Exchange", ""))
        if exch:
            rows.append((r["ACT Symbol"], r["Security Name"], exch))
    out = []
    for sym, name, exch in rows:
        if BAD_NAME.search(name or ""):
            continue
        t = to_yahoo(sym)
        if t:
            out.append({"ticker": t, "name": (name or "").strip(), "exchange": exch})
    df = pd.DataFrame(out, columns=["ticker", "name", "exchange"])
    return df.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def fetch_candidates() -> pd.DataFrame:
    return parse(_read(URL_NASDAQ), _read(URL_OTHER))


if __name__ == "__main__":
    c = fetch_candidates()
    print(c.exchange.value_counts().to_string())
    print(len(c), "Kandidaten")
