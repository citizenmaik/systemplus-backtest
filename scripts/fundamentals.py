"""Quartalszahlen aus SEC EDGAR (XBRL companyfacts) mit Meldedatum -> data/fundamentals.parquet.

Je Titel und Kennzahl alle Werte mit Zeitraum (start, end), Meldedatum (filed) und Formular.
Point-in-time: Ein Wert darf im Backtest erst ab `filed` verwendet werden.

Kennzahlen: Umsatz (mehrere us-gaap-Bezeichnungen bzw. ifrs Revenue), EPS verwässert,
Nettogewinn. Ausländische Emittenten ohne XBRL-Quartalszahlen (20-F) liefern meist nur Jahreswerte.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import requests

from common import BENCHMARKS, DATA, load_universe

UA = {"User-Agent": "systemplus-backtest research citizenmaik@gmail.com",
      "Accept-Encoding": "gzip, deflate"}
OUT = DATA / "fundamentals.parquet"

CONCEPTS = {
    "revenue": [("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                ("us-gaap", "SalesRevenueNet"), ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
                ("us-gaap", "SalesRevenueGoodsNet"), ("us-gaap", "SalesRevenueServicesNet"),
                ("us-gaap", "RevenuesNetOfInterestExpense"), ("us-gaap", "InterestAndDividendIncomeOperating"),
                ("us-gaap", "OperatingLeasesIncomeStatementLeaseRevenue"), ("us-gaap", "RealEstateRevenueNet"),
                ("us-gaap", "RevenueFromContractsWithCustomers"), ("us-gaap", "RegulatedAndUnregulatedOperatingRevenue"),
                ("us-gaap", "ElectricUtilityRevenue"), ("us-gaap", "OilAndGasRevenue"),
                ("ifrs-full", "Revenue"), ("ifrs-full", "RevenueFromContractsWithCustomers")],
    "eps": [("us-gaap", "EarningsPerShareDiluted"), ("us-gaap", "EarningsPerShareBasic"),
            ("ifrs-full", "DilutedEarningsLossPerShare")],
    "net_income": [("us-gaap", "NetIncomeLoss"), ("ifrs-full", "ProfitLoss")],
}


def get(url: str, tries: int = 4):
    for i in range(tries):
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        time.sleep(2 * (i + 1))
    r.raise_for_status()


def main() -> None:
    m = get("https://www.sec.gov/files/company_tickers.json")
    cik = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in m.values()}
    uni = [t for t in load_universe()["ticker"] if t not in BENCHMARKS]
    todo = [t for t in uni if t in cik]
    print(f"{len(uni)} Titel, davon {len(todo)} mit CIK")
    rows, miss = [], []
    for i, t in enumerate(todo, 1):
        try:
            facts = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik[t]:010d}.json")
        except Exception as e:
            print(f"  {t}: {e}")
            miss.append(t)
            continue
        time.sleep(0.12)  # SEC: max. 10 Anfragen/s
        if not facts:
            miss.append(t)
            continue
        f = facts.get("facts", {})
        for key, cands in CONCEPTS.items():
            for ns, name in cands:
                units = f.get(ns, {}).get(name, {}).get("units", {})
                for unit, vals in units.items():
                    if key != "eps" and not unit.startswith("USD") and ns == "us-gaap":
                        continue
                    for v in vals:
                        if v.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A", "20-F", "40-F", "6-K"):
                            continue
                        rows.append((t, key, name, unit, v.get("start"), v.get("end"), v.get("filed"),
                                     v.get("form"), v.get("fp"), float(v["val"])))
        if i % 250 == 0:
            print(f"  {i}/{len(todo)}")
    df = pd.DataFrame(rows, columns=["ticker", "kennzahl", "konzept", "einheit", "start", "end", "filed",
                                     "form", "fp", "wert"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.drop_duplicates(["ticker", "kennzahl", "konzept", "start", "end", "filed"])
    df.to_parquet(OUT, index=False, compression="zstd")
    (DATA / "fundamentals_info.json").write_text(json.dumps(
        {"titel_mit_cik": len(todo), "ohne_daten": len(miss), "zeilen": len(df),
         "titel_mit_umsatz": int(df.loc[df.kennzahl == "revenue", "ticker"].nunique()),
         "titel_mit_eps": int(df.loc[df.kennzahl == "eps", "ticker"].nunique())}, indent=2))
    print(open(DATA / "fundamentals_info.json").read())


if __name__ == "__main__":
    main()
