"""V4.2 mit und ohne delistete Titel (Survivorship-Test).

Voraussetzung: scripts/delisted.py und scripts/delisted_sec.py sind gelaufen (data/delisted/).

Das Skript baut eine Arbeitskopie (.delisted_work/, nicht im Git), mischt dort die delisteten
Titel in Kurse, Splits, Kohorten, Datenqualität, Quartalszahlen und Earnings-Termine und rechnet
V4.2 mit v42_report.signals/account. Engine und Regeln bleiben unverändert, das echte data/
wird nicht angefasst. Die Variante ohne delistete Titel muss die Kennzahlen des Berichts
(backtest/out/v42_report.json) treffen – das prüft das Skript mit.

Einzige Ergänzung: Ist ein delisteter Titel bei Datenende noch im Depot, wird der Rest zum
letzten Schlusskurs verkauft. Bei einer Übernahme entspricht das ungefähr der Barabfindung. Bei
einer Pleite ist es eher zu gut, weil der letzte Kurs vor dem Delisting oft noch über dem liegt,
was man tatsächlich bekommt. Ohne diese Regel bliebe das Geld bis zum Ende gebunden.

Ausgabe (nur abgeleitete Werte, keine Rohkurse):
  backtest/out/v42_delisted.csv         Kennzahlen je Variante
  backtest/out/v42_delisted_jahre.csv   Rendite je Jahr
  backtest/out/v42_delisted_trades.csv  gekaufte Trades in delisteten Titeln
Aufruf:  python backtest/research/v42_delisted.py [--keep]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".delisted_work"
DL = ROOT / "data" / "delisted"
COHORT = "delistet"


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""], comment="#", dtype={"ticker": str})


# ---------------------------------------------------------------------------
# 1. Arbeitskopie

def prepare() -> None:
    for f in ("ohlcv.parquet", "splits.csv", "tickers.csv", "fundamentals.parquet", "earnings.csv"):
        if not (DL / f).exists():
            raise SystemExit(f"data/delisted/{f} fehlt – zuerst scripts/delisted.py und scripts/delisted_sec.py")
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(ROOT, WORK, ignore=shutil.ignore_patterns(".git", WORK.name, "delisted", "__pycache__", ".venv"))
    d = WORK / "data"
    tk = _csv(DL / "tickers.csv")
    shutil.copy(DL / "tickers.csv", d / "delisted_tickers.csv")

    px = pd.read_parquet(DL / "ohlcv.parquet")
    px["date"] = pd.to_datetime(px["date"])
    (d / "ohlcv" / COHORT).mkdir(parents=True)
    for m, part in px.groupby(px["date"].dt.strftime("%Y-%m")):
        part.to_parquet(d / "ohlcv" / COHORT / f"{m}.parquet", index=False)

    sp = pd.concat([_csv(d / "splits.csv"), _csv(DL / "splits.csv")], ignore_index=True)
    sp.to_csv(d / "splits.csv", index=False)

    coh = (WORK / "universe" / "cohorts.csv")
    text = coh.read_text()
    lines = [f"{t},{COHORT},{p}" for t, p in zip(tk["ticker"], tk["provider"])]
    coh.write_text(text + ("" if text.endswith("\n") else "\n") + "\n".join(lines) + "\n")

    f = pd.concat([pd.read_parquet(d / "fundamentals.parquet"), pd.read_parquet(DL / "fundamentals.parquet")],
                  ignore_index=True)
    f.to_parquet(d / "fundamentals.parquet", index=False)
    e = pd.concat([_csv(d / "earnings.csv"), _csv(DL / "earnings.csv")], ignore_index=True)
    e.to_csv(d / "earnings.csv", index=False)

    meta = _csv(d / "meta.csv")
    add = pd.DataFrame({"ticker": tk["ticker"], "fetched": COHORT, "sector": tk.get("sector"),
                        "industry": tk.get("industry")})
    pd.concat([meta, add], ignore_index=True).to_csv(d / "meta.csv", index=False)

    # Datenqualität wie scripts/quality.py (SPY als Handelskalender)
    sys.path.insert(0, str(WORK / "scripts"))
    from common import read_splits, read_store, to_adjusted  # noqa: E402  (aus der Arbeitskopie)
    from quality import check  # noqa: E402
    s = read_store()
    s = s[s["ticker"].isin(set(tk["ticker"]) | {"SPY"})]
    q = check(to_adjusted(s, read_splits()))
    q = q[q["ticker"] != "SPY"]
    qual = _csv(d / "quality.csv")
    pd.concat([qual, q[qual.columns.intersection(q.columns)]], ignore_index=True).to_csv(d / "quality.csv", index=False)
    print(f"Arbeitskopie: {len(tk)} delistete Titel, {len(px):,} Kurszeilen, Qualität "
          f"{q['status'].value_counts().to_dict()}", flush=True)


# ---------------------------------------------------------------------------
# 2. Rechnung (läuft in der Arbeitskopie)

def close_at_delisting(tr: pd.DataFrame, x: pd.DataFrame, D: set[str]) -> pd.DataFrame:
    """Offene Positionen in delisteten Titeln am letzten Handelstag zum Schlusskurs schließen.
    R und % ändern sich nicht: m2_exit bewertet den offenen Rest bereits mit dem letzten Close."""
    xd = x[x["ticker"].isin(D)]
    if xd.empty or tr.empty:
        return tr
    last = xd.loc[xd.groupby("ticker")["date"].idxmax(), ["ticker", "date", "close"]].set_index("ticker")
    end = x["date"].max()
    tr = tr.copy()
    fills, opens, exits = [], [], []
    for k, r in tr.iterrows():
        fl = list(r["fills"])
        if r["open"] and r["ticker"] in last.index and last.at[r["ticker"], "date"] < end:
            d, px = last.at[r["ticker"], "date"], float(last.at[r["ticker"], "close"])
            rem = 1.0 - sum(f for _, f, _, _ in fl)
            if rem > 1e-9:
                fl.append((d, rem, px, "Delisting"))
            fills.append(fl); opens.append(False); exits.append(d)
        else:
            fills.append(fl); opens.append(bool(r["open"])); exits.append(r["exit_date"])
    tr["fills"], tr["open"], tr["exit_date"] = fills, opens, exits
    return tr


def run_inside(out: Path) -> None:
    sys.path.insert(0, str(ROOT / "backtest"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import v42_report as R  # noqa: E402
    from pit_meta import add_pit  # noqa: E402
    from signals import build  # noqa: E402

    tk = _csv(ROOT / "data" / "delisted_tickers.csv")
    D = set(tk["ticker"])
    x = add_pit(build(start="2019-01-01", verbose=True))
    # nur die Spalten, die signals/simulate_trades lesen – spart Speicher für die zweite Variante
    x = x[["date", "ticker", "open", "high", "low", "close", "sma10", "sma20", "entry_sig", "raw_close", "avgvol30",
           "atr_pct", "off52", "atr_mult", "notbremse", "mcap", "mcap_pit", "float_pct", "float_pit"]]
    fund = pd.read_parquet(ROOT / "data" / "fundamentals.parquet", columns=["ticker", "kennzahl"])
    rev = set(fund.loc[fund["kennzahl"] == "revenue", "ticker"])
    stocks = set(x["ticker"])
    cov = {"Delistete Titel": len(stocks & D),
           "Umsatzdaten delistete %": round(100 * len(stocks & D & rev) / max(len(stocks & D), 1), 1),
           "Umsatzdaten übrige %": round(100 * len((stocks - D) & rev) / max(len(stocks - D), 1), 1)}

    def kpis(name, c, tt, tr_all):
        r = c.pct_change().dropna()
        yrs = (c.index[-1] - c.index[0]).days / 365.25
        dl = tt[tt["ticker"].isin(D)]
        return {"Variante": name,
                "CAGR": round(100 * ((c.iloc[-1] / R.START_EQ) ** (1 / yrs) - 1), 1),
                "MaxDD": round(100 * (c / c.cummax() - 1).min(), 1),
                "Sharpe": round(r.mean() / r.std() * np.sqrt(252), 2), "Ende": round(c.iloc[-1]),
                "Trades": len(tt), "Ø R": round(tt["R"].mean(), 2),
                "Trefferquote %": round(100 * (tt["R"] > 0).mean(), 1),
                "Signale nach Gates": len(tr_all), "Signale delistet": int(tr_all["ticker"].isin(D).sum()),
                "Trades delistet": len(dl), "Ø R delistet": round(dl["R"].mean(), 2) if len(dl) else None,
                "Trefferquote delistet %": round(100 * (dl["R"] > 0).mean(), 1) if len(dl) else None,
                "Exits durch Delisting": int(sum(any(g == "Delisting" for *_, g in f) for f in dl["fills"])),
                **cov}

    rows, years, trades = [], {}, []
    for name, xx in [("ohne delistete Titel (wie Bericht)", x[~x["ticker"].isin(D)]),
                     ("mit delisteten Titeln", x)]:
        tr, closes = R.signals(x=xx)
        tr = close_at_delisting(tr, xx, D)
        c, taken, log, e = R.account(tr, closes, {}, cash_rate=R.cash_rate(closes.index))
        tt = tr.loc[[k for k, _, _ in taken]]
        rows.append(kpis(name, c, tt, tr))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        ye = c.resample("YE").last()
        yb = pd.concat([pd.Series([float(R.START_EQ)]), ye.iloc[:-1]]).to_numpy()
        years[name] = {d.year: round(100 * (v / yb[i] - 1), 1) for i, (d, v) in enumerate(ye.items())}
        if name.startswith("mit"):
            sh = {k: s for k, s, _ in taken}
            info = tk.set_index("ticker")
            for k, q in tt[tt["ticker"].isin(D)].iterrows():
                dd = info.at[q["ticker"], "delist_date"] if "delist_date" in info else None
                trades.append({"Entry": str(q["date"].date()), "Ticker": q["ticker"],
                               "Name": info.at[q["ticker"], "name"],
                               "Delisting": dd if isinstance(dd, str) else "",
                               "Grund": info.at[q["ticker"], "delist_reason"] if "delist_reason" in info else "",
                               "Letzter Kurs": info.at[q["ticker"], "last_date"],
                               "Risiko %": 0.5 if q["rw"] == 0.5 else 1.0, "Stueck": sh[k],
                               "Umsatz-Beschl. Pp": round(q["rev_accel"], 1), "Tage": int(q["bars_held"]),
                               "Exits": " ".join(f"{g} {round(100 * f)}%" for _, f, _, g in q["fills"]),
                               "Ergebnis %": round(q["pct"], 1), "Ergebnis R": round(q["R"], 2),
                               "Ergebnis USD": round(sh[k] * q["risk"] * q["R"])})

    rep = ROOT / "backtest" / "out" / "v42_report.json"
    if rep.exists():
        m = json.loads(rep.read_text())["meta"]
        b = rows[0]
        same = (b["CAGR"], b["MaxDD"], b["Sharpe"], b["Ende"], b["Trades"]) == \
               (m["cagr"], m["maxdd"], m["sharpe"], m["end"], m["trades"])
        print(f"Abgleich mit v42_report.json ({m['last']}): {'gleich' if same else 'ABWEICHUNG'} – "
              f"Bericht CAGR {m['cagr']}, MaxDD {m['maxdd']}, Sharpe {m['sharpe']}, Ende {m['end']}, Trades {m['trades']}")
        for r in rows:
            r["Abgleich Bericht"] = ("gleich" if same else "abweichend") if r is b else ""

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "v42_delisted.csv", index=False)
    pd.DataFrame(years).rename_axis("Jahr").reset_index().to_csv(out / "v42_delisted_jahre.csv", index=False)
    pd.DataFrame(trades).to_csv(out / "v42_delisted_trades.csv", index=False)
    print(f"-> {out / 'v42_delisted.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--inside", help=argparse.SUPPRESS)
    ap.add_argument("--keep", action="store_true", help="Arbeitskopie .delisted_work/ behalten")
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.inside:
        run_inside(Path(a.inside))
    else:
        # beide Schritte als eigene Prozesse, damit der Speicher des ersten wieder frei ist
        subprocess.run([sys.executable, __file__, "--prepare"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(WORK / "backtest" / "research" / "v42_delisted.py"),
                        "--inside", str(ROOT / "backtest" / "out")], cwd=WORK, check=True)
        if not a.keep:
            shutil.rmtree(WORK)
