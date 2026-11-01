"""System+ V4.2 – Backtest-Bericht für die Seite „System+ V4.2 Trades“.

Regeln V4.2 (Stand 27.10.2026):
  Universum  Kurs > 5, Ø-Volumen 30 T > 1 Mio, Market Cap > 300 Mio (oder unbekannt), Float > 70 % (oder unbekannt)
  Setup      Pivot-Box 3/3, Höhe ≤ 3 ATR, erster Close > Oberkante und > SMA10 (entry_sig)
  Gates      ATR % 4,5–10, 52W 10–30 % unter Hoch, Umsatz-YoY letztes gemeldetes Quartal > Quartal davor,
             kein Entry 0–5 Handelstage nach Earnings (Tag 0 = Reaktionstag, wie sinceEarn im Pine-Skript;
             seit 31.10.2026, Varianten: out/v42_earnings_varianten.csv)
  Notbremse  QQQE aktiv → 0,5 % statt 1 % Risiko
  Depot      Bestand am letzten Tag (Stück, Kurs, Werte, offener G/V) unter portfolio im JSON
  Konto      1 % Risiko, 15 % Deckel, max. 3 Käufe/Tag nach niedrigstem ATR-Multiple, eine Position je Titel, Start 10.000
  Exits M2   SL = SMA10 am Entry-Tag; vor 1R Close < SMA10 alles; ab 1R Close < SMA10 40 %, Close < SMA20 Rest
             Close < SMA20 gilt auch vor 1R (SMA20 über SMA10 → dort alles raus), Varianten: out/v42_sma20_varianten.csv

Ausgabe: backtest/out/v42_report.json (Trades, Jahre, Wochenkurve vs. QQQ/SPY, nicht gekaufte Signale, Kennzahlen).
Aufruf:  python backtest/v42_report.py
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(ROOT / "scripts"))
from signals import build  # noqa: E402
from simulate import simulate_trades  # noqa: E402
from growth import asof_features  # noqa: E402
from common import read_store  # noqa: E402

START_EQ, RISK, CAP, PER_DAY = 10_000, 1.0, 15.0, 3
EARN_BLOCK = 5  # kein Entry an Handelstag 0..5 nach Earnings
OUT = ROOT / "backtest" / "out" / "v42_report.json"


def earn_since(tr: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.Series:
    """Handelstage seit dem letzten Earnings-Reaktionstag (0 = Reaktionstag) je Trade, NaN ohne Daten.
    Reaktionstag: Meldung vor 12 Uhr = selber Handelstag, sonst nächster Handelstag."""
    E = pd.read_csv(ROOT / "data" / "earnings.csv", keep_default_na=False, na_values=[""])
    E = E[E["date"].notna()].copy()
    E["d"] = pd.to_datetime(E["date"])
    E["hh"] = E["time"].fillna("16:00").str.slice(0, 2).replace("", "16").astype(int)
    ri = cal.searchsorted(E["d"].to_numpy())
    same = (ri < len(cal)) & (cal[np.minimum(ri, len(cal) - 1)] == E["d"].to_numpy())
    E["ri"] = ri + (same & (E["hh"].to_numpy() >= 12))
    ev = {t: np.sort(g["ri"].unique()) for t, g in E.groupby("ticker")}
    pos = {d: i for i, d in enumerate(cal)}
    out = []
    for t, d in zip(tr["ticker"], tr["date"]):
        e, i0 = ev.get(t), pos[d]
        pv = e[e <= i0] if e is not None else []
        out.append(i0 - pv[-1] if len(pv) else np.nan)
    return pd.Series(out, index=tr.index)


def signals(earn_block: int | None = EARN_BLOCK, float_days: int | None = None,
            x: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """float_days: Float-Filter (heutiger Stand aus meta.csv) nur in den letzten x Kalendertagen;
    None = ganze Historie, 0 = aus. x: bereits gebautes build()-Ergebnis (spart Zeit in Varianten)."""
    if x is None:
        x = build(start="2019-01-01", verbose=True)
    flt_ok = x.float_pct.isna() | (x.float_pct > 70)
    if float_days is not None:
        flt_ok = flt_ok | (x.date < x.date.max() - pd.Timedelta(days=float_days))
    m = (x.entry_sig & (x.raw_close > 5) & (x.avgvol30 > 1e6) & x.atr_pct.between(4.5, 10)
         & (x.mcap.isna() | (x.mcap > 300e6)) & flt_ok
         & x.off52.between(-30, -10) & (x.ticker != "NXH"))
    sig = x[m]
    tr = simulate_trades(x, sig)
    tr = tr.join(x.loc[tr.index, ["atr_mult", "atr_pct", "off52", "notbremse"]])
    tr = tr.join(asof_features(tr[["date", "ticker"]]))
    tr = tr[tr.rev_accel > 0].copy()
    tr["rw"] = np.where(tr.notbremse, 0.5, 1.0)
    closes = x.pivot_table(index="date", columns="ticker", values="close")
    tr["since"] = earn_since(tr, closes.index)
    if earn_block is not None:
        tr = tr[~tr["since"].between(0, earn_block)].copy()
    return tr, closes


def account(tr: pd.DataFrame, closes: pd.DataFrame, state: dict | None = None,
            lev: float = 1.0, rate: pd.Series | None = None):
    """lev: max. Positionswert / Konto (1.0 = ohne Margin). rate: Jahreszins % je Tag für negatives Cash."""
    tr = tr.sort_values(["date", "rw", "atr_mult"], ascending=[True, False, True])
    by_day = {d: g for d, g in tr.groupby("date")}
    cash = START_EQ; held, curve, taken, log, expo = {}, [], [], [], []
    interest = 0.0
    for d in closes.index:
        if cash < 0 and rate is not None:
            z = cash * float(rate.get(d, 0.0)) / 100 / 252
            cash += z; interest += z
        for k in list(held):
            p = held[k]
            while p["fills"] and pd.Timestamp(p["fills"][0][0]) == d:
                _, frac, px, _ = p["fills"].pop(0)
                q = min(p["sh0"] * frac, p["sh"]); cash += q * px; p["sh"] -= q
            if p["sh"] <= 1e-6:
                del held[k]
        row = closes.loc[d]
        for p in held.values():
            v = row.get(p["t"])
            if pd.notna(v):
                p["last"] = v
        eq = cash + sum(p["sh"] * p["last"] for p in held.values())
        n = 0
        for k, r in (by_day[d].iterrows() if d in by_day else []):
            if n >= PER_DAY:
                log.append((k, "max. 3 pro Tag", eq, cash, 0)); continue
            if any(p["t"] == r.ticker for p in held.values()):
                log.append((k, "schon im Depot", eq, cash, 0)); continue
            sh_t = np.floor(min(eq * RISK * r.rw / 100 / r.risk, eq * CAP / 100 / r.entry) * 10) / 10
            sh = min(sh_t, np.floor(max(cash + (lev - 1) * eq, 0) / r.entry * 10) / 10)
            if sh <= 0:
                log.append((k, "kein Cash", eq, cash, sh_t)); continue
            if sh < sh_t:
                log.append((k, "gekauft, gekürzt", eq, cash, sh_t))
            cash -= sh * r.entry
            held[k] = dict(t=r.ticker, sh=sh, sh0=sh, fills=list(r.fills), last=r.entry); n += 1
            taken.append((k, sh, eq))
            p = held[k]
            while p["fills"] and pd.Timestamp(p["fills"][0][0]) == d:
                _, frac, px, _ = p["fills"].pop(0)
                q = min(p["sh0"] * frac, p["sh"]); cash += q * px; p["sh"] -= q
            if p["sh"] <= 1e-6:
                del held[k]
        mv = sum(p["sh"] * p["last"] for p in held.values())
        expo.append((d, mv / (cash + mv)))
        curve.append((d, cash + mv))
    if state is not None:  # Depot am letzten Tag (für den Bericht)
        state.update(cash=cash, held={k: dict(p) for k, p in held.items()}, interest=interest)
    c = pd.DataFrame(curve, columns=["date", "eq"]).set_index("date")["eq"]
    e = pd.DataFrame(expo, columns=["date", "exp"]).set_index("date")["exp"]
    return c, taken, log, e


def main() -> None:
    tr, closes = signals()
    st = {}
    c, taken, log, e = account(tr, closes, st)
    eq_last = float(c.iloc[-1])
    pos = []
    for k, p in st["held"].items():
        q = tr.loc[k]
        real = sum(f * p["sh0"] * (px - q.entry) for d, f, px, _ in q.fills if pd.Timestamp(d) <= c.index[-1])
        pos.append({"Ticker": q.ticker, "Entry": str(q.date.date()), "Entry-Kurs": round(q.entry, 2), "SL": round(q.stop, 2),
                    "Kurs": round(p["last"], 2), "Stueck": round(p["sh"], 1), "Stueck Start": round(p["sh0"], 1),
                    "Einstand USD": round(p["sh"] * q.entry), "Wert USD": round(p["sh"] * p["last"]),
                    "GV offen USD": round(p["sh"] * (p["last"] - q.entry)), "GV offen %": round(100 * (p["last"] / q.entry - 1), 1),
                    "Realisiert USD": round(real), "Anteil %": round(100 * p["sh"] * p["last"] / eq_last, 1),
                    "Bis SL USD": round(p["sh"] * (q.stop - p["last"]))})
    pos.sort(key=lambda r: r["Entry"])
    portfolio = dict(date=str(c.index[-1].date()), cash=round(st["cash"]), equity=round(eq_last), positions=pos)
    rows = []
    for k, sh, eq in taken:
        q = tr.loc[k]
        ex = pd.Timestamp(q.fills[-1][0]).date() if (not q.open and len(q.fills)) else None
        rows.append({"Entry": str(q.date.date()), "Ticker": q.ticker, "Risiko %": 0.5 if q.rw == 0.5 else 1.0,
                     "Entry-Kurs": round(q.entry, 2), "SL": round(q.stop, 2), "SL-Abstand %": round(q.sl_dist, 1),
                     "ATR %": round(q.atr_pct, 1), "52W %": round(q.off52, 1), "Umsatz-Beschl. Pp": round(q.rev_accel, 1),
                     "Stueck": sh, "Einsatz USD": round(sh * q.entry), "Konto bei Entry": round(eq),
                     "Exit (letzter)": str(ex) if ex else "", "Status": "offen" if q.open else "geschlossen",
                     "Tage": int(q.bars_held), "1R erreicht": "ja" if q.r1 else "nein",
                     "Exits": " ".join(f"{g} {round(100 * f)}% @{p:.2f}" for _, f, p, g in q.fills),
                     "Ergebnis %": round(q.pct, 1), "Ergebnis R": round(q.R, 2), "Ergebnis USD": round(sh * q.risk * q.R)})
    s = read_store(); s = s[s.ticker.isin(["SPY", "QQQ"])]; s["date"] = pd.to_datetime(s["date"])
    px = s.pivot(index="date", columns="ticker", values="close").reindex(c.index).ffill()
    cur = pd.DataFrame({"S": c, "QQQ": START_EQ * px.QQQ / px.QQQ.iloc[0], "SPY": START_EQ * px.SPY / px.SPY.iloc[0]})
    w = cur.resample("W-FRI").last(); w.iloc[0] = cur.iloc[0]
    if w.index[-1] != cur.index[-1]:
        w.loc[cur.index[-1]] = cur.iloc[-1]
    ye = c.resample("YE").last(); yb = pd.concat([pd.Series([float(START_EQ)]), ye.iloc[:-1]]).to_numpy()
    qy, sy = cur.QQQ.resample("YE").last(), cur.SPY.resample("YE").last()
    years = []
    for i, (dt, v) in enumerate(ye.items()):
        sub = c[c.index.year == dt.year]
        years.append(dict(y=dt.year, ret=round(100 * (v / yb[i] - 1), 1), end=round(v),
                          dd=round(100 * (sub / c[c.index <= dt].cummax()[sub.index] - 1).min(), 1),
                          qqq=round(100 * (qy.iloc[i] / (START_EQ if i == 0 else qy.iloc[i - 1]) - 1), 1),
                          spy=round(100 * (sy.iloc[i] / (START_EQ if i == 0 else sy.iloc[i - 1]) - 1), 1),
                          inv=round(100 * e[e.index.year == dt.year].mean())))
    L = pd.DataFrame(log, columns=["k", "grund", "eq", "cash", "sh_t"]).join(
        tr[["date", "ticker", "entry", "stop", "R", "pct", "open", "atr_mult", "atr_pct", "off52", "rev_accel", "rw"]], on="k")
    skips = [{"Datum": str(r.date.date()), "Ticker": r.ticker, "Grund": r.grund, "Risiko %": 0.5 if r.rw == 0.5 else 1.0,
              "Entry-Kurs": round(r.entry, 2), "SL": round(r.stop, 2), "ATR-Multiple": round(r.atr_mult, 2),
              "ATR %": round(r.atr_pct, 1), "52W %": round(r.off52, 1), "Umsatz-Beschl. Pp": round(r.rev_accel, 1),
              "Freies Cash": round(r.cash), "Soll-Einsatz": None if r.sh_t == 0 else round(r.sh_t * r.entry),
              "Ergebnis R (hypoth.)": round(r.R, 2), "Ergebnis % (hypoth.)": round(r.pct, 1),
              "Status": "offen" if r.open else "geschlossen"} for r in L.itertuples()]
    r = c.pct_change().dropna()
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    meta = dict(first=str(c.index[0].date()), last=str(c.index[-1].date()),
                cagr=round(100 * ((c.iloc[-1] / START_EQ) ** (1 / yrs) - 1), 1), maxdd=round(100 * (c / c.cummax() - 1).min(), 1),
                sharpe=round(r.mean() / r.std() * np.sqrt(252), 2), end=round(c.iloc[-1]), trades=len(rows),
                generated=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    OUT.write_text(json.dumps(dict(meta=meta, portfolio=portfolio, trades=rows, years=years, skips=skips,
                                   curve=dict(x=[d.strftime("%Y-%m-%d") for d in w.index], s=[round(v) for v in w.S],
                                              q=[round(v) for v in w.QQQ], p=[round(v) for v in w.SPY])),
                              ensure_ascii=False, default=lambda o: None if pd.isna(o) else str(o)))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
