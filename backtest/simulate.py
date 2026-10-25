"""System+ v3.8 – Exits nach M2 und Kontosimulation.

M2 (wie f_step im Indikator, Modell-Index 1), ab dem Tag nach dem Entry:
  1. Tief ≤ SL (SMA10 vom Entry-Tag)   → Rest raus zu min(Open, SL)
  2. Hoch ≥ Entry + 1R                 → 1R erreicht
  3. 1R nicht erreicht und Close < SMA10 → 100 % raus zum Close (Fail)
  4. sonst: Close < SMA10 und TP1 offen → 40 % raus (TP1); Close < SMA20 → Rest raus (TP2)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def m2_exit(bars: dict, i0: int) -> dict:
    """bars: Arrays eines Tickers (open, high, low, close, sma10, sma20, date). i0 = Entry-Index."""
    o, h, l, c, s10, s20, d = (bars[k] for k in ("open", "high", "low", "close", "sma10", "sma20", "date"))
    entry, stop = c[i0], s10[i0]
    risk = entry - stop
    rem, r_sum, pct_sum = 1.0, 0.0, 0.0
    r1, tp1 = False, False
    fills = []

    def fill(i, frac, px, tag):
        nonlocal rem, r_sum, pct_sum
        f = min(frac, rem)
        if f > 0:
            rem -= f
            r_sum += f * (px - entry) / risk
            pct_sum += f * 100 * (px - entry) / entry
            fills.append((d[i], f, px, tag))

    last = i0
    for i in range(i0 + 1, len(c)):
        last = i
        if l[i] <= stop:
            fill(i, rem, min(o[i], stop), "SL")
        else:
            if not r1 and h[i] >= entry + risk:
                r1 = True
            if not r1 and c[i] < s10[i]:
                fill(i, rem, c[i], "Fail")
            else:
                if not tp1 and c[i] < s10[i]:
                    fill(i, 0.4, c[i], "TP1")
                    tp1 = True
                if c[i] < s20[i]:
                    fill(i, rem, c[i], "TP2")
        if rem <= 1e-9:
            break
    open_ = rem > 1e-9
    if open_:  # offen: mit letztem Close bewertet
        r_open = rem * (c[last] - entry) / risk
        pct_open = rem * 100 * (c[last] - entry) / entry
    else:
        r_open = pct_open = 0.0
    return dict(exit_date=d[last], R=r_sum + r_open, pct=pct_sum + pct_open, open=open_,
                r1=r1, bars_held=last - i0, fills=fills, entry=entry, stop=stop, risk=risk,
                sl_dist=100 * risk / entry)


def simulate_trades(x: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """Simuliert jede Signalzeile (Index in x) einzeln nach M2."""
    out = []
    pos = {t: g.index.to_numpy() for t, g in x.groupby("ticker", sort=False)}
    for t, g in sig.groupby("ticker", sort=False):
        ix = pos[t]
        sub = x.loc[ix]
        bars = {k: sub[k].to_numpy() for k in ("open", "high", "low", "close", "sma10", "sma20", "date")}
        loc = {v: n for n, v in enumerate(ix)}
        for r in g.itertuples():
            res = m2_exit(bars, loc[r.Index])
            res.update(idx=r.Index, ticker=t, date=r.date)
            out.append(res)
    return pd.DataFrame(out).set_index("idx")


def no_overlap(tr: pd.DataFrame) -> pd.Series:
    """Kein neuer Entry, solange im selben Titel eine Position offen ist."""
    keep = pd.Series(False, index=tr.index)
    for _, g in tr.sort_values("date").groupby("ticker"):
        busy_until = pd.Timestamp.min
        for i, r in g.iterrows():
            if r["date"] > busy_until:
                keep[i] = True
                busy_until = r["exit_date"] if not r["open"] else pd.Timestamp.max
    return keep


def stats(tr: pd.DataFrame) -> dict:
    t = tr[~tr["open"]]
    if t.empty:
        return dict(n=0)
    win = t["R"] > 0
    pos, neg = t.loc[win, "R"].sum(), -t.loc[~win, "R"].sum()
    return dict(n=len(t), win=win.mean(), avgR=t["R"].mean(), medR=t["R"].median(),
                PF=pos / neg if neg > 0 else np.inf, sl=t["sl_dist"].mean(),
                gain=t.loc[win, "pct"].mean(), loss=t.loc[~win, "pct"].mean(),
                totR=t["R"].sum(), days=t["bars_held"].mean())


# ---------------------------------------------------------------------------
def account(x: pd.DataFrame, tr: pd.DataFrame, start_eq: float = 20_000, risk_pct: float = 1.0,
            cap_pct: float = 10.0, per_day: int = 1) -> tuple[pd.Series, pd.DataFrame]:
    """Konto mit täglicher Bewertung. Entries zum Close, sortiert nach niedrigstem ATR-Multiple,
    höchstens per_day Neukäufe pro Tag. Exits gemäß den M2-Fills des Trades."""
    tr = tr.sort_values(["date", "atr_mult"])
    closes = x.pivot_table(index="date", columns="ticker", values="close")
    dates = closes.index
    by_day = {d: g for d, g in tr.groupby("date")}
    fills_by_day: dict = {}
    cash, eq = start_eq, start_eq
    held: dict = {}       # trade_idx -> dict(ticker, shares, fills)
    taken = []
    curve = []
    for d in dates:
        # Exits des Tages
        for k in list(held):
            p = held[k]
            while p["fills"] and p["fills"][0][0] == d:
                _, frac, px, _ = p["fills"].pop(0)
                q = p["shares0"] * frac
                q = min(q, p["shares"])
                cash += q * px
                p["shares"] -= q
            if p["shares"] <= 1e-6:
                del held[k]
        # Bewertung vor neuen Käufen
        row = closes.loc[d]
        mv = sum(p["shares"] * (row.get(p["ticker"]) if pd.notna(row.get(p["ticker"])) else p["last"])
                 for p in held.values())
        for p in held.values():
            v = row.get(p["ticker"])
            if pd.notna(v):
                p["last"] = v
        eq = cash + mv
        # Entries
        n_new = 0
        if d in by_day:
            for k, r in by_day[d].iterrows():
                if n_new >= per_day:
                    break
                if any(p["ticker"] == r["ticker"] for p in held.values()):
                    continue
                sh = np.floor(min(eq * risk_pct / 100 / r["risk"], eq * cap_pct / 100 / r["entry"]) * 10) / 10
                sh = min(sh, np.floor(cash / r["entry"] * 10) / 10)
                if sh <= 0:
                    continue
                cash -= sh * r["entry"]
                held[k] = dict(ticker=r["ticker"], shares=sh, shares0=sh, fills=list(r["fills"]),
                               open=r["open"], exit_date=r["exit_date"], last=r["entry"])
                taken.append(dict(idx=k, shares=sh, value=sh * r["entry"], eq=eq,
                                  usd_risk=sh * r["risk"], pnl=sh * r["risk"] * r["R"]))
                n_new += 1
        mv = sum(p["shares"] * p["last"] for p in held.values())
        curve.append((d, cash + mv, len(held)))
    c = pd.DataFrame(curve, columns=["date", "equity", "positions"]).set_index("date")
    return c, pd.DataFrame(taken)


def acc_stats(curve: pd.DataFrame) -> dict:
    e = curve["equity"]
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1
    dd = (e / e.cummax() - 1).min()
    return dict(cagr=cagr, maxdd=dd, end=e.iloc[-1], avg_pos=curve["positions"].mean())
