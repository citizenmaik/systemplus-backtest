"""V4-Lockerungen testen. Aufruf: python v4_var.py <name> (Gruppe) – schreibt v4_var_<name>.csv"""
import sys
import zlib
import numpy as np
import pandas as pd

SP = "/tmp/claude-0/-home-claude/35b01006-eaca-5291-968d-0197c9ef7c47/scratchpad"
b = pd.read_pickle(f"{SP}/v4_base.pkl")
b = b[b.date <= "2026-10-23"]
b["half"] = np.where(b.ticker.map(lambda t: zlib.crc32(t.encode()) % 2) == 0, "A", "B")
closes = pd.read_pickle(f"{SP}/closes.pkl")
YEARS = (b.date.max() - pd.Timestamp("2019-01-02")).days / 365.25
NDAYS = closes.index.size


def sel(atr=(4.5, 10), w52=(10, 30), rev="accel", rev_thr=0.0, rev_missing=False,
        nb=("q_ema", "q_dd", "q_slope"), dd_thr=-10, box=3.0):
    m = b.atr_pct.between(*atr) & (b.off52 <= -w52[0]) & (b.off52 >= -w52[1]) & (b.box_h_atr <= box)
    if nb:
        on = np.zeros(len(b), bool)
        for c in nb:
            on |= (b.q_dd_val <= dd_thr).to_numpy() if c == "q_dd" else b[c].fillna(True).to_numpy()
        m &= ~on
    if rev == "accel":
        r = b.rev_accel > rev_thr
        known = b.rev_accel.notna()
    elif rev == "yoy":
        r = b.rev_yoy > rev_thr
        known = b.rev_yoy.notna()
    elif rev == "accel_or_yoy":
        r = (b.rev_accel > 0) | (b.rev_yoy >= rev_thr)
        known = b.rev_accel.notna() | b.rev_yoy.notna()
    elif rev == "accel_and_yoy":
        r = (b.rev_accel > 0) & (b.rev_yoy > rev_thr)
        known = b.rev_accel.notna()
    else:
        r = pd.Series(True, index=b.index)
        known = r
    if rev_missing:
        r = r | ~known
    return b[m & r]


def account(tr, risk_pct=1.0, cap_pct=20.0, per_day=3, start_eq=20_000):
    if "rw" not in tr:
        tr = tr.assign(rw=1.0)
    tr = tr.sort_values(["date", "rw", "atr_mult"], ascending=[True, False, True])
    by_day = {d: g for d, g in tr.groupby("date")}
    cash = eq = start_eq
    held, curve, taken = {}, [], 0
    for d in closes.index:
        for k in list(held):
            p = held[k]
            while p["fills"] and p["fills"][0][0] == d:
                _, frac, px, _ = p["fills"].pop(0)
                q = min(p["sh0"] * frac, p["sh"])
                cash += q * px
                p["sh"] -= q
            if p["sh"] <= 1e-6:
                del held[k]
        row = closes.loc[d]
        for p in held.values():
            v = row.get(p["t"])
            if pd.notna(v):
                p["last"] = v
        eq = cash + sum(p["sh"] * p["last"] for p in held.values())
        n = 0
        if d in by_day:
            for k, r in by_day[d].iterrows():
                if n >= per_day:
                    break
                if any(p["t"] == r.ticker for p in held.values()):
                    continue
                sh = np.floor(min(eq * risk_pct * r.rw / 100 / r.risk, eq * cap_pct / 100 / r.entry) * 10) / 10
                sh = min(sh, np.floor(cash / r.entry * 10) / 10)
                if sh <= 0:
                    continue
                cash -= sh * r.entry
                held[k] = dict(t=r.ticker, sh=sh, sh0=sh, fills=list(r.fills), last=r.entry)
                n += 1
                taken += 1
        curve.append((d, cash + sum(p["sh"] * p["last"] for p in held.values()), len(held)))
    c = pd.DataFrame(curve, columns=["date", "eq", "pos"]).set_index("date")["eq"]
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    yr = c.resample("YE").last()
    yr = yr / pd.concat([pd.Series([start_eq]), yr.iloc[:-1]]).to_numpy() - 1
    return dict(cagr=100 * ((c.iloc[-1] / start_eq) ** (1 / yrs) - 1), dd=100 * (c / c.cummax() - 1).min(),
                taken=taken, worst_yr=100 * yr.min(), loss_yrs=int((yr < 0).sum()))


def row(name, t):
    c = t[~t.open]
    w = c.R > 0
    days = t.groupby("date").size()
    a20 = account(t, 1, 20, PER_DAY)
    a10 = account(t, 1, 10, PER_DAY)
    return dict(Variante=name, Signale=len(t), Sig_Jahr=round(len(t) / YEARS),
                Tage_mit_Signal=round(100 * len(days) / NDAYS, 1),
                Avg_R=round(c.R.mean(), 3), Med_R=round(c.R.median(), 2), Win=round(100 * w.mean(), 1),
                PF=round(c.R[w].sum() / -c.R[~w].sum(), 2),
                R_A=round(c[c.half == "A"].R.mean(), 3), R_B=round(c[c.half == "B"].R.mean(), 3),
                R_19_22=round(c[c.date < "2023"].R.mean(), 3), R_23_26=round(c[c.date >= "2023"].R.mean(), 3),
                Konto20_CAGR=round(a20["cagr"], 1), Konto20_DD=round(a20["dd"], 1),
                Konto20_Trades_Jahr=round(a20["taken"] / YEARS), Konto20_schlechtestes_Jahr=round(a20["worst_yr"], 1),
                Konto10_CAGR=round(a10["cagr"], 1), Konto10_DD=round(a10["dd"], 1))


PER_DAY = 3
NB3 = ("q_ema", "q_dd", "q_slope")
GROUPS = {
    "basis": {"V4 (Basis)": {}, "V3 (ohne Umsatz)": dict(rev=None)},
    "atr": {f"ATR % ab {a}": dict(atr=(a, 10)) for a in (4.0, 3.5, 3.0, 2.5)} | {
        "ATR % 4,5-12": dict(atr=(4.5, 12))},
    "w52": {f"52W {lo}-{hi} %": dict(w52=(lo, hi)) for lo, hi in
            [(5, 30), (0, 30), (10, 35), (10, 40), (5, 40), (15, 30), (7.5, 30)]},
    "umsatz": {"Beschl. > -5 Pp": dict(rev_thr=-5), "Beschl. > -10 Pp": dict(rev_thr=-10),
               "Beschl. oder fehlend": dict(rev_missing=True),
               "Umsatz YoY > 0": dict(rev="yoy", rev_thr=0), "Umsatz YoY > 10": dict(rev="yoy", rev_thr=10),
               "Umsatz YoY > 20": dict(rev="yoy", rev_thr=20),
               "Beschl. oder YoY >= 25": dict(rev="accel_or_yoy", rev_thr=25),
               "Beschl. oder YoY >= 40": dict(rev="accel_or_yoy", rev_thr=40)},
    "notbremse": {"NB nur EMA200": dict(nb=("q_ema",)), "NB ohne SMA-Steigung": dict(nb=("q_ema", "q_dd")),
                  "NB ohne DD60": dict(nb=("q_ema", "q_slope")), "NB nur SMA-Steigung": dict(nb=("q_slope",)),
                  "NB DD60 ab -15 %": dict(dd_thr=-15), "NB aus": dict(nb=())},
    "box": {"Box bis 2,5 ATR": dict(box=2.5), "Box bis 2,0 ATR": dict(box=2.0)},
}

if __name__ == "__main__":
    g = sys.argv[1]
    if g == "combo":
        import json
        variants = json.loads(open(f"{SP}/combos.json").read())
        variants = {k: {kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}
                    for k, v in variants.items()}
    else:
        variants = GROUPS[g]
    if len(sys.argv) > 2:
        PER_DAY = int(sys.argv[2])
    rows = []
    for name, kw in variants.items():
        kw = dict(kw)
        tier2 = kw.pop("tier2", None)
        t = sel(**kw)
        if tier2:
            t2 = sel(**tier2)
            t2 = t2[~t2.index.isin(t.index)].assign(rw=0.5)
            t = pd.concat([t.assign(rw=1.0), t2])
        rows.append(row(name, t))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{SP}/v4_var_{g}.csv", index=False)
