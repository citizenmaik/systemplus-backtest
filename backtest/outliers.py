"""Was haben die großen Gewinner am Entry-Tag gemeinsam? (Gesamtmarkt, 25.10.2026)

Vorgehen gegen Data-Mining:
  * Alle Pivot-Breakouts im Screener-Universum (Close > SMA10), Exits nach M2.
  * Ticker per CRC32 in Hälfte A (Suche) und B (Prüfung) geteilt; B zusätzlich nach Zeit.
  * Schwellen nur aus A, Prüfung nur auf B.
  * Danach wenige, vorab festgelegte Varianten im Konto (1 Neukauf/Tag, 1 % Risiko).

    python backtest/outliers.py
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

import signals
from features import box_age, build_features
from simulate import account, acc_stats, no_overlap, simulate_trades, stats

FEATURES = ["perf_1m", "perf_3m", "perf_6m", "perf_12m", "rs_rating", "dist_sma50", "dist_sma200",
            "sma200_slope", "vol_ratio", "dollar_vol", "close_loc", "gap_pct", "day_ret", "brk_atr",
            "box_h_atr", "box_age", "box_depth", "atr_pct", "atr_mult", "sl_pct", "off52", "raw_close",
            "mcap_m", "bar_no", "qqqe_dist_ema200", "qqqe_dd60", "sector_rank"]


def main():
    x = signals.build(start="2019-01-01")
    x["box_age"] = box_age(x)
    sig = x[x["entry_sig"] & x["g_screen"]]
    tr = simulate_trades(x, sig)
    F = build_features(x.loc[sig.index].reset_index()).set_index("index")
    D = tr.join(F.drop(columns=["date", "ticker"]), rsuffix="_f").replace([np.inf, -np.inf], np.nan)
    D = D[(~D["open"]) & (D["date"] <= "2026-10-09")].copy()
    D["half"] = np.where(D["ticker"].map(lambda t: zlib.crc32(t.encode()) % 2) == 0, "A", "B")
    D["contrib"] = np.minimum(100 / D["sl_dist"], 10) / 100 * D["pct"]
    D["big"] = D["pct"] >= 30

    A, B = D[D["half"] == "A"], D[D["half"] == "B"]
    rows = []
    for f in FEATURES:
        for d in ("hi", "lo"):
            thr = A[f].quantile(0.8 if d == "hi" else 0.2)
            row = dict(Merkmal=f, Richtung=d, Schwelle_A=round(thr, 2))
            for nm, Z in [("A", A), ("B19_22", B[B["date"] < "2023"]), ("B23_26", B[B["date"] >= "2023"])]:
                Z = Z.dropna(subset=[f])
                k = Z[f] >= thr if d == "hi" else Z[f] <= thr
                row[f"{nm}_R_drin"] = round(Z.loc[k, "R"].mean(), 3)
                row[f"{nm}_R_rest"] = round(Z.loc[~k, "R"].mean(), 3)
            rows.append(row)
    pd.DataFrame(rows).to_csv("out/merkmale.csv", index=False)

    tr = tr.join(sig[["atr_pct", "atr_mult", "g_cb", "g_atr", "g_sl", "g_52w", "g_lead"]])
    hv = tr["atr_pct"] >= 4.5
    V = {"V0 v3.8": tr.g_cb & tr.g_atr & tr.g_sl & tr.g_52w & tr.g_lead,
         "V1 ATR% >= 4,5": hv,
         "V2 ATR% >= 4,5 + 52W": hv & tr.g_52w,
         "V3 V2 + Notbremse": hv & tr.g_52w & tr.g_cb,
         "V4 v3.8 + ATR% >= 4,5": tr.g_cb & tr.g_atr & tr.g_sl & tr.g_52w & tr.g_lead & hv}
    xs = x[["date", "ticker", "close"]]
    out = []
    for k, m in V.items():
        for excl in ([], ["NXH"]):  # NXH: Kürzel neu vergeben, Kursdaten 2020 fraglich
            t = tr[m & ~tr["ticker"].isin(excl)]
            s = stats(t[no_overlap(t)])
            for cap in (10, 20):
                cur, tk = account(xs, t, cap_pct=cap)
                a = acc_stats(cur)
                out.append(dict(Variante=k, ohne=",".join(excl), Deckel=cap, Trades=s["n"],
                                OR=round(s["avgR"], 2), PF=round(s["PF"], 2),
                                CAGR=round(a["cagr"], 3), MaxDD=round(a["maxdd"], 3)))
    O = pd.DataFrame(out)
    O.to_csv("out/varianten.csv", index=False)
    print(O.to_string(index=False))


if __name__ == "__main__":
    main()
