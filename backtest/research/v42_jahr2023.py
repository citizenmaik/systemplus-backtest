"""Warum lief 2023 schwächer? Signale je Jahr, Gate-Quoten, Ergebnis mit/ohne Umsatz-Gate,
Notbremse, Investitionsgrad. Ausgabe: Konsole + backtest/out/v42_jahr2023.csv
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backtest")); sys.path.insert(0, str(ROOT / "scripts"))
import v42_report as R
from signals import build
from simulate import simulate_trades
from growth import asof_features
from pit_meta import add_pit

x = add_pit(build(start="2019-01-01", verbose=False))
mcap = x.mcap_pit.fillna(x.mcap)
uni = (x.entry_sig & (x.raw_close > 5) & (x.avgvol30 > 1e6) & (mcap.isna() | (mcap > 300e6))
       & (x.float_pct.isna() | (x.float_pct > 70)) & (x.ticker != "NXH"))
b = x[uni].copy(); b["y"] = b.date.dt.year
b["atr_ok"] = b.atr_pct.between(4.5, 10); b["w52_ok"] = b.off52.between(-30, -10)
b["w52_tief"] = b.off52 < -30; b["w52_hoch"] = b.off52 > -10
# Umsatzbeschleunigung für alle Breakouts mit ATR- und 52W-Gate
g = b[b.atr_ok & b.w52_ok].copy()
g = g.join(asof_features(g[["date", "ticker"]]))
g["rev_ok"] = g.rev_accel > 0
closes = x.pivot_table(index="date", columns="ticker", values="close")
g["since"] = R.earn_since(g, closes.index)
g["earn_ok"] = ~g.since.between(0, R.EARN_BLOCK)
sim = simulate_trades(x, g)
g = g.join(sim[["R"]])
rows = []
for y, a in b.groupby("y"):
    gy = g[g.y == y]; ok = gy[gy.rev_ok & gy.earn_ok]; no = gy[~gy.rev_ok & gy.earn_ok]
    rows.append({"Jahr": y, "Breakouts Universum": len(a), "ATR ok %": round(100 * a.atr_ok.mean()),
                 "52W ok %": round(100 * a.w52_ok.mean()), "52W > 30 % unter Hoch %": round(100 * a.w52_tief.mean()),
                 "52W < 10 % unter Hoch %": round(100 * a.w52_hoch.mean()),
                 "nach ATR+52W": len(gy), "Umsatz beschl. %": round(100 * gy.rev_ok.mean()),
                 "Umsatz unbekannt %": round(100 * gy.rev_accel.isna().mean()),
                 "Signale final": len(ok), "Ø R final (hypoth.)": round(ok.R.mean(), 2),
                 "Ø R ohne Umsatzbeschl. (hypoth.)": round(no.R.mean(), 2), "n ohne Umsatzbeschl.": len(no),
                 "Notbremse-Tage %": round(100 * x[x.ticker == x.ticker.iloc[0]].assign(y=lambda d: d.date.dt.year).query("y == @y").notbremse.mean())})
df = pd.DataFrame(rows)
D = json.load(open(ROOT / "backtest" / "out" / "v42_report.json"))
t = pd.DataFrame(D["trades"]); t["y"] = t.Entry.str[:4].astype(int)
yy = pd.DataFrame(D["years"]).rename(columns={"y": "Jahr"})
tt = t.groupby("y").agg(Trades=("Ticker", "size"), Ø_R=("Ergebnis R", "mean"), Gewinnquote=("Ergebnis R", lambda r: round(100 * (r > 0).mean())),
                        halbes_Risiko=("Risiko %", lambda r: int((r == 0.5).sum()))).reset_index().rename(columns={"y": "Jahr"})
df = df.merge(tt, on="Jahr").merge(yy[["Jahr", "ret", "qqq", "spy", "inv", "dd"]], on="Jahr")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(df.T.to_string())
m = t[t.y == 2023].copy(); m["mon"] = m.Entry.str[5:7]
print(m.groupby("mon").agg(n=("Ticker", "size"), R=("Ergebnis R", "sum"), halb=("Risiko %", lambda r: int((r == 0.5).sum()))).T.to_string())
print("Top/Flop 2023:", m.nlargest(5, "Ergebnis R")[["Ticker", "Entry", "Ergebnis R"]].values.tolist(), m.nsmallest(3, "Ergebnis R")[["Ticker", "Entry", "Ergebnis R"]].values.tolist())
df.to_csv(ROOT / "backtest" / "out" / "v42_jahr2023.csv", index=False)
