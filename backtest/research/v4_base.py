"""Basis für V4-Lockerungen: alle Breakouts (erster Close > Pivot-Oberkante, Close > SMA10)
mit erweitertem Universum (52W bis -40 %, ATR % bis 12), M2-Ergebnis, Umsatz, QQQE-Komponenten."""
import sys
import pandas as pd

SP = "/tmp/claude-0/-home-claude/35b01006-eaca-5291-968d-0197c9ef7c47/scratchpad"
sys.path.insert(0, "/home/claude/systemplus-backtest/backtest")
sys.path.insert(0, "/home/claude/systemplus-backtest/scripts")
from simulate import simulate_trades  # noqa: E402
from growth import asof_features  # noqa: E402
from common import read_store  # noqa: E402

x = pd.read_pickle(f"{SP}/x.pkl")
t3 = pd.read_pickle(f"{SP}/tr_full3.pkl")
t3 = t3[t3.ticker != "NXH"]

m = (x.entry_sig & (x.raw_close > 5) & (x.avgvol30 > 1e6) & x.atr_pct.between(2.5, 12)
     & (x.mcap.isna() | (x.mcap > 500e6)) & (x.float_pct.isna() | (x.float_pct > 70))
     & (x.off52 >= -40) & (x.ticker != "NXH"))
sig = x[m]
new = sig[~sig.index.isin(t3.index)]
print("Signale erweitert", len(sig), "davon neu", len(new))
tn = simulate_trades(x, new)
keep = ["date", "ticker", "atr_mult", "atr_pct", "off52", "sl_pct", "box_h_atr", "notbremse",
        "rs_rating", "sector_rank"]
tn = tn.join(x.loc[tn.index, [c for c in keep if c not in ("date", "ticker")]])
print("neu simuliert", len(tn))
f = pd.read_parquet("/home/claude/systemplus-backtest/data/fundamentals.parquet")
f[f.ticker.isin(tn.ticker)].to_parquet(f"{SP}/f_new.parquet")
g = asof_features(tn[["date", "ticker"]], fund_path=f"{SP}/f_new.parquet")
tn = tn.join(g)

t3 = t3.join(x.loc[t3.index, ["off52", "sl_pct", "box_h_atr", "notbremse", "rs_rating", "sector_rank"]])
cols = ["exit_date", "R", "pct", "open", "r1", "bars_held", "fills", "entry", "stop", "risk", "sl_dist",
        "ticker", "date", "atr_mult", "atr_pct", "off52", "sl_pct", "box_h_atr", "notbremse",
        "rs_rating", "sector_rank", "rev_yoy", "rev_yoy_prev", "rev_accel", "eps_yoy", "fund_age"]
b = pd.concat([t3[cols].assign(base=True), tn[cols].assign(base=False)])

# QQQE-Komponenten der Notbremse
s = read_store()
q = s[s.ticker == "QQQE"].sort_values("date").set_index("date")["close"]
ema = q.ewm(span=200, adjust=False).mean()
dd = 100 * (q / q.rolling(60).max() - 1)
s200 = q.rolling(200).mean()
nb = pd.DataFrame({"q_ema": q < ema, "q_dd": dd <= -10, "q_slope": s200 < s200.shift(20), "q_dd_val": dd})
nb.index = pd.to_datetime(nb.index)
b = b.join(nb, on="date")
b.to_pickle(f"{SP}/v4_base.pkl")
print(b.shape, b.base.sum())
print("Abgleich Notbremse:", ((b.q_ema | b.q_dd | b.q_slope) == b.notbremse).mean())
