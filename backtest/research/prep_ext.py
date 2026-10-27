import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/systemplus-backtest/backtest")
from growth import quarterly
b = pd.read_pickle("v4_base_fix.pkl")
x = pd.read_pickle("x.pkl")[["raw_close","avgvol30","mcap","float_pct"]]
b = b.join(x)
b["dollarvol"] = b.raw_close * b.avgvol30
# QoQ-Umsatz point-in-time
f = pd.read_parquet("/home/claude/systemplus-backtest/data/fundamentals.parquet")
f = f[(f.kennzahl == "revenue") & f.ticker.isin(b.ticker.unique())]
q = quarterly(f).sort_values(["ticker","end"])
q["prevq"] = q.groupby("ticker").wert.shift(1)
gap = (q.end - q.groupby("ticker").end.shift(1)).dt.days
q["qoq"] = np.where((gap.between(80,100)) & (q.prevq.abs() > 1e5), (q.wert/q.prevq - 1)*100, np.nan)
q["known"] = (q.filed + pd.Timedelta(days=1)).astype("datetime64[ns]")
q = q.sort_values(["ticker","known","end"]).drop_duplicates(["ticker","known"], keep="last")
q["qoq_prev"] = q.groupby("ticker").qoq.shift(1)
tt = b[["date","ticker"]].copy(); tt["date"] = pd.to_datetime(tt.date).astype("datetime64[ns]"); tt["_i"] = tt.index
m = pd.merge_asof(tt.sort_values("date"), q[["ticker","known","qoq","qoq_prev"]].sort_values("known"),
                  left_on="date", right_on="known", by="ticker", direction="backward", tolerance=pd.Timedelta(days=200)).set_index("_i")
b["qoq"] = m.qoq; b["qoq_prev"] = m.qoq_prev; b["qoq_accel"] = m.qoq - m.qoq_prev
b.to_pickle("v4_base_ext.pkl")
print(b[["raw_close","avgvol30","mcap","dollarvol","qoq","qoq_accel"]].describe().round(1).to_string())
