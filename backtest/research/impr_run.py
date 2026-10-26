import sys, pickle
GRP = sys.argv[1]
exec(open('impr.py').read())
grp = GRP
T = {"std": t}
if grp == "exit":
    T["L3"] = resim(3); T["L5"] = resim(5); T["L3_30"] = resim(3, "sma20")  # Kontrolle
    V = {"Exit: Rest an SMA50 ab 3R": ("L3", {}), "Exit: Rest an SMA50 ab 5R": ("L5", {})}
else:
    V = {"Basis": ("std", {}),
         "max. 8 Positionen": ("std", dict(max_pos=8)), "max. 6 Positionen": ("std", dict(max_pos=6)),
         "max. 5 Positionen": ("std", dict(max_pos=5)), "max. 2 je Sektor": ("std", dict(max_sec=2)),
         "max. 6 Pos. + 2 je Sektor": ("std", dict(max_pos=6, max_sec=2)),
         "DD-Bremse -10 %": ("std", dict(dd_brake=-10)), "DD-Bremse -15 %": ("std", dict(dd_brake=-15))}
rows = []
for name, (tk, kw) in V.items():
    res = {"Variante": name}
    c, n = run(T[tk], **kw); res.update(st(c, n))
    for h in ("A", "B"):
        tt = T[tk][T[tk].half == h]
        ch, nh = run(tt, **kw); s2 = st(ch, nh)
        res[f"{h}_CAGR"] = s2["CAGR"]; res[f"{h}_DD"] = s2["MaxDD"]; res[f"{h}_Sharpe"] = s2["Sharpe"]
    rows.append(res); print(res, flush=True)
pd.DataFrame(rows).to_csv(f"impr_{grp}.csv", index=False)
