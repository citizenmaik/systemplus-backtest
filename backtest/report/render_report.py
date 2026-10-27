"""Füllt die Vorlage v42_trades.tpl.html mit backtest/out/v42_report.json → backtest/report/v42_trades.html.

Aufruf: python backtest/report/render_report.py
Die fertige Datei wird danach als Seite „System+ V4.2 Trades“ veröffentlicht
(https://claude.ai/artifact/Wtcef27VLJnNBvbHuqVsQb).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
data = json.loads((ROOT / "backtest" / "out" / "v42_report.json").read_text())
m = data["meta"]
last_de = f'{m["last"][8:10]}.{m["last"][5:7]}.{m["last"][:4]}'
maxdd = f'{m["maxdd"]:.1f}'.replace(".", ",").replace("-", "−")
html = (HERE / "v42_trades.tpl.html").read_text()
html = (html.replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__LAST_DE__", last_de).replace("__LAST_YM__", m["last"][:7]).replace("__MAXDD__", maxdd))
html = html.replace("Universum: heute gelistete Titel", f"Berechnet {m['generated']}. Universum: heute gelistete Titel")
out = HERE / "v42_trades.html"
out.write_text(html)
print(out, "| Stand", last_de, "| Trades", m["trades"], "| CAGR", m["cagr"], "| MaxDD", m["maxdd"], "| Ende", m["end"])
