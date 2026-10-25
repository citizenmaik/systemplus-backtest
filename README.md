# systemplus-backtest

Tageskerzen für den System+-Backtest: alle US-Aktien an Nasdaq und NYSE, die seit 2018
die Grundfilter erfüllt haben – automatisch per GitHub Actions und yfinance.

## Was passiert wann

| Lauf | Zeit (UTC) | Inhalt |
|---|---|---|
| Mo–Fr | 22:17 | letzter Monat für alle Universum-Titel, anhängen |
| Sonntag | 06:43 | Vollaufbau: anhängen wie Mo–Fr, dazu Kandidatenliste neu, neue Titel mit Historie ab 2018 aufnehmen, Sektor/Float |
| manuell | – | Actions → update-data → Run workflow → `full` oder `incremental` |

Ist noch kein Datensatz da, macht auch der tägliche Lauf einen Vollaufbau (Kurse ca. 10 Min., Metadaten ca. 30 Min.).

## Universum

1. Kandidaten: alle Stammaktien aus den Nasdaq-Trader-Symbolverzeichnissen (Nasdaq + NYSE),
   ohne ETFs, Fonds, SPACs, Optionsscheine, Units, Rechte, Vorzugsaktien.
2. Aufnahme: an mindestens 20 Tagen seit 2018 Rohkurs > 5 USD und Ø-Volumen(30) ≥ 1 Mio.
3. **Wer einmal drin ist, bleibt drin** – auch nach Delisting. Ab dem ersten Lauf
   wächst so ein Datensatz ohne Survivorship-Lücke heran.

Plus Benchmarks (SPY, QQQ, QQQE, RSP, IWM) und alle Titel aus `cohorts.csv`, auch wenn sie die Filter nicht erfüllen.

## Dateien

| Datei | Inhalt |
|---|---|
| `data/ohlcv/<Aufnahmedatum>/YYYY-MM.parquet` | **Rohkurse** wie damals gehandelt: date, ticker, open, high, low, close, volume |
| `data/splits.csv` | Splits (ticker, date, ratio) – Bereinigung passiert erst beim Laden |
| `universe/universe.csv` | Ticker, Name, Börse, aufgenommen am, zuletzt gelistet, Tage mit erfüllten Grundfiltern |
| `universe/cohorts.csv` | Herkunft: `handverlesen`, `zufall`, sonst `neu` |
| `data/meta.csv` | Sektor, Branche, Market Cap, Float – **Stand Abruf, nicht point-in-time** |
| `data/quality.csv` | Prüfung je Ticker, `status` ok / warn / bad |
| `data/last_run.json` | Ergebnis des letzten Laufs |

Warum Rohkurse: Ein Split ändert dann nur `splits.csv`, nicht die gespeicherte Historie.
Und der Filter „Kurs > 5“ prüft den echten damaligen Kurs.

Warum Ordner je Aufnahmedatum: Neue Titel landen in einem eigenen kleinen Ordner, statt
alle bestehenden Monatsdateien zu verändern. Täglich ändert sich nur der laufende Monat –
die Git-Historie wächst so um wenige MB pro Woche statt um den ganzen Datensatz.

## Laden im Backtest

```python
import sys; sys.path.insert(0, "scripts")
from load import load

df = load(start="2019-01-01")             # bereinigt + point-in-time-Filter
df = load(cohorts=["zufall", "neu"])      # ohne die handverlesenen Titel
signals = df[df.base_ok & df.atr_ok]      # Grundfilter am jeweiligen Tag
```

Spalten: bereinigtes OHLCV, `raw_close`, `raw_volume`, `bar_no`, `avgvol30`, `dollarvol30`,
`atr_pct`, `base_ok`, `atr_ok`, `cohort`, `quality`, `sector`, `industry`, `float_pct`.

## Grenzen

- Titel, die **vor dem ersten Lauf** delistet wurden, fehlen. Ganz sauber geht das nur mit
  einer survivorship-freien Quelle (z. B. Norgate Data).
- Market Cap und Float gibt es nur mit heutigem Stand. Im Backtest dient `dollarvol30` als Ersatz.
- yfinance ist inoffiziell. Bei Drosselung holt der nächste Lauf fehlende Titel nach.

## Lokal

```bash
pip install -r requirements.txt
cd scripts
python update.py --mode full    # Vollaufbau
python quality.py
python load.py                   # Kurzübersicht
```
