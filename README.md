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

Plus Benchmarks (SPY, QQQ, QQQE, RSP, IWM, die 10 SPDR-Sektor-ETFs) und alle Titel aus `cohorts.csv`, auch wenn sie die Filter nicht erfüllen.

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

## Delistete Titel nachladen (Survivorship-Test)

Titel, die zwischen 2018 und dem ersten Lauf delistet wurden, liefert yfinance nicht mehr.
Einmalig nachladen und V4.2 damit vergleichen:

1. **Probelauf ohne Abo:** Actions → **delisted** → Run workflow, Häkchen „probelauf“ an.
   Nutzt Sharadars öffentlichen `test-api-key` (Dow 30, 5 Jahre) und prüft Abruf, Rohkurse,
   Splits und Dublettenprüfung gegen die yfinance-Daten. Ausgabe nur OK/FEHLER.
2. Bei sharadar.com als Privatperson den Plan **Prices, 10 Years** buchen (monatlich kündbar).
3. API-Key (sharadar.com/account) als Secret `SHARADAR_API_KEY` hinterlegen
   (Settings → Secrets and variables → Actions). Für EODHD: `EODHD_API_KEY`.
4. Workflow ohne Häkchen „probelauf“ starten.

| Schritt | Skript | Ergebnis |
|---|---|---|
| Kurse | `scripts/delisted.py --provider sharadar\|eodhd` | `data/delisted/` (Rohkurse, Splits, Ticker, Delisting-Grund) |
| SEC | `scripts/delisted_sec.py` | Quartalszahlen + Earnings-Termine (8-K Item 2.02) |
| Vergleich | `backtest/research/v42_delisted.py` | `backtest/out/v42_delisted*.csv` |

- Sharadar: Kurse per Bulk-ZIP (`years=10`, ≈ 325 MB, schont das Rate-Limit), sonst
  Einzelabfragen à 4 Ticker. Delisting-Datum und -Grund (Übernahme, Fusion, Konkurs) aus `actions`.
- Gleiche Aufnahmeregel wie das Universum. Ticker = Symbol + Monat des Delistings
  (`TWTR-202210`), Kohorte `delistet`.
- Pleiten, die nach dem Delisting außerbörslich weiterlaufen (Q-Suffix), bleiben mit ihren
  OTC-Kursen drin – so trifft der Absturz eine offene Position wie in echt.
- Lizenz (privat): Rohdaten nicht weitergeben, nach Abo-Ende binnen 30 Tagen löschen. Deshalb
  ist `data/delisted/` nicht im Git, und der Actions-Runner wird nach dem Lauf verworfen.
  Ergebnisse (Kennzahlen, Trade-Liste) dürfen bleiben. Datenqualitäts-Vergleiche nicht
  veröffentlichen – der Probelauf schreibt Details nur nach `data/delisted_sample/`.
- Der Vergleich rechnet in einer Arbeitskopie. Engine, Regeln und `data/` bleiben unverändert.
  Einzige Ergänzung: Enden die Kurse eines delisteten Titels vor dem Datensatzende, wird eine
  offene Position am letzten Kurstag zum Schlusskurs verkauft.
- Sharadar liefert die CIK mit. Bei EODHD wird sie über den Firmennamen gesucht, dadurch
  fehlen dort eher Quartalszahlen (Spalte `Umsatzdaten delistete %`).

Lokal: `export SHARADAR_API_KEY=…`, dann die drei Skripte nacheinander. Danach
`data/delisted/` löschen, spätestens 30 Tage nach Abo-Ende.

## Grenzen

- Titel, die **vor dem ersten Lauf** delistet wurden, fehlen im regulären Datensatz. Für den
  Test siehe „Delistete Titel nachladen“.
- Market Cap und Float gibt es nur mit heutigem Stand. Im Backtest dient `dollarvol30` als Ersatz.
- yfinance ist inoffiziell. Bei Drosselung holt der nächste Lauf fehlende Titel nach.

## Merkpunkte V4.2-Bericht

- **Earnings-Sperre am Datenende (behoben 28.11.2026):** `earn_since(..., until=True)` in
  `backtest/v42_report.py` zählte jeden Earnings-Termin nach dem letzten Kurstag als „nächster
  Handelstag“. Die Sperre „1–2 Tage vor Earnings“ hat dadurch jedes Signal der letzten zwei
  Kurstage verworfen (z. B. PLTR, NWL, EAT am 27.11.2026). Künftige Termine werden jetzt über
  NYSE-Börsentage hinter dem Datenende gezählt (`_nyse_days`).
- **Prüfen bei jedem Bericht:** Neue Datumsregeln immer auch am **letzten Kurstag** testen.
  Daten jenseits des Kurskalenders (künftige Earnings, Feiertage) nie per `searchsorted` auf den
  Kalender legen, ohne das Ende abzufangen. Signale des letzten Tages gegen eine Rechnung ohne die
  neue Regel vergleichen; fehlen dort plötzlich alle, ist es ein Randfehler, kein Marktbild.

## Lokal

```bash
pip install -r requirements.txt
cd scripts
python update.py --mode full    # Vollaufbau
python quality.py
python load.py                   # Kurzübersicht
```
