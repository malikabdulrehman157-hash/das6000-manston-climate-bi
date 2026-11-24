# DAS6000 – Manston (EGMH) Climate Data: Cleaning and Analysis

Business Analytics & Intelligence coursework. Python cleaning and analysis of NOAA NCEI Local Climatological Data (LCD) for Manston, Kent, ready for a Tableau dashboard.

## What the script does
`manston_analysis.py` runs in two parts:

**1. Data cleaning (13 steps)**: profiles the raw hourly data, handles missing values, builds timestamps and removes duplicates, keeps hourly reports only, converts units (km, hPa, mm), checks consistency, removes physically impossible values (keeping real extremes), turns weather codes into Rain/Snow/Fog/StrongWind flags, and builds daily KPIs with a 75% completeness rule.

**2. Analysis (Q1–Q8)**: temperature by year and month, summer hot days, heating/cooling degree days, fog, wet weather, wind direction and strong winds, humidity by season, and data completeness.

## Input
- `manston_hourly_raw.csv` (hourly rows extracted from the NCEI LCD report), placed next to the script

## Outputs
| File | Contents |
|---|---|
| `manston_hourly_clean.csv` | Cleaned hourly data |
| `manston_daily_tableau.csv` | Cleaned daily data with KPIs (used in Tableau) |
| `cleaning_log.csv` | Each cleaning step and its row count |
| `data_quality_report.csv` | Missing values, consistency checks, outliers, quality scores |
| `analysis_results.txt` | Everything the script prints |

## How to run
```bash
pip install pandas numpy
python manston_analysis.py
```
