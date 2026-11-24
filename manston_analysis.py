"""
manston_analysis.py - clean the NCEI LCD data for Manston (EGMH) and answer Q1-Q8
=================================================================================

Input : manston_hourly_raw.csv  (67,791 hourly rows, already extracted from the NCEI LCD report)
Output: manston_hourly_clean.csv   - cleaned hourly data
        manston_daily_tableau.csv  - cleaned daily data with KPIs  <- use this in Tableau
        cleaning_log.csv           - every cleaning step and its count
        data_quality_report.csv    - missing values, consistency checks, outliers, quality scores
        analysis_results.txt       - everything printed below

Run in PyCharm: right-click this file -> Run.  Needs: pip install pandas numpy
"""

import os
import re
import sys

import numpy as np
import pandas as pd

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

RAW_NAME = "manston_hourly_raw.csv"
HOURLY_OUT = os.path.join(BASE_DIR, "manston_hourly_clean.csv")
DAILY_OUT = os.path.join(BASE_DIR, "manston_daily_tableau.csv")
LOG_OUT = os.path.join(BASE_DIR, "cleaning_log.csv")
RESULTS_OUT = os.path.join(BASE_DIR, "analysis_results.txt")
QUALITY_OUT = os.path.join(BASE_DIR, "data_quality_report.csv")

HOURLY = "h" if tuple(int(x) for x in re.findall(r"\d+", pd.__version__)[:2]) >= (2, 2) else "H"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SECTORS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)

# everything printed is also saved to analysis_results.txt
_lines = []


def out(text=""):
    print(text)
    _lines.append(str(text))


def section(title):
    out()
    out("=" * 70)
    out(title)
    out("=" * 70)


def season_of(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


# ---------------------------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------------------------
def find_raw():
    for path in (os.path.join(BASE_DIR, RAW_NAME), os.path.join(BASE_DIR, "output", RAW_NAME)):
        if os.path.isfile(path):
            return path
    sys.exit("Cannot find " + RAW_NAME + " in " + BASE_DIR + " (or its 'output' folder).")


def sub_header(step, title):
    out()
    out("-" * 70)
    out("Step {:<3} {}".format(step, title))
    out("-" * 70)


def clean():
    log = []
    quality = []          # rows for data_quality_report.csv
    out("=" * 70)
    out("=== DATA CLEANING ===")
    out("=" * 70)

    # ---------------------------------------------------------------- 1 profile
    raw_path = find_raw()
    raw = pd.read_csv(raw_path, dtype=str)
    sub_header(1, "Load and profile the raw data")
    out("File: {}  ({:.1f} MB)".format(os.path.basename(raw_path), os.path.getsize(raw_path) / 1e6))
    out("Shape: {:,} rows x {} columns | memory {:.1f} MB".format(
        raw.shape[0], raw.shape[1], raw.memory_usage(deep=True).sum() / 1e6))
    out("All columns arrive as TEXT because the PDF stores numbers as printed characters.")
    out("Columns: " + ", ".join(raw.columns))
    out("First 3 raw rows (selected columns):")
    out(raw[["Year", "Month", "Day", "Time", "DryBulb_F", "DryBulb_C", "WindDir_deg",
             "Visibility_mi", "WeatherType", "SkyConditions"]].head(3).to_string(index=False))
    log.append(("Raw hourly rows loaded", len(raw)))

    # ---------------------------------------------------------------- 2 missing
    sub_header(2, "Missing values and how each is treated")
    treatment = {
        "DryBulb_C": "Kept as null; day excluded if <18 of 24 hours",
        "DewPoint_C": "Kept as null",
        "RelHum_pct": "Kept as null",
        "WindSpeed_mph": "Kept as null",
        "WindDir_deg": "Kept as null",
        "SeaLevelPress_inHg": "Kept as null",
        "Visibility_mi": "Kept as null; fog % uses hours with visibility only",
        "WeatherType": "Blank = no significant weather -> 'None'",
        "WindGust_mph": "Blank = no gust -> not used as a KPI",
        "Precip_in": "Only Mar 2022-Dec 2024 -> excluded as a KPI",
        "SkyConditions": "Kept as null",
        "WetBulb_C": "Dropped (duplicates temperature + humidity)",
        "StationPress_inHg": "Dropped (sea-level pressure used instead)",
        "Altimeter_inHg": "Dropped (100% empty)",
    }
    miss = pd.DataFrame({"missing": raw[list(treatment)].isna().sum()})
    miss["missing_pct"] = (miss["missing"] / len(raw) * 100).round(1)
    miss["treatment"] = [treatment[c] for c in miss.index]
    miss = miss.sort_values("missing_pct", ascending=False)
    out(miss.to_string())
    for col, row in miss.iterrows():
        quality.append(("missing", col, int(row["missing"]), row["missing_pct"], row["treatment"]))

    # ---------------------------------------------------------------- 3 dates
    sub_header(3, "Dates, coverage and duplicates")
    raw["Timestamp"] = pd.to_datetime(raw["Year"] + "-" + raw["Month"] + "-" + raw["Day"] + " " + raw["Time"],
                                      format="%Y-%m-%d %H%M", errors="coerce")
    bad = int(raw["Timestamp"].isna().sum())
    df = raw.dropna(subset=["Timestamp"])
    before = len(df)
    df = df.drop_duplicates("Timestamp").sort_values("Timestamp").reset_index(drop=True)
    dups = before - len(df)
    first, last = df["Timestamp"].min(), df["Timestamp"].max()
    month_keys = set(zip(df["Timestamp"].dt.year, df["Timestamp"].dt.month))
    all_months = pd.period_range(first, last, freq="M")
    gaps = [str(p) for p in all_months if (p.year, p.month) not in month_keys]
    out("Combined Year + Month + Day + Time into one Timestamp (format YYYY-MM-DD HHMM)")
    out("Unreadable timestamps removed: {} | duplicate timestamps removed: {}".format(bad, dups))
    out("Period: {} to {}  ({} months, missing months: {})".format(
        first.strftime("%d %b %Y %H:%M"), last.strftime("%d %b %Y %H:%M"), len(all_months), gaps or "none"))
    out("Note: Jan 2020 starts on the 17th and Aug 2025 ends on the 19th (partial months)")
    log += [("Unreadable timestamps removed", bad), ("Duplicate timestamps removed", dups)]

    # ---------------------------------------------------------------- 4 frequency
    sub_header(4, "Reporting frequency (hourly vs 10-minute)")
    per_year = df.groupby(df["Timestamp"].dt.year).size()
    minute_mix = (df.assign(minute=df["Timestamp"].dt.minute, year=df["Timestamp"].dt.year)
                  .pivot_table(index="year", columns="minute", values="Time", aggfunc="count", fill_value=0))
    out("Rows per year: {}".format(per_year.to_dict()))
    out("Rows by minute past the hour (shows the 2025 switch to 10-minute reports):")
    out(minute_mix.to_string())
    sub = df["Timestamp"].dt.minute != 0
    df = df[~sub].copy()
    out("Decision: keep only :00 reports so every year has the same frequency")
    out("Removed {:,} sub-hourly rows -> {:,} hourly rows remain".format(int(sub.sum()), len(df)))
    log += [("Sub-hourly (10-min) rows removed", int(sub.sum())), ("Hourly rows kept", len(df))]

    # ---------------------------------------------------------------- 5 types and units
    sub_header(5, "Data types and unit conversion")
    numeric = ["Visibility_mi", "DryBulb_F", "DryBulb_C", "DewPoint_C", "RelHum_pct", "WindSpeed_mph",
               "WindGust_mph", "SeaLevelPress_inHg", "Precip_in"]
    rows = []
    for col in numeric:
        as_num = pd.to_numeric(df[col], errors="coerce")
        not_num = df[col].notna() & as_num.isna()
        rows.append((col, int(not_num.sum()), ", ".join(df.loc[not_num, col].unique()[:3]) or "-"))
        df[col] = as_num
    wd = df["WindDir_deg"]
    rows.append(("WindDir_deg", int((wd == "VRB").sum()), "VRB"))
    out(pd.DataFrame(rows, columns=["column", "non_numeric_values", "examples"]).to_string(index=False))
    df["Visibility_km"] = (df["Visibility_mi"] * 1.609344).round(2)
    df["SeaLevelPress_hPa"] = (df["SeaLevelPress_inHg"] * 33.8639).round(1)
    df["Precip_mm"] = (df["Precip_in"] * 25.4).round(1)
    example = df[["Visibility_mi", "Visibility_km", "SeaLevelPress_inHg", "SeaLevelPress_hPa"]].dropna().head(3)
    out("Unit conversion (miles -> km x1.609; inHg -> hPa x33.86; inches -> mm x25.4), examples:")
    out(example.to_string(index=False))

    # ---------------------------------------------------------------- 6 consistency
    sub_header(6, "Consistency and validity checks")
    f_to_c = (df["DryBulb_F"] - 32) * 5 / 9
    checks = [
        ("Fahrenheit and Celsius disagree by >0.6 C", int(((f_to_c - df["DryBulb_C"]).abs() > 0.6).sum())),
        ("Dew point above air temperature (impossible)", int((df["DewPoint_C"] > df["DryBulb_C"] + 0.5).sum())),
        ("Relative humidity outside 0-100%", int((~df["RelHum_pct"].between(0, 100) & df["RelHum_pct"].notna()).sum())),
        ("Gust lower than mean wind", int((df["WindGust_mph"] < df["WindSpeed_mph"]).sum())),
        ("Calm (0 mph) but direction recorded as 000", int(((df["WindSpeed_mph"] == 0) & (df["WindDir_deg"] == "000")).sum())),
        ("Variable wind direction 'VRB'", int((df["WindDir_deg"] == "VRB").sum())),
    ]
    out(pd.DataFrame(checks, columns=["check", "rows_failing"]).to_string(index=False))
    df["WindVariable"] = df["WindDir_deg"] == "VRB"
    df["WindDir_deg"] = pd.to_numeric(df["WindDir_deg"].replace("VRB", np.nan), errors="coerce")
    calm = df["WindSpeed_mph"] == 0
    df.loc[calm, "WindDir_deg"] = np.nan
    out("Action: 'VRB' and calm-hour directions set to null (000 would wrongly count as northerly)")
    for name, n in checks:
        quality.append(("consistency", name, n, round(n / len(df) * 100, 2), "checked"))
    log += [("VRB wind directions set to null", int(df["WindVariable"].sum())),
            ("Calm-hour directions set to null", int(calm.sum()))]

    # ---------------------------------------------------------------- 7 outliers
    sub_header(7, "Outlier detection (physical limits + IQR rule)")
    limits = {"DryBulb_C": (-25, 40), "DewPoint_C": (-30, 30), "RelHum_pct": (1, 100),
              "WindSpeed_mph": (0, 100), "SeaLevelPress_hPa": (940, 1060), "Visibility_km": (0, 100)}
    rows = []
    n_out = 0
    for col, (low, high) in limits.items():
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        iqr_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        impossible = df[col].notna() & ~df[col].between(low, high)
        n_out += int(impossible.sum())
        df.loc[impossible, col] = np.nan
        rows.append((col, s.min(), s.max(), "{}-{}".format(low, high), int(impossible.sum()), iqr_out))
        quality.append(("outliers", col, iqr_out, round(iqr_out / len(s) * 100, 2), "IQR flags kept (real weather)"))
    out(pd.DataFrame(rows, columns=["column", "min", "max", "allowed_range", "impossible_removed",
                                    "IQR_flags"]).to_string(index=False))
    out("Decision: IQR flags are real extreme weather (e.g. 35.6 C heatwave 19 Jul 2022,")
    out("          956.7 hPa Storm Ciaran 2 Nov 2023), so they are KEPT; only impossible values are removed.")
    log.append(("Impossible values set to null", n_out))

    # ---------------------------------------------------------------- 8 weather codes
    sub_header(8, "Weather codes -> analysis flags")
    weather = df["WeatherType"].fillna("").str.replace("|", "", regex=False).str.strip()
    df["WeatherCode"] = weather.replace("", "None")
    meaning = {"None": "no significant weather", "RA": "rain", "DZ": "drizzle", "HZ": "haze",
               "SHRA": "rain showers", "FG": "fog", "SN": "snow", "SG": "snow grains",
               "PL": "ice pellets", "SHSN": "snow showers"}
    codes = df["WeatherCode"].value_counts().to_frame("hours")
    codes["pct"] = (codes["hours"] / len(df) * 100).round(2)
    codes["meaning"] = [meaning.get(c, "") for c in codes.index]
    out(codes.to_string())
    df["Rain"] = weather.str.contains(r"\b(?:RA|SHRA|DZ)\b", regex=True)
    df["Snow"] = weather.str.contains(r"\b(?:SN|SHSN|SG|PL)\b", regex=True)
    df["Fog"] = weather.str.contains(r"\bFG\b", regex=True) | (df["Visibility_km"] < 1)
    df["StrongWind"] = df["WindSpeed_mph"] >= 25
    out("Flags: Rain = RA/SHRA/DZ ({:,} h) | Snow = SN/SHSN/SG/PL ({:,} h) | Fog = FG or visibility <1 km ({:,} h)"
        .format(int(df["Rain"].sum()), int(df["Snow"].sum()), int(df["Fog"].sum())))
    out("       StrongWind = mean wind >= 25 mph ({:,} h)".format(int(df["StrongWind"].sum())))

    df["Date"] = df["Timestamp"].dt.normalize()
    df["Year"] = df["Timestamp"].dt.year
    df["Month"] = df["Timestamp"].dt.month
    df["Hour"] = df["Timestamp"].dt.hour
    df["Season"] = df["Month"].apply(season_of)

    # ---------------------------------------------------------------- 9 daily
    sub_header(9, "Daily aggregation and the 75% completeness rule")
    g = df.groupby("Date")
    daily = pd.DataFrame({
        "TempObs": g["DryBulb_C"].count(), "Tmax_C": g["DryBulb_C"].max(), "Tmin_C": g["DryBulb_C"].min(),
        "Tmean_C": g["DryBulb_C"].mean().round(2), "RH_mean": g["RelHum_pct"].mean().round(1),
        "Wind_mean_mph": g["WindSpeed_mph"].mean().round(1), "Wind_max_mph": g["WindSpeed_mph"].max(),
        "RainHours": g["Rain"].sum(), "FogHours": g["Fog"].sum(), "StrongWindHours": g["StrongWind"].sum(),
        "MinSLP_hPa": g["SeaLevelPress_hPa"].min(),
    })
    days = pd.date_range(first.normalize(), last.normalize(), freq="D")
    daily = daily.reindex(days)
    daily.index.name = "Date"
    daily["TempObs"] = daily["TempObs"].fillna(0).astype(int)
    bands = pd.cut(daily["TempObs"], bins=[-1, 0, 11, 17, 23, 24],
                   labels=["0 (no data)", "1-11", "12-17", "18-23", "24 (complete)"])
    out("Days by number of hourly temperature readings:")
    out(bands.value_counts().sort_index().to_string())
    daily["Valid"] = daily["TempObs"] >= 18
    for col in ["Tmax_C", "Tmin_C", "Tmean_C"]:
        daily.loc[~daily["Valid"], col] = np.nan
    out("Rule: a day is VALID with at least 18 of 24 readings (75%); otherwise Tmax/Tmin/Tmean = null")
    out("Result: {:,} days | {:,} valid | {} excluded".format(
        len(daily), int(daily["Valid"].sum()), int((~daily["Valid"]).sum())))
    out("Excluded days by year: {}".format(
        daily[~daily["Valid"]].groupby(daily[~daily["Valid"]].index.year).size().to_dict()))
    daily["HDD"] = (15.5 - daily["Tmean_C"]).clip(lower=0).round(2)
    daily["CDD"] = (daily["Tmean_C"] - 22).clip(lower=0).round(2)
    daily["HotDay25"] = daily["Tmax_C"] >= 25
    daily["FrostDay"] = daily["Tmin_C"] < 0
    daily = daily.reset_index()
    daily["Year"] = daily["Date"].dt.year
    daily["Month"] = daily["Date"].dt.month
    daily["Season"] = daily["Month"].apply(season_of)
    out("Derived fields: HDD = max(15.5 - Tmean, 0) | CDD = max(Tmean - 22, 0) | HotDay25 = Tmax >= 25 | FrostDay = Tmin < 0")
    log += [("Calendar days", len(daily)), ("Days with no data", int((daily["TempObs"] == 0).sum())),
            ("Days failing 75% rule", int((~daily["Valid"]).sum())), ("Valid days", int(daily["Valid"].sum()))]

    # ---------------------------------------------------------------- 10 describe
    sub_header(10, "Descriptive statistics of the clean hourly data")
    stats_cols = ["DryBulb_C", "DewPoint_C", "RelHum_pct", "WindSpeed_mph", "Visibility_km", "SeaLevelPress_hPa"]
    out(df[stats_cols].describe().round(1).to_string())

    # ---------------------------------------------------------------- 11 before / after
    hourly_cols = ["Timestamp", "Date", "Year", "Month", "Season", "Hour", "DryBulb_C", "DewPoint_C",
                   "RelHum_pct", "WindSpeed_mph", "WindDir_deg", "WindVariable", "WindGust_mph",
                   "Visibility_km", "SeaLevelPress_hPa", "Precip_mm", "WeatherCode", "Rain", "Snow",
                   "Fog", "StrongWind"]
    hourly = df[hourly_cols].copy()
    sub_header(11, "Before vs after")
    compare = pd.DataFrame({
        "raw": ["{:,}".format(len(raw)), raw.shape[1] - 1, "text", "F + C, miles, inHg, inches",
                "mixed hourly / 10-min", "text codes (|RA |)", "none"],
        "clean": ["{:,} hourly + {:,} daily".format(len(hourly), len(daily)), "{} + {}".format(
            hourly.shape[1], daily.shape[1]), "numbers, dates, true/false", "C, km, hPa, mm",
            "hourly only", "Rain / Snow / Fog flags", "Valid, HDD, CDD, HotDay25, FrostDay, Season"],
    }, index=["rows", "columns", "data types", "units", "frequency", "weather", "derived fields"])
    out(compare.to_string())

    # ---------------------------------------------------------------- 12 scorecard
    sub_header(12, "Data quality scorecard (clean data)")
    expected = pd.date_range(first.normalize(), last.normalize() + pd.Timedelta(hours=23), freq=HOURLY)
    key = ["DryBulb_C", "DewPoint_C", "RelHum_pct", "WindSpeed_mph", "SeaLevelPress_hPa"]
    score = pd.DataFrame([
        ("Completeness", "hourly slots with a temperature", hourly["DryBulb_C"].notna().sum() / len(expected) * 100),
        ("Completeness", "key fields filled (temp, dew point, RH, wind, pressure)", hourly[key].notna().mean().mean() * 100),
        ("Validity", "values inside physical limits", 100.0),
        ("Consistency", "F and C temperatures agree", 100 - checks[0][1] / len(df) * 100),
        ("Uniqueness", "one record per hour", 100.0),
        ("Usability", "days usable for daily KPIs (75% rule)", daily["Valid"].mean() * 100),
    ], columns=["dimension", "measure", "score_pct"])
    score["score_pct"] = score["score_pct"].round(1)
    out(score.to_string(index=False))
    for _, r in score.iterrows():
        quality.append(("scorecard", r["dimension"] + ": " + r["measure"], "", r["score_pct"], ""))

    # ---------------------------------------------------------------- 13 save
    sub_header(13, "Save clean files")
    hourly.to_csv(HOURLY_OUT, index=False)
    daily.to_csv(DAILY_OUT, index=False)
    pd.DataFrame(log, columns=["Step", "Count"]).to_csv(LOG_OUT, index=False)
    pd.DataFrame(quality, columns=["type", "item", "count", "pct", "note"]).to_csv(QUALITY_OUT, index=False)
    out("Saved {} {}".format(os.path.basename(HOURLY_OUT), hourly.shape))
    out("Saved {} {}".format(os.path.basename(DAILY_OUT), daily.shape))
    out("Saved {} and {}".format(os.path.basename(LOG_OUT), os.path.basename(QUALITY_OUT)))
    return hourly, daily


# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------
def analyse(hourly, daily):
    valid = daily[daily["Valid"]]
    expected = pd.date_range(hourly["Timestamp"].min().normalize(),
                             hourly["Timestamp"].max().normalize() + pd.Timedelta(hours=23), freq=HOURLY)
    completeness = hourly["DryBulb_C"].notna().sum() / len(expected) * 100
    peak = hourly.loc[hourly["DryBulb_C"].idxmax()]
    low = hourly.loc[hourly["DryBulb_C"].idxmin()]

    section("Headline KPIs")
    out("Station: Manston, Kent (EGMH) | {} to {} | {:,} hourly records".format(
        hourly["Timestamp"].min().date(), hourly["Timestamp"].max().date(), len(hourly)))
    out("Mean temp 2021-24: {:.1f} C | peak {:.1f} C ({}) | lowest {:.1f} C ({})".format(
        valid[valid["Year"].between(2021, 2024)].groupby("Year")["Tmean_C"].mean().mean(),
        peak["DryBulb_C"], peak["Timestamp"], low["DryBulb_C"], low["Timestamp"]))
    out("Data completeness: {:.1f}% of expected hourly temperature records".format(completeness))

    section("Q1  Temperature by year (valid days only)")
    q1 = valid.groupby("Year").agg(valid_days=("Tmean_C", "count"), mean_C=("Tmean_C", "mean"),
                                   mean_max_C=("Tmax_C", "mean"), mean_min_C=("Tmin_C", "mean"),
                                   frost_days=("FrostDay", "sum")).round(2)
    out(q1.to_string())
    out()
    out("Mean temperature by month (all years):")
    q1m = valid.groupby("Month")["Tmean_C"].mean().round(1)
    q1m.index = [MONTHS[m - 1] for m in q1m.index]
    out(q1m.to_string())

    section("Q2  Summer hot days (Jun-Aug, valid days)")
    summer = valid[valid["Month"].isin([6, 7, 8])]
    q2 = summer.groupby("Year").agg(summer_days=("HotDay25", "size"), hot_days=("HotDay25", "sum"),
                                    jja_mean_C=("Tmean_C", "mean"), max_C=("Tmax_C", "max"))
    q2["hot_day_rate_pct"] = (q2["hot_days"] / q2["summer_days"] * 100).round(1)
    out(q2.round(2).to_string())
    out("Hottest days:")
    out(valid.nlargest(5, "Tmax_C")[["Date", "Tmax_C"]].to_string(index=False))

    section("Q3  Heating and cooling degree days")
    q3 = valid.groupby("Year").agg(valid_days=("HDD", "size"), HDD=("HDD", "sum"), CDD=("CDD", "sum")).round(1)
    out(q3.to_string())
    share = (valid.groupby("Month")["HDD"].sum() / valid["HDD"].sum() * 100).round(1)
    share.index = [MONTHS[m - 1] for m in share.index]
    out("Share of heating degree days by month (%):")
    out(share.to_string())
    out("Dec-Mar share of HDD: {:.1f}%".format(
        valid[valid["Month"].isin([12, 1, 2, 3])]["HDD"].sum() / valid["HDD"].sum() * 100))

    section("Q4  Fog (% of hours with visibility data)")
    vis = hourly[hourly["Visibility_km"].notna()]
    q4 = (vis.groupby("Month")["Fog"].mean() * 100).round(2)
    q4.index = [MONTHS[m - 1] for m in q4.index]
    out(q4.to_string())
    by_hour = (vis.groupby("Hour")["Fog"].mean() * 100).round(2)
    out("Peak fog hour: {:02d}:00 ({:.2f}%) | lowest: {:02d}:00 ({:.2f}%)".format(
        int(by_hour.idxmax()), by_hour.max(), int(by_hour.idxmin()), by_hour.min()))
    out("Fog hours by year: {}".format(vis.groupby("Year")["Fog"].sum().to_dict()))

    section("Q5  Wet weather")
    q5 = hourly.groupby("Year").agg(hours=("Rain", "size"), rain_hours=("Rain", "sum"), snow_hours=("Snow", "sum"))
    q5["rain_per_1000_hours"] = (q5["rain_hours"] / q5["hours"] * 1000).round(0)
    out(q5.to_string())
    out("% of hours with rain by season:")
    out((hourly.groupby("Season")["Rain"].mean() * 100).round(1).to_string())

    section("Q6  Wind")
    wind = hourly[hourly["WindDir_deg"].notna()].copy()
    wind["Sector"] = (((wind["WindDir_deg"] + 11.25) % 360) // 22.5).astype(int).map(lambda i: SECTORS[i])
    q6 = pd.DataFrame({
        "all_hours_pct": wind["Sector"].value_counts(normalize=True) * 100,
        "strong_hours_pct": wind[wind["StrongWind"]]["Sector"].value_counts(normalize=True) * 100,
    }).fillna(0).round(1).sort_values("all_hours_pct", ascending=False)
    out(q6.head(8).to_string())
    strong = hourly[hourly["StrongWind"]]
    by_month = strong.groupby("Month").size().reindex(range(1, 13), fill_value=0)
    out("Strong-wind hours (mean wind >= 25 mph): {:,} | Nov-Mar share: {:.1f}%".format(
        len(strong), by_month[[11, 12, 1, 2, 3]].sum() / len(strong) * 100))
    by_month.index = MONTHS
    out("By month: {}".format(by_month.to_dict()))
    out("Lowest-pressure days:")
    out(daily.nsmallest(3, "MinSLP_hPa")[["Date", "MinSLP_hPa", "Wind_max_mph"]].to_string(index=False))

    section("Q7  Humidity by season")
    q7 = hourly.groupby("Season").agg(mean_RH=("RelHum_pct", "mean"), mean_temp_C=("DryBulb_C", "mean")).round(1)
    out(q7.to_string())
    summer_h = hourly[hourly["Season"] == "Summer"]
    out("Summer RH at 06:00 vs 15:00: {:.0f}% vs {:.0f}%".format(
        summer_h[summer_h["Hour"] == 6]["RelHum_pct"].mean(), summer_h[summer_h["Hour"] == 15]["RelHum_pct"].mean()))
    out("Correlation temperature vs humidity: all {:.2f} | summer {:.2f}".format(
        hourly["DryBulb_C"].corr(hourly["RelHum_pct"]), summer_h["DryBulb_C"].corr(summer_h["RelHum_pct"])))

    section("Q8  Data completeness")
    exp_y = pd.Series(1, index=expected).groupby(expected.year).sum()
    got_y = hourly[hourly["DryBulb_C"].notna()].groupby("Year").size()
    q8 = pd.DataFrame({"expected_hours": exp_y, "with_temperature": got_y})
    q8["complete_pct"] = (q8["with_temperature"] / q8["expected_hours"] * 100).round(1)
    q8["visibility_pct"] = (hourly.groupby("Year")["Visibility_km"].count() / exp_y * 100).round(1)
    q8["rain_amount_pct"] = (hourly.groupby("Year")["Precip_mm"].count() / exp_y * 100).round(1)
    out(q8.to_string())
    exp_m = pd.Series(1, index=expected).groupby([expected.year, expected.month]).sum()
    with_t = hourly[hourly["DryBulb_C"].notna()]
    got_m = with_t.groupby([with_t["Year"], with_t["Month"]]).size()
    pct_m = (got_m / exp_m * 100).round(1)
    worst = pct_m.nsmallest(5)
    out("Months below 90%: {} of {} | worst: {}".format(
        int((pct_m < 90).sum()), len(pct_m),
        ", ".join("{} {}: {}%".format(MONTHS[int(m) - 1], int(y), v) for (y, m), v in worst.items())))
    
def main():
    hourly, daily = clean()
    analyse(hourly, daily)
    out()
    out("DONE. Files saved in: " + BASE_DIR)
    for path in (HOURLY_OUT, DAILY_OUT, LOG_OUT, QUALITY_OUT, RESULTS_OUT):
        out("  " + os.path.basename(path))
    with open(RESULTS_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()