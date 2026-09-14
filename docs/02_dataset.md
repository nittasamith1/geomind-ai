# 02 — Dataset Documentation & Scientific Profiling

## 1. Dataset Provenance & Overview
- **Dataset Name:** Metro Interstate Traffic Volume
- **Source:** UCI Machine Learning Repository / Minnesota Department of Transportation (MnDOT) ATR station 301.
- **Physical Location:** Interstate 94 (I-94) Westbound, located roughly midway between Minneapolis and St. Paul, MN.
- **File Location:** `data/Metro_Interstate_Traffic_Volume.csv`
- **Total Records:** 48,204 rows
- **Total Raw Columns:** 9 features

---

## 2. Schema Definition & Variables

| Column Name | Data Type | Role | Description |
| :--- | :--- | :--- | :--- |
| `holiday` | Categorical | Exogenous Feature | US National / State holiday indicator (or `None` for regular days) |
| `temp` | Continuous (float) | Exogenous Feature | Average ambient temperature in Kelvin |
| `rain_1h` | Continuous (float) | Exogenous Feature | Rainfall amount accumulated in the past hour (mm) |
| `snow_1h` | Continuous (float) | Exogenous Feature | Snowfall amount accumulated in the past hour (mm) |
| `clouds_all` | Integer | Exogenous Feature | Cloud coverage percentage (0% to 100%) |
| `weather_main` | Categorical | Exogenous Feature | Short categorical summary of current weather (e.g., Rain, Clear, Clouds, Snow) |
| `weather_description`| Categorical | Exogenous Feature | Granular meteorological description (e.g., light rain, scattered clouds) |
| `date_time` | Timestamp | Temporal Index | Hourly timestamp of the observation (UTC/Local) |
| `traffic_volume` | Integer | **Target Variable ($y$)** | Hourly westbound traffic volume count (vehicles per hour) |

---

## 3. Target Distribution Summary
- **Mean:** 3,259.82 vehicles/hour
- **Standard Deviation:** 1,986.86 vehicles/hour
- **Minimum:** 0 vehicles/hour
- **25th Percentile:** 1,193 vehicles/hour
- **Median (50th Percentile):** 3,380 vehicles/hour
- **75th Percentile:** 4,933 vehicles/hour
- **Maximum:** 7,280 vehicles/hour

---

## 4. Scientific Findings, Irregularities & Data Integrity

Through automated profiling in `src/data/dataset_loader.py`, we identified several real-world data issues that must be properly handled:

### A. Timestamp Multiplicity (Duplicate Timestamps)
- **Observed:** 7,629 duplicate timestamps out of 48,204 rows; exactly 40,575 unique timestamps exist.
- **Root Cause:** When weather conditions changed within an hour or multiple weather phenomena co-occurred (e.g., rain and clouds), the station recorded multiple rows for that single hour.
- **Scientific Solution:** Group by `date_time`, averaging sensor features (`temp`, `rain_1h`, `clouds_all`) and selecting the primary weather condition before building strict time-series sequences.

### B. Temporal Gaps (Sampling Discontinuity)
- **Time Horizon:** 2012-10-02 09:00:00 to 2018-09-30 23:00:00 (approx. 6 years).
- **Theoretical Hourly Count:** 52,551 hours.
- **Actual Unique Hours:** 40,575 hours (Completeness: ~77.21%).
- **Implication:** Over 11,900 hourly steps are missing due to sensor downtime and telemetry interruptions. If computing lag features (e.g., $\text{lag}_1$), we cannot simply shift indices; we must compute lags with respect to true temporal delta ($\Delta t = 1\text{ hour}$).

### C. Physical Sensor Anomalies (Outliers)
1. **Unphysical Temperature:** 10 records recorded temperatures below 200 K (including 0.0 K / -273.15 °C). This represents sensor disconnection or power failure, not valid Minnesota weather.
2. **Extreme Rain Spike:** 1 record recorded `rain_1h = 9,831.3 mm` in one hour (nearly 10 meters of rain). This is an erroneous hardware telemetry spike that must be clipped or imputed.
