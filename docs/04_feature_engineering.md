# 04 — Feature Engineering & Preprocessing Rationale

## 1. Core Architectural Principle: No Target Leakage & Clean User Inference

In real-world deployment, traffic volume is the **unknown forecast target**. End users should never be asked to input the current hour's traffic count to predict traffic. 

In `src/feature_engineering.py`, feature generation is completely decoupled from past or present traffic volume:
- **During Training:** `traffic_volume` is isolated strictly as the label (`traffic_volume_target`) and excluded from the feature matrix.
- **During Inference:** The model accepts only user-accessible contextual signals: date/time, temperature, weather condition, rain, snow, cloud cover, and holiday status.

---

## 2. Feature Taxonomies & Engineering Rationale

### A. Multi-Scale Cyclical Trigonometric Encodings
Integer representations of time ($0, \dots, 23$) introduce artificial numerical discontinuities (e.g. 23:00 and 00:00 are 1 hour apart in reality, but 23 units apart numerically). We project temporal cycles onto the 2D Euclidean unit circle:

$$\sin_{\text{hour}} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \cos_{\text{hour}} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
$$\sin_{\text{dow}} = \sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right), \quad \cos_{\text{dow}} = \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$$
$$\sin_{\text{month}} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \cos_{\text{month}} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
$$\sin_{\text{doy}} = \sin\left(\frac{2\pi \cdot \text{day\_of\_year}}{365}\right), \quad \cos_{\text{doy}} = \cos\left(\frac{2\pi \cdot \text{day\_of\_year}}{365}\right)$$

This guarantees seamless continuity across midnight, weekly transitions, and seasonal shifts.

### B. Commute Regimes & Part-of-Day Categorization
- `is_weekend`: Binary flag identifying Saturday and Sunday.
- `is_holiday`: Binary flag active during US Federal and State holidays.
- `rush_bucket`: Discretized commuter timeline:
  - `0`: Off-peak / standard flow
  - `1`: Morning rush hour (07:00–09:00, weekdays)
  - `2`: Evening rush hour (16:00–18:00, weekdays)
  - `3`: Midday commercial flow (10:00–15:00)
  - `4`: Overnight trough (00:00–05:00)
- `is_rush_hour`: Active flag for morning and evening weekday peaks.
- `part_of_day`: Binned day phases (night, morning, midday, afternoon, evening).

### C. Meteorological Signals & Derived Domain Indicators
- **Temperature Normalization & Comfort Index:** Ambient temperature converted to Celsius ($T_{\text{C}} = T_{\text{K}} - 273.15$). We compute `temp_comfortable` ($15^\circ\text{C} \le T_{\text{C}} \le 25^\circ\text{C}$), `temp_extreme_cold` ($< 0^\circ\text{C}$), and `temp_extreme_hot` ($> 35^\circ\text{C}$).
- **Precipitation Severity:** `has_rain`, `has_snow`, `rain_heavy` ($> 5\text{ mm}$), `snow_heavy` ($> 0.5\text{ mm}$), and `precip_total` ($=\text{rain} + \text{snow}$).
- **Cloud Coverage Buckets:** `mostly_clear` ($\le 25\%$) and `overcast` ($\ge 75\%$).
- **Weather Severity Hierarchy:** Ordinal mapping reflecting roadway safety risk:
  - Severe ($2$): `Snow`, `Thunderstorm`
  - Moderate ($1$): `Rain`, `Drizzle`, `Mist`, `Fog`, `Haze`, `Smoke`
  - Mild ($0$): `Clear`, `Clouds`
- `bad_weather`: Unified binary alert when road conditions degrade commuter flow.

### D. Compound Non-Linear Interaction Terms
- `rush_bad_weather`: Intersection of peak commute hours and adverse weather.
- `rush_weekday`: Peak commute hours isolated to non-holiday working days.
- `holiday_rush`: Suppression indicator for commuter hours falling on holidays.
- `weekend_midday`: Captures midday weekend shopping and leisure travel peaks.
- `summer_weekend`: Accounts for seasonal summer vacation and weekend recreational traffic surges (June–August).

---

## 3. Preprocessing Pipeline & Production Artifacts

Implemented in `src/data_preprocessing.py`:
1. **ColumnTransformer:**
   - **Numeric Features (~33):** Scaled via `StandardScaler` (zero mean, unit variance).
   - **Categorical Features (`weather_main`):** One-hot encoded via `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
2. **Train-Only Fitting:** Fitted strictly on `data/processed/train.csv` (28,400+ samples).
3. **Artifact Persistence:** Serialized to `models/preprocessor.joblib`. During FastAPI inference, the pipeline transforms incoming single-row observations instantly without refitting.
