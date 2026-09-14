# 04 — Feature Engineering & Preprocessing Rationale

## 1. Objective & Target Variable Formulation
In this phase, we transform raw tabular and time telemetry into an expressive, leakage-safe feature space designed for predicting next-hour traffic volume ($t+1$):

$$y_t = \text{traffic\_volume}_{t+1}$$

### Temporal Continuity Verification (Gap Protection)
In empirical time-series datasets, telemetry can be interrupted due to power failures or sensor maintenance. A naive indexing shift (`df.shift(-1)`) naively associates consecutive rows regardless of time elapsed. 

In `src/feature_engineering.py`, we enforce:
$$\text{Target}(t) = \begin{cases} y_{t+1}, & \text{if } \Delta t = t_{i+1} - t_i == 1\text{ hour} \\ \text{NaN}, & \text{otherwise} \end{cases}$$

This guarantees that multi-hour or multi-day telemetry interruptions never contaminate our 1-hour ahead forecasting models.

---

## 2. Feature Taxonomies & Scientific Rationale

### A. Cyclical Trigonometric Encodings
In linear and neural network architectures, raw integers representing circular time ($0, 1, \dots, 23$) introduce an artificial discontinuity: hour 23 and hour 0 are separated by 1 hour in reality, but by $23$ in integer arithmetic.

We project `hour` and `day_of_week` onto the unit circle:
$$\sin_{\text{hour}} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \cos_{\text{hour}} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
$$\sin_{\text{day}} = \sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right), \quad \cos_{\text{day}} = \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$$

This preserves Euclidean distance $\|\mathbf{x}_{23} - \mathbf{x}_0\|_2 \approx \|\mathbf{x}_1 - \mathbf{x}_0\|_2$.

### B. Autoregressive Lag Features
Guided by our Autocorrelation Function (ACF) discoveries in Phase 2 ($\rho_1 = 0.896, \rho_{24} = 0.708$), we extract:
- Short-term momentum: $\text{lag}_1, \text{lag}_2, \text{lag}_3$
- Intermediate diurnal transition: $\text{lag}_6, \text{lag}_{12}$
- Diurnal seasonal memory: $\text{lag}_{24}$

Each lag explicitly checks whether $t - t_{-k} == k\text{ hours}$ to prevent gap distortion.

### C. Closed-Left Rolling Window Statistics
To measure whether traffic volume is accelerating or decelerating entering a congestion phase, we compute:
- Rolling means: $\mu_3, \mu_6, \mu_{24}$
- Rolling volatilities: $\sigma_3, \sigma_6, \sigma_{24}$

**Strict Leakage Constraint:** Rolling windows are computed using `closed='left'` (i.e., over $\{y_{t-w}, \dots, y_{t-1}\}$), ensuring the observation at time $t$ is strictly decoupled from the historical rolling state.

### D. Regime & Categorical Indicators
- `is_weekend`: Binary flag $\{0, 1\}$.
- `is_rush_hour`: Binary flag $\{0, 1\}$ active during weekday commuter windows (07:00-09:00 and 16:00-18:00).
- `is_holiday`: Binary flag active during US national/state holidays.
- `weather_main`: One-hot encoded categorical vector (Clear, Rain, Clouds, Snow, Mist, Fog, Drizzle, Haze, Thunderstorm, Squall, Smoke).

---

## 3. Preprocessing & Leakage Prevention Guarantees

1. **Independent Preprocessing:** Preprocessing is implemented in `TrafficPreprocessor` (`src/data_preprocessing.py`) using scikit-learn `ColumnTransformer`.
2. **Train-Only Fitting:** The mean, standard deviation, and categorical vocabularies are fitted **strictly on `train.csv`**. 
3. **Artifact Persistence:** The fitted transformer is saved to `models/preprocessor.joblib`. During validation, testing, and production FastAPI inference, the preprocessor transforms incoming vectors using previously learned scaling parameters without refitting.
