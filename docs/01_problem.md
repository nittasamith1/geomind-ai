# 01 — Problem Formulation: Contextual Urban Traffic Forecasting

## 1. Problem Statement
Accurate urban traffic volume forecasting enables dynamic route navigation, fleet optimization, municipal congestion mitigation, and emission reduction. In **GeoMind AI**, we formulate traffic forecasting as an **exogenous contextual and meteorological regression problem** using classical Machine Learning.

In real-world deployment, an end user or planning authority looking to forecast traffic for a given hour **does not know the real-time sensor traffic count** ahead of time. Requiring users to enter the current traffic volume to predict next-hour volume creates a severe operational barrier and introduces autoregressive error compounding during inference.

Therefore, GeoMind formulates the forecasting task to estimate traffic volume purely from **known deterministic temporal signals** and **observable meteorological forecasts**:

$$\hat{y}_{t} = f\left(\mathbf{z}_{t}, \mathbf{w}_{t}\right)$$

Where:
- $\hat{y}_t \in \mathbb{R}_{\ge 0}$: Predicted traffic volume (vehicles per hour) at target hour $t$.
- $\mathbf{z}_t \in \mathbb{R}^p$: Deterministic calendar & temporal features known precisely for any future timestamp (hour of day, day of week, month, day of year, holiday status, rush-hour schedule, cyclical trigonometric projections).
- $\mathbf{w}_t \in \mathbb{R}^d$: Meteorological features obtainable from weather forecasts or local meteorological stations (temperature, rain precipitation, snowfall accumulation, cloud cover percentage, primary weather category).

---

## 2. Research Questions & Design Decisions

1. **Contextual Sufficiency without Autoregressive Lags:**  
   Can classical ML models achieve production-grade predictive accuracy ($R^2 > 0.95$, $\text{MAE} < 280\text{ veh/hr}$) using purely temporal and weather indicators, without relying on real-time traffic volume sensor inputs?
2. **Temporal Dynamics Modeling:**  
   How effectively do continuous cyclical encodings ($\sin/\cos$ transformations across daily, weekly, and annual cycles) combined with discrete regime indicators (commuter rush hours, weekend leisure curves, holiday dampening) capture complex diurnal traffic patterns?
3. **Meteorological Interactions:**  
   How do adverse weather shocks (heavy rainfall, snowfall, sub-zero freezes, thunderstorms) modulate baseline commuter demand, and can non-linear tree ensembles capture these compound friction effects?
4. **Classical ML vs. Heavy Architectures:**  
   Why prioritize classical ML (XGBoost, Random Forest, Ridge Regression)? Classical models deliver sub-millisecond inference latencies (< 5ms), transparent feature importances, lower memory footprints, and instant reproducibility without GPU dependencies.

---

## 3. Evaluation Metrics & Optimization Objective

Traffic volume $y_t$ is a continuous, non-negative quantity. We assess models across three standard statistical metrics:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   *Direct operational interpretability: average vehicle error per hour.*

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
   *Penalizes large outlier forecast errors heavily, crucial for highway congestion planning.*

3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
   *Quantifies the proportion of traffic variance explained by the model relative to the unconditional mean.*
