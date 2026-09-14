# 01 — Problem Formulation: Urban Traffic Forecasting & Intelligence

## 1. Problem Statement
Accurate short-term urban traffic forecasting is essential for dynamic route planning, fleet dispatching, carbon emission reduction, and intelligent transportation management. In this project (**GeoMind AI**), we formulate the traffic forecasting task as a multivariate, contextual time-series regression problem.

Given a sequence of historical traffic observations and exogenous environmental factors observed up to time $t$, the objective is to predict the traffic volume at future time steps $t+h$:

$$\hat{y}_{t+h} = f\left(\{y_{t-k}, \mathbf{x}_{t-k}\}_{k=0}^{K}, \mathbf{z}_{t+h}\right)$$

Where:
- $y_t \in \mathbb{R}_{\ge 0}$: Traffic volume (vehicles per hour) at time $t$.
- $\mathbf{x}_t \in \mathbb{R}^d$: Exogenous contextual observations (ambient temperature, rainfall, snow, cloud cover, weather conditions).
- $\mathbf{z}_{t+h} \in \mathbb{R}^p$: Deterministic calendar features known ahead of time (hour of day, day of week, holiday indicator, cyclical sine/cosine encodings).
- $K$: Lookback window (history length, e.g., 12 or 24 previous hours).
- $h \in \{1, 2, \dots, H\}$: Forecast horizon (initially $h=1$ for next-hour prediction, later extended to multi-step $h \in \{1, \dots, 6\}$).

---

## 2. Research Questions (Amazon Applied Scientist Perspective)
An Applied Scientist does not merely fit standard models; we formulate testable empirical hypotheses:

1. **Information Horizon:** How much historical context ($K$) is required to achieve optimal predictive performance? Is a 12-hour or 24-hour sequence sufficient, or does older context introduce noise?
2. **Exogenous vs. Autoregressive Signals:** How much predictive gain comes from weather/calendar features versus pure autoregressive lag terms ($\text{lag}_1, \text{lag}_{24}$)?
3. **Model Inductive Biases:** When do deep sequential architectures (LSTM / GRU) outperform gradient-boosted decision trees (XGBoost / LightGBM) on tabular time-series?
4. **Error Distribution & Failure Modes:** Are forecast errors uniformly distributed, or do they cluster around non-stationary boundary states (e.g., transition into rush hour, sudden torrential downpours, holidays)?
5. **Continuous Adaptation:** How can the model autonomously detect distributional drift and decide when to retrain and promote a new candidate checkpoint?

---

## 3. Evaluation Metrics & Optimization Objective
Traffic volume $y_t$ is continuous and non-negative. We assess models using three complementary statistical metrics:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   *Direct operational interpretability: average vehicle error per hour.*

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
   *Penalizes large errors heavily; critical for congestion spike prevention.*

3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
   *Quantifies percentage of variance explained relative to a naïve mean baseline.*
