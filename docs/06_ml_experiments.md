# 06 — Traditional Machine Learning Experiments & Tradeoff Analysis

## 1. Experimental Setup & Hypotheses
We benchmark three representative model families on our 39-dimensional engineered feature space:
1. **Hypothesis 1 (Linear Sufficiency):** Can an $L_2$-regularized linear model (Ridge) capture traffic dynamics through engineered cyclical and lag features?
2. **Hypothesis 2 (Ensemble Non-linearities):** Do tree ensembles (Random Forest, XGBoost) capture non-linear interactions between weather shocks, weekend status, and diurnal rush hours?
3. **Hypothesis 3 (Compute vs. Accuracy Tradeoff):** How does gradient boosting compare to bagging in terms of training latency and generalization error?

---

## 2. Experimental Results Leaderboard

All metrics evaluated on chronological validation and test partitions:

| Model | Model Family | Val MAE | Val RMSE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Regressor** | Gradient Boosted Trees | 156.07 | 240.28 | 0.9850 | **145.79** | **221.43** | **0.9875** | **0.85s** |
| **Random Forest** | Bagged Decision Trees | **154.12** | 240.33 | **0.9850** | 147.58 | 226.20 | 0.9869 | 6.32s |
| **Linear Regression (Ridge)**| Regularized Linear | 326.24 | 450.12 | 0.9475 | 296.92 | 409.35 | 0.9572 | **0.02s** |
| **Persistence Baseline** | Baseline | 588.04 | 823.55 | 0.8242 | 589.64 | 814.35 | 0.8304 | 0.00s |
| **Mean Baseline** | Baseline | 1,723.00 | 1,968.55 | -0.0044 | 1,735.79 | 1,980.81 | -0.0033 | 0.00s |

---

## 3. Analysis & Applied Scientist Interview Discussion Points

### A. Linear Regression vs. Persistence
Ridge regression achieves $\text{Test MAE} = 296.92$, cutting the persistence error by **$50\%$**. This proves that our engineered features (cyclical encodings, rolling averages, 24-hour lags) provide strong linear predictive signal.

### B. Tree Non-Linearity Gains
Moving from Linear Regression to XGBoost drops Test MAE from $296.92$ down to $145.79$ (**50.9% further error reduction**). Tree ensembles excel because they can carve orthogonal decision boundaries around discrete regimes (e.g. `is_rush_hour == 1` AND `is_weekend == 0` AND `rain_1h > 2.0`).

### C. Random Forest vs. XGBoost Tradeoffs
- **Accuracy:** Generalization performance is nearly identical (Random Forest Test MAE: 147.58 vs. XGBoost Test MAE: 145.79).
- **Inference & Training Latency:** XGBoost trains in **$0.85$ seconds** compared to **$6.32$ seconds** for Random Forest ($7.4\times$ faster). The gradient boosting depth regularization (`max_depth=6`) and histogram-based split finding provide dramatic computational advantages for continuous retraining pipelines.
