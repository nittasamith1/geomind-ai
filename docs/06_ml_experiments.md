# 06 — Classical Machine Learning Experiments & Leaderboard

## 1. Experimental Setup & Model Families

We evaluate three classical Machine Learning paradigms trained purely on contextual and meteorological inputs (zero dependence on real-time traffic sensor counts):

1. **Regularized Linear Regression (Ridge):**  
   $L_2$-penalized linear baseline with $\alpha = 10.0$ to prevent multicollinearity across cyclical features.
2. **Bagged Decision Trees (Random Forest):**  
   Ensemble of $300$ deep regression trees (`max_depth=20`, `min_samples_leaf=4`, `max_features='sqrt'`).
3. **Gradient Boosted Decision Trees (XGBoost Regressor):**  
   Sequentially boosted trees ($500$ estimators, `max_depth=7`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`, early stopping with validation monitoring).

---

## 2. Empirical Benchmark Leaderboard

All evaluations performed on chronological, non-overlapping train ($70\%$), validation ($15\%$), and test ($15\%$) splits from the Metro Interstate dataset:

| Model | Paradigm | Val MAE (veh/hr) | Val RMSE (veh/hr) | Val $R^2$ | Test MAE (veh/hr) | Test RMSE (veh/hr) | Test $R^2$ | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ⚡ **XGBoost Regressor** | Gradient Boosted Trees | **273.21** | **415.81** | **0.9553** | **243.22** | **410.36** | **0.9570** | **1.57s** |
| 🌲 **Random Forest** | Bagged Decision Trees | 281.27 | 429.56 | 0.9522 | 251.86 | 416.79 | 0.9557 | 2.20s |
| 📈 **Ridge Regression** | Regularized Linear | 796.56 | 1,017.90 | 0.7319 | 791.73 | 999.63 | 0.7450 | **0.07s** |
| 📊 **Unconditional Mean** | Statistical Baseline | 1,723.00 | 1,968.55 | -0.0044 | 1,735.79 | 1,980.81 | -0.0033 | 0.00s |

---

## 3. Analysis & Key Takeaways

### A. Dominance of Gradient Boosting
- **XGBoost achieves the lowest test error:** $\text{Test MAE} = 243.22\text{ veh/hr}$ and $\text{Test } R^2 = 0.9570$.
- Compared to the Ridge linear model, XGBoost reduces mean absolute error by **$69.3\%$** (from $791.73$ down to $243.22$).
- The algorithm excels at learning non-linear step-function boundaries (e.g., sharp volume transitions at 06:00 and 19:00, holiday commutability suppression, and precipitation braking thresholds).

### B. High Accuracy Without Historical Traffic Volume
- A common misconception is that time-series traffic forecasting strictly requires real-time traffic volume inputs ($y_{t-1}$).
- Our findings demonstrate that with comprehensive multi-scale cyclical encodings, commute regimes, and weather friction indicators, **over $95.7\%$ of variance is captured** without needing any current traffic count.
- This creates an intuitive, practical user experience where end users simply specify or auto-fetch date/time and weather parameters.

### C. Latency and Deployability
- **Training Efficiency:** Full pipeline training across 40,000+ records executes in under **4 seconds** total on standard CPU.
- **Inference Speed:** Single-sample inference latency is under **$2\text{ ms}$**, enabling instant real-time predictions in web interfaces and mobile applications.
