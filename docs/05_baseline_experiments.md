# 05 — Statistical Baseline Experiments

## 1. Scientific Objective
In applied machine learning, developing predictive models without establishing rigorous statistical baselines is poor scientific practice. Baselines provide:
1. The **lower bound of acceptable predictive performance**.
2. An empirical gauge for how much predictive variance is captured by simple unconditional averages vs. structured contextual features.

---

## 2. Implemented Baselines

### A. Unconditional Mean Baseline
Predicts the historical training mean $\bar{y}_{\text{train}}$ for every future hour regardless of time or weather:
$$\hat{y}_{t} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} y_i \approx 3,259.8\text{ vehicles/hour}$$

### B. Linear Feature Baseline (Ridge Regression)
Predicts traffic volume as a regularized linear combination of all engineered temporal and meteorological features:
$$\hat{y}_t = \mathbf{w}^T \mathbf{x}_t + b$$
This serves as the benchmark for how much signal can be captured by linear relationships alone without tree-based partitioning.

---

## 3. Empirical Results

Evaluated on the held-out validation ($N=6,086$) and test ($N=6,087$) chronological partitions:

| Baseline Model | Val MAE (veh/hr) | Val RMSE (veh/hr) | Val $R^2$ | Test MAE (veh/hr) | Test RMSE (veh/hr) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unconditional Mean** | 1,723.00 | 1,968.55 | -0.0044 | 1,735.79 | 1,980.81 | -0.0033 |
| **Ridge Regression ($L_2$)** | **796.56** | **1,017.90** | **0.7319** | **791.73** | **999.63** | **0.7450** |

---

## 4. Key Scientific Insights

1. **Linear Predictive Signal ($R^2 = 0.745$):**  
   Pure temporal and meteorological features in a linear model explain nearly **$75\%$ of the total variance** in traffic volume. This demonstrates that traffic patterns are strongly governed by deterministic circadian rhythms and weather conditions.
2. **The Bar for Non-Linear Ensembles:**  
   Any non-linear model (Random Forest, XGBoost) must comfortably outperform the Ridge baseline ($\text{MAE} < 791\text{ veh/hr}$, $R^2 > 0.75$) to justify deployment. As shown in Phase 6, our tree ensembles shatter this bar, reaching **$R^2 = 0.957$** and **$\text{MAE} = 243.2\text{ veh/hr}$**.
