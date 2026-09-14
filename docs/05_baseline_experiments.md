# 05 — Statistical Baseline Experiments

## 1. Scientific Objective
In applied machine learning, developing complex neural networks or gradient boosted trees without establishing rigorous statistical baselines is bad scientific practice. Baselines provide:
1. The **lower bound of acceptable predictive utility**.
2. An empirical gauge for how much predictive variance is captured by simple temporal persistence vs. learned patterns.

---

## 2. Implemented Baselines

### A. Mean Baseline
Predicts the unconditional historical training mean $\bar{y}_{\text{train}}$ for every future timestep:
$$\hat{y}_{t+1} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} y_i = 3,259.82\text{ vehicles/hour}$$

### B. Persistence Baseline (Random Walk / Naive Drift)
Predicts that traffic volume in the next hour will equal the volume observed in the current hour:
$$\hat{y}_{t+1} = y_t$$

---

## 3. Empirical Results

| Baseline Model | Val MAE (veh/h) | Val RMSE (veh/h) | Val $R^2$ | Test MAE (veh/h) | Test RMSE (veh/h) | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Baseline** | 1,723.00 | 1,968.55 | -0.0044 | 1,735.79 | 1,980.81 | -0.0033 |
| **Persistence Baseline** | **588.04** | **823.55** | **0.8242** | **589.64** | **814.35** | **0.8304** |

---

## 4. Scientific Insights for Amazon Applied Scientists
1. **The High-Persistence Benchmark:** Persistence alone explains **$82.4\%$ of the variance** in next-hour traffic volume. This aligns with our EDA finding of strong immediate autocorrelation ($\rho_1 = 0.896$). 
2. **The Bar to Beat:** Any candidate ML or Deep Learning model that fails to achieve $\text{MAE} < 588\text{ veh/h}$ is worse than a trivial 0-parameter rule.
