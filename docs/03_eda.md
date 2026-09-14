# 03 — Exploratory Data Analysis & Statistical Pattern Discovery

## 1. Executive Summary
This document synthesizes the exploratory analysis conducted on 40,575 hourly traffic observations on I-94 Westbound between Minneapolis and St. Paul, MN. 

Rather than generating descriptive plots without a clear scientific hypothesis, every analysis is structured to answer a specific research question relevant to building predictive forecasting models.

---

## 2. Statistical Findings & Empirical Evidence

### A. Probability Density & Bimodal Target Structure
- **Empirical Observation:** The target variable `traffic_volume` has a mean of $3,259.8\text{ veh/hr}$ and a standard deviation of $1,986.9\text{ veh/hr}$, with negative kurtosis ($-1.15$) and mild positive skewness ($+0.12$).
- **Bimodal Density:**
  - **Nocturnal Trough (~500 veh/hr):** Overnight lull (00:00 to 05:00).
  - **Daytime Commuter Plateau (~4,500 - 5,500 veh/hr):** Sustained daytime throughput with commuter peaks.
- **Scientific Implication:** Standard global MSE loss functions penalize quadratic error, which can cause unconditioned models to regress toward the global mean ($\approx 3,260$). Conditioning predictions on temporal cycle features is strictly necessary to resolve the two modes.

### B. Diurnal Commuting Cycles & Weekday/Weekend Asymmetry
- **Weekday Commute (Dual Peak):**
  - **Morning Rush Peak:** Centered at **07:00** ($\sim 6,000\text{ veh/hr}$), characterized by an extremely steep volume ramp between 05:00 and 07:00 ($+3,200\text{ veh/hr}$ acceleration).
  - **Evening Return Rush:** Centered at **16:00–17:00** ($\sim 6,240\text{ veh/hr}$).
- **Weekend Leisure Flow (Unimodal):**
  - Completely lacks morning commuter spikes.
  - Follows a smooth, single-peaked bell curve reaching a maximum around **13:00–14:00** ($\sim 4,400\text{ veh/hr}$).
- **Scientific Implication:** Linear additive models without interaction terms will fail to capture the phase shift between weekdays and weekends. We must construct explicit cross-features or separate cyclical encodings conditioned on `is_weekend`.

### C. Public Holiday Traffic Suppression
- **Impact Measurement:** Normal weekday hourly average is $3,648\text{ veh/hr}$; holiday weekday average drops to $2,260\text{ veh/hr}$ — a **38.0% volume suppression**.
- **Behavioral Shift:** On major national holidays (Thanksgiving, Christmas, Labor Day, Memorial Day), the weekday traffic curve flattens and mirrors Sunday leisure traffic.

### D. Weather Telemetry Interactions
- **Rainfall:** Extreme rain spikes were previously detected and sanitized (e.g. $9,831\text{ mm}$). Real moderate rainfall produces a modest dampening effect on volume ($\sim 3-6\%$).
- **Snow and Squalls:** Severe winter conditions reduce capacity by up to $10-12\%$ due to reduced highway speeds.
- **Scientific Conclusion:** Weather features alone have weak unconditional correlation with volume ($r_{\text{temp}} \approx 0.13, r_{\text{rain}} \approx -0.01$). Weather acts as a non-linear friction coefficient rather than a primary driver of demand.

### E. Autocorrelation Structure & Stationarity
- **Autocorrelation Function (ACF):**
  - Immediate autocorrelation $\rho_1 = 0.896$ ($R^2 \approx 0.80$ from $t-1$ alone).
  - Diurnal autocorrelation $\rho_{24} = 0.708$ (strong 24-hour periodicity).
  - Secondary diurnal $\rho_{48} = 0.672$.
  - Weekly autocorrelation $\rho_{168} = 0.534$ (strong 7-day cyclical memory).
- **Augmented Dickey-Fuller (ADF) Test:**
  - Test Statistic: $-26.85$ ($p < 10^{-15}$).
  - Critical values: $1\%: -3.43, 5\%: -2.86$.
  - **Conclusion:** The traffic volume series is mean-stationary. It does not require first-order differencing ($\Delta y_t$), but heavily requires seasonal lag conditioning.

---

## 3. Directives for Phase 3 (Feature Engineering)

1. **Autoregressive Lags:** Must extract $\text{lag}_1, \text{lag}_2, \text{lag}_3, \text{lag}_6, \text{lag}_{12}, \text{lag}_{24}$.
2. **Rolling Statistics:** Compute 3-hour, 6-hour, and 24-hour moving averages and standard deviations to capture acceleration into rush hour.
3. **Cyclical Temporal Encoding:** Implement $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(2\pi \cdot \text{hour}/24)$ to preserve continuity between 23:00 and 00:00.
4. **Calendar Indicators:** Binary indicators for `is_weekend`, `is_holiday`, and rush-hour regimes.
