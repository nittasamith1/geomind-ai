# 07 — Deep Learning: Sequential Recurrent Architectures (LSTM & GRU)

## 1. Problem Formulation & Sequential Representation
In traditional machine learning (Phase 4), temporal dependencies were captured via stationary point-in-time lags ($\text{lag}_1, \text{lag}_{24}$) and static summary statistics (rolling mean, rolling std).

In Deep Learning, we formulate the input as an **ordered temporal trajectory**:
$$\mathbf{X}_{t}^{(L)} = \left[ \mathbf{x}_{t-L+1}, \mathbf{x}_{t-L+2}, \dots, \mathbf{x}_t \right] \in \mathbb{R}^{L \times D}$$

Where:
- $L$: Temporal lookback horizon (context length in hours, e.g., $L \in \{6, 12, 24\}$).
- $D = 39$: Feature dimensionality at each hourly step.
- Objective: Predict next-hour traffic count $\hat{y}_{t+1} \in \mathbb{R}_{\ge 0}$.

---

## 2. Neural Architecture Design & Hyperparameter Rationale

### A. Gated Recurrent Units: LSTM & GRU
1. **Long Short-Term Memory (LSTM):**
   Maintains dual cell state $\mathbf{c}_t$ and hidden state $\mathbf{h}_t$ governed by input, forget, and output gates:
   $$\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f)$$
   $$\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$$
   $$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c)$$
   $$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
   $$\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o)$$
   $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

2. **Gated Recurrent Unit (GRU):**
   Merges cell and hidden states into a single state $\mathbf{h}_t$, controlled by reset $\mathbf{r}_t$ and update $\mathbf{z}_t$ gates with $\sim 25\%$ fewer parameters:
   $$\mathbf{z}_t = \sigma(\mathbf{W}_z \mathbf{x}_t + \mathbf{U}_z \mathbf{h}_{t-1})$$
   $$\mathbf{r}_t = \sigma(\mathbf{W}_r \mathbf{x}_t + \mathbf{U}_r \mathbf{h}_{t-1})$$
   $$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \mathbf{x}_t + \mathbf{U}_h (\mathbf{r}_t \odot \mathbf{h}_{t-1}))$$
   $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### B. Hyperparameter Specifications & Technical Rationale
- **Input Dimension ($D = 39$):** Encompasses continuous sensor measurements, weather categoricals, cyclical encodings, and rolling dynamics.
- **Hidden Dimension ($H = 64$):** Sufficient representational capacity to capture highway flow dynamics without overfitting.
- **Number of Layers ($2$):** Enables hierarchical temporal feature abstraction; layer 1 models local hourly rate-of-change, layer 2 models diurnal macro-patterns.
- **Recurrent Dropout ($0.2$):** Applied between recurrent layers to prevent co-adaptation of hidden units.
- **Loss Function (Huber Loss / Smooth $L_1$):**
  $$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$
  *Rationale:* Huber loss behaves quadratically for small residuals (efficient convergence near the minimum) but transitions to linear penalties for large errors, preventing gradient explosion caused by sudden severe weather shocks.
- **Optimization Strategy:** AdamW ($\eta = 10^{-3}$, weight decay $= 10^{-4}$) paired with gradient clipping ($\|\mathbf{g}\|_2 \le 1.0$) and `ReduceLROnPlateau` scheduler.

---

## 3. Controlled Research Experiments
We investigate two core Applied Scientist questions:
1. **Research Question 1 (Information Horizon):** *Does increasing historical context from $L=6$ to $L=12$ and $L=24$ monotonically improve next-hour forecasting accuracy?*
2. **Research Question 2 (Gating Efficiency):** *Does the simplified GRU architecture match or exceed LSTM accuracy while decreasing training latency?*
