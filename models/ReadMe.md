# 🛸 Project Dawama: Technical Architectural Framework

Project **Dawama** (دوّامة) shifts AI-driven aerodynamic modeling from **instantaneous mapping** to **stateful operator sequence tracking**. This shift is what allowed the Super OFormer to easily beat the standard model during stress tests. 

Below is the complete engineering breakdown of how both models process data, how their internals differ, and why the math behind the Super variant fixes the real-world physical issues.

---

## 🏗️ Technical Architecture & System Evolution

The core problem of standard Operator Learning frameworks (like DeepONet) is their **instantaneous nature**—they lack context regarding how physical states evolve across a timeline. Project Dawama directly modifies these component mathematical blocks to enforce temporal continuity and robustness.

### A. The Structural Input Pipeline
The dataset models a continuous physical system. Instead of shuffling inputs randomly, data is streamed sequentially. Telemetry streams consist of $u$ (8 surface pressure sensors mounted on the drone chassis) and $y = (x, y)$ coordinate queries representing spatial locations relative to a moving aerodynamic vortex center.

### B. Branch Network: Instantaneous Context vs. Stateful Fusion

* **The Baseline (`OFormerDawama`):** Maps instantaneous sensor values directly to the cross-attention layers. Under heavy wind noise or sensor failure, this architecture causes performance to plummet because the model has no baseline context to verify if the incoming step is physically realistic.
* **The Upgraded Super Block (`FittedDawamaOFormer`):** Retains a hidden state (`prev_latent`) across flight sequences. It concatenates the current sensor tokens with the historical path trace, passing them through a linear compression layer.

**Physical Impact:** Under **Critical Sensor Failure** (Sensors 3 and 7 zeroed out), the standard model drops to a failing **74.64%** accuracy. The Super variant utilizes this exact recurrent loop to infer the missing pressure values based on the trajectory path history, maintaining a safe **86.10%** $R^2$ rating.

---

### C. Trunk Network: Overcoming Spectral Bias via RFF

Both configurations implement a **Random Fourier Feature (RFF)** projection layer instead of raw continuous coordinate lines. Standard neural networks suffer from *spectral bias*, natively prioritizing low-frequency smooth patterns while ignoring high-frequency fluid boundaries.

The trunk space explicitly projects coordinates into a high-dimensional harmonic space:

$$\mathbf{B} \in \mathbb{R}^{2 \times \frac{d}{2}}, \quad B_{ij} \sim \mathcal{N}(0, \sigma^2)$$

$$\gamma(y) = [\sin(y\mathbf{B}), \cos(y\mathbf{B})]$$

**Physical Impact:** This transformation maps simple 2D coordinate positions into a dense space built on varying sinusoidal frequencies. This allows the trunk net to accurately model sharp fluid drops and turbulent velocity changes happening near vortex boundaries.

---

### D. Upgraded Cross-Attention: 2D Rotary Positional Encoding (RoPE)

* **The Baseline (`GalerkinCrossAttention`):** Merges the coordinates ($Q$) and the structural contexts ($K, V$) purely through global matrix multiplications, neglecting localized geometric shifts.
* **The Upgraded Block (`UpgradedRoPEGalerkinAttention`):** Embeds explicit relative coordinate transformations inside the matrix multiplication layer. It slices the hidden vectors and applies continuous coordinate-based rotations via a 2D vector rotation mechanism.

**Physical Impact:** Position is represented as a smooth, continuous rotation. When the drone encounters an **Out-of-Bounds Shift (1.5x)**, the standard network degrades because it hits unseen values. The Super OFormer handles this easily because the latent vectors simply continue their mathematical rotation naturally, keeping an $R^2$ score of **98.32%**.

---

## 📊 Performance Verification

The systems were benchmarked head-to-head on the dynamic flight path trajectory dataset (using sequence-preserving batches where `batch_size=100` tracks a continuous trajectory loop).

| Scenario | Standard MSE | Standard $R^2$ (%) | Super MSE | Super $R^2$ (%) | Operational Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Clean Test Data** | `0.9565` | 98.32% | **`0.3827`** | **99.33%** | High-fidelity fluid alignment. |
| **2. Heavy Wind Noise** | `3.3105` | 94.18% | **`2.0845`** | **96.33%** | Cleans out structural vibration noise. |
| **3. Critical Failure** | `14.4118` | 74.64% | **`7.8996`** | **86.10%** | Infers dead telemetry over time. |
| **4. Out-of-Bounds (1.5x)** | `3.3026` | 94.19% | **`0.9546`** | **98.32%** | Extrapolates past trained grid maps. |