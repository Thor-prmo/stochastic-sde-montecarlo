# Continuous-Time Stochastic SDEs, Variance-Reduced Monte Carlo & Generative Market Calibration

[![C++17](https://img.shields.io/badge/C%2B%2B-17%20(-O3%20-march%3Dnative)-00599C.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](#)
[![Tests](https://img.shields.io/badge/pytest-9%20passed-008C2D.svg)](#)
[![Technical Note](https://img.shields.io/badge/PDF-Technical__Note.pdf-8A0076.svg)](./Technical_Note.pdf)

A unified C++17 and Python quantitative research suite implementing **monotone missing-data Gaussian Mixture Model (GMM) Expectation-Maximization** for short-history equity tail-risk backfilling, **variance-reduced barrier option Monte Carlo pricing** with Brownian bridge survival weighting and Bellman MDP execution, **Crank–Nicolson Fokker–Planck PDE density integration**, **100,000-path Euler–Maruyama Langevin SDEs**, and **continuous-time sovereign credit migration generators**.

📄 **[Read the 2-Page Research Summary (`Technical_Note.pdf`)](./Technical_Note.pdf)** | 📐 **[LaTeX Source (`paper/main.tex`)](./paper/main.tex)**

---

## 1. Visual Executive Summary

| **(a) Out-of-Sample GMM Tail Backfill & Q–Q Calibration** | **(b) Variance-Reduced Barrier Option MC & MCMC Mixing** |
| :---: | :---: |
| ![GMM Tail Density and QQ](./figures/fig1_gmm_qq_and_tail_var.png) | ![Barrier MC and MCMC Mixing](./figures/fig2_mc_variance_reduction_convergence.png) |
| **(c) Fokker–Planck PDE vs. 100k-Path Langevin SDE & MSD** | **(d) Sovereign Credit Generator & Par CDS Term Structure** |
| ![Fokker Planck and MSD](./figures/fig3_fokker_planck_density_and_msd.png) | ![Credit Migration and CDS](./figures/fig4_credit_migration_term_structure.png) |

---

## 2. Core Mathematical Formulation

### 2.1 Monotone Missing-Data GMM EM Calibration (Kienitz / Anderson–Stambaugh Factorization)
Let benchmark log-returns $X_t$ (AAPL) be observed over $t \in \{1, \dots, N\}$, while a newly IPO-listed equity $Y_t$ is observed only on the recent post-IPO window $\mathcal{O} = \{N - N_{\mathrm{ipo}} + 1, \dots, N\}$ and unobserved on $\mathcal{M} = \{1, \dots, N - N_{\mathrm{ipo}}\}$. Joint returns follow a $K$-state Gaussian mixture:

$$p(x_t, y_t) = \sum_{k=1}^K \pi_k \,\mathcal{N}\!\left(\begin{bmatrix} x_t \\ y_t \end{bmatrix} \;\middle|\; \begin{bmatrix} \mu_{x,k} \\ \mu_{y,k} \end{bmatrix}, \begin{bmatrix} \Sigma_{xx,k} & \Sigma_{xy,k} \\ \Sigma_{xy,k} & \Sigma_{yy,k} \end{bmatrix}\right)$$

In the **E-step**, regime responsibilities $\gamma_{tk} = \mathbb{P}(Z_t = k \mid \mathcal{D}_t)$ evaluate the bivariate log-density on $t \in \mathcal{O}$ and the marginal benchmark density $\mathcal{N}(x_t \mid \mu_{x,k}, \Sigma_{xx,k})$ on $t \in \mathcal{M}$. In the **M-step**, marginal parameters $(\pi_k, \mu_{x,k}, \Sigma_{xx,k})$ are updated across all $N$ observations, while cross-asset conditional projection $Y_t = \alpha + \beta X_t + \varepsilon_{t,k}$ estimated on $\mathcal{O}$ backfills the missing tail distribution of $Y_t$ via the **Law of Total Covariance**:

$$\mu_{y,k} = \alpha + \beta \mu_{x,k}, \qquad \Sigma_{xy,k} = \beta \Sigma_{xx,k}, \qquad \Sigma_{yy,k} = \beta^2 \Sigma_{xx,k} + \sigma^2_{\varepsilon,k}$$

### 2.2 Brownian Bridge Antithetic + Control Variate Barrier Option Monte Carlo
For a Down-and-Out Call under $dS_t = r S_t\,dt + \sigma S_t\,dW_t$ with strike $K$ and barrier $B < S_0$, Merton's reflection principle yields the exact continuous-monitoring price ($B \le K$):

$$C_{\mathrm{DO}}(S_0, K, B) = C_{\mathrm{BS}}(S_0, K) - \left(\frac{B}{S_0}\right)^{2\lambda} C_{\mathrm{BS}}\!\left(\frac{B^2}{S_0}, K\right), \qquad \lambda = \frac{r + \frac{1}{2}\sigma^2}{\sigma^2}$$

To eliminate discrete monitoring bias across $M$ steps ($\Delta t = T/M$), each surviving trajectory is weighted by its continuous Brownian bridge survival probability $w_{\mathrm{BB}} = \prod_{m=1}^M \left[1 - \exp\!\left(-\frac{2\ln(S_{t_{m-1}}/B)\ln(S_{t_m}/B)}{\sigma^2 \Delta t}\right)\right]$ and paired with antithetic draws $(Z, -Z)$ and the terminal vanilla call control variate $C_i = e^{-rT}(S_T^{(i)} - K)^+$:

$$\hat{V}_{\mathrm{AV+CV}} = \bar{Y}_{\mathrm{AV}} - \hat{\beta}^*\!\left(\bar{C}_{\mathrm{AV}} - C_{\mathrm{BS}}(S_0, K)\right), \qquad \hat{\beta}^* = \frac{\widehat{\mathrm{Cov}}(Y_{\mathrm{AV}}, C_{\mathrm{AV}})}{\widehat{\mathrm{Var}}(C_{\mathrm{AV}})}$$

### 2.3 Fokker–Planck PDE, Langevin SDEs & Continuous-Time Credit Generators
The probability density $p(x,t)$ of an Ornstein–Uhlenbeck Langevin process $dX_t = -\theta(X_t - \mu)\,dt + \sigma\,dW_t$ evolves according to the forward Fokker–Planck PDE:

$$\frac{\partial p(x,t)}{\partial t} = \theta\frac{\partial}{\partial x}\big((x - \mu)p(x,t)\big) + \frac{\sigma^2}{2}\frac{\partial^2 p(x,t)}{\partial x^2}$$

integrated numerically via unconditionally stable Crank–Nicolson finite differences alongside $10^5$ Euler–Maruyama trajectories. For sovereign credit migration across $S=8$ rating states with absorbing Default state $D$, the generator matrix $Q = (q_{ij})$ satisfies $q_{ii} = -\sum_{j \neq i} q_{ij}$, yielding transition matrices $P(t) = \exp(Qt)$ and par CDS spreads $s_i(T) = (1-R)\int_0^T e^{-rt}\,dP_{i,D}(t) \big/ \int_0^T e^{-rt}(1 - P_{i,D}(t))\,dt$.

---

## 3. Empirical & Runtime Benchmark Summary

| Module / Benchmark Metric | Baseline Reference | Implemented Engine | Key Diagnostic |
| :--- | :--- | :--- | :--- |
| **GMM EM + 500k Path Simulation** | `127.5 ms` (Python/SciPy) | **`31.5 ms` (C++17 `-O3`)** | **~4.1x – 4.2x Runtime Speedup** |
| **Out-of-Sample KL Divergence $D_{\mathrm{KL}}$** | `0.0581` (Single Gaussian) | **`0.0395` (3-State GMM)** | **-32.0% Out-of-Sample KL Reduction** |
| **99% Tail Value-at-Risk Accuracy** | `4.58%` (Gaussian, 16.4% err) | **`5.46%` vs. `5.48%` Emp.** | **0.28% Relative Error (< 1.8% Target)** |
| **Kolmogorov–Smirnov Statistic** | `0.0749` (Single Gaussian) | **`0.0616` (3-State GMM)** | **Superior Heavy-Tail Goodness-of-Fit** |
| **Barrier Option MC Standard Error** | `0.0487` (Crude MC) | **`0.0248` (AV + Control Var.)** | **3.83x Monte Carlo Variance Reduction** |
| **Barrier Pricing Error vs. Analytical BS** | `0.110%` (Crude MC) | **`0.079%` (C++17 / Python)** | **Within 0.12% of Black–Scholes** |
| **Free Brownian Diffusion MSD Scaling** | Theoretical $\langle X_t^2 \rangle = 2Dt$ | **`0.166%` Slope Error** | **Verified on 100,000 Paths (< 0.4%)** |
| **Fokker–Planck Crank–Nicolson PDE** | Analytical OU Gaussian PDF | **`0.00012` $L_1$ Error** | **Exact Probability Conservation** |
| **Sovereign CDS Term-Structure Fit** | Empirical Sovereign Quotes | **`2.58 bps` Mean Abs. Error** | **Max Error 8.53 bps (< 15 bps Target)** |

---

## 4. Repository Structure

```text
stochastic-sde-montecarlo/
├── README.md                                 # Mathematical overview, benchmark table & plots
├── Technical_Note.pdf                        # Compiled 2-page two-column research summary
├── CMakeLists.txt                            # C++17 build configuration (-O3 -march=native)
├── Makefile                                  # Build, test, figure generation & suite targets
├── requirements.txt                          # Python scientific dependencies
├── paper/
│   └── main.tex                              # LaTeX source for the 2-page technical note
├── data/
│   └── aapl_ipo_returns.csv                  # Joint AAPL + short-history IPO return dataset
├── cpp_core/
│   ├── include/
│   │   ├── gmm_em.hpp                        # Bivariate K-state GMM EM & Box-Muller sampler header
│   │   └── barrier_mc.hpp                    # Antithetic + Control Variate Barrier MC & Bellman MDP header
│   └── src/
│       ├── gmm_em.cpp                        # Compiled C++17 monotone EM calibration & path simulator
│       ├── barrier_mc.cpp                    # Compiled C++17 barrier option pricer & Bellman MDP solver
│       └── main_benchmark.cpp                # CLI runner exporting dataset, JSON metrics & table
├── python_stochastics/
│   ├── __init__.py
│   ├── gmm_tail_benchmark.py                 # Python GMM EM, KS/Anderson-Darling, KL & 99% VaR evaluator
│   ├── sde_langevin_fokker_planck.py         # 100k-path Euler-Maruyama Brownian/OU SDE & Fokker-Planck PDE
│   ├── markov_mcmc_perron.py                 # Metropolis-Hastings MCMC, Gambler's Ruin & Perron-Frobenius bounds
│   └── credit_migration_cds.py               # Generator matrix Q, Padé exp(Qt) & CDS term-structure pricer
├── scripts/
│   ├── run_full_suite.sh                     # End-to-end C++ build, unit test, figure & PDF pipeline
│   ├── reproduce_all_figures.py              # Generates all 4 high-DPI figures in figures/
│   └── build_technical_note.py               # Compiles the 2-page Technical_Note.pdf
├── figures/
│   ├── fig1_gmm_qq_and_tail_var.png
│   ├── fig2_mc_variance_reduction_convergence.png
│   ├── fig3_fokker_planck_density_and_msd.png
│   └── fig4_credit_migration_term_structure.png
└── tests/
    ├── test_stochastic_calculus.py           # Tests MSD (<0.4% err), Fokker-Planck conservation, Q row-sums & causality
    └── test_black_scholes_limit.py           # Tests B->0 Black-Scholes limit, 3.8x variance reduction & C++ parity
```

---

## 5. One-Command Reproducibility & Invariant Verification

```bash
# 1. Run the complete C++17 build, benchmark, unit test suite, and figure regeneration pipeline
bash scripts/run_full_suite.sh

# 2. Or invoke individual targets via Make
make build      # Compile C++17 static library and main_benchmark binary
make test       # Run all 9 mathematical invariant & anti-leakage unit tests
make plots      # Regenerate all 4 figures in figures/
```

