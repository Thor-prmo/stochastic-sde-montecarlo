"""Finite-state Markov chains, Perron-Frobenius mixing bounds, Gambler's Ruin, and Barrier MC."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class GamblersRuinResult:
    initial_states: np.ndarray
    empirical_win_prob: np.ndarray
    analytical_win_prob: np.ndarray
    empirical_duration: np.ndarray
    analytical_duration: np.ndarray
    max_abs_prob_error: float


@dataclass
class PerronMixingResult:
    steps: np.ndarray
    exact_tv_distance: np.ndarray
    empirical_mcmc_tv: np.ndarray
    perron_frobenius_bound: np.ndarray
    second_eigenvalue_mod: float
    spectral_gap: float
    stationary_distribution: np.ndarray


@dataclass
class BarrierOptionPythonResult:
    analytical_vanilla: float
    analytical_barrier: float
    crude_price: float
    crude_se: float
    antithetic_price: float
    antithetic_se: float
    controlled_price: float
    controlled_se: float
    variance_reduction_factor: float
    relative_error_pct: float
    sample_sizes: np.ndarray
    crude_se_curve: np.ndarray
    antithetic_se_curve: np.ndarray
    controlled_se_curve: np.ndarray


class GamblersRuinSimulator:
    """Martingale analysis and Monte Carlo verification of Gambler's Ruin on {0, 1, ..., N}."""

    def __init__(self, target_wealth: int = 20, p_up: float = 0.48, seed: int = 202605):
        if target_wealth < 2:
            raise ValueError("target_wealth must be >= 2.")
        if not (0.0 < p_up < 1.0):
            raise ValueError("p_up must lie strictly in (0, 1).")
        self.N = int(target_wealth)
        self.p = float(p_up)
        self.q = 1.0 - self.p
        self.seed = int(seed)

    def analytical_curves(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Closed-form absorption probabilities and expected stopping times via Doob's optional stopping."""
        s = states.astype(np.float64)
        if abs(self.p - 0.5) < 1e-12:
            win_prob = s / self.N
            duration = s * (self.N - s)
        else:
            ratio = self.q / self.p
            win_prob = (1.0 - np.power(ratio, s)) / (1.0 - np.power(ratio, self.N))
            drift = self.p - self.q
            duration = (self.N * win_prob - s) / drift
        return win_prob, duration

    def run_simulation(self, n_trials_per_state: int = 12_000) -> GamblersRuinResult:
        rng = np.random.default_rng(self.seed)
        states = np.arange(1, self.N)
        emp_prob = np.zeros_like(states, dtype=np.float64)
        emp_dur = np.zeros_like(states, dtype=np.float64)

        for idx, x0 in enumerate(states):
            wealth = np.full(n_trials_per_state, x0, dtype=np.int32)
            active = np.ones(n_trials_per_state, dtype=bool)
            steps = np.zeros(n_trials_per_state, dtype=np.int32)

            while np.any(active):
                n_act = int(np.sum(active))
                draws = rng.random(n_act) < self.p
                delta = np.where(draws, 1, -1)
                wealth[active] += delta
                steps[active] += 1
                active = (wealth > 0) & (wealth < self.N)

            emp_prob[idx] = float(np.mean(wealth == self.N))
            emp_dur[idx] = float(np.mean(steps))

        ana_prob, ana_dur = self.analytical_curves(states)
        max_err = float(np.max(np.abs(emp_prob - ana_prob)))

        return GamblersRuinResult(
            initial_states=states,
            empirical_win_prob=emp_prob,
            analytical_win_prob=ana_prob,
            empirical_duration=emp_dur,
            analytical_duration=ana_dur,
            max_abs_prob_error=max_err,
        )


class MetropolisHastingsMCMC:
    """Metropolis-Hastings Markov chain mixer and Perron-Frobenius spectral gap verifier."""

    def __init__(self, n_states: int = 16, beta_inv_temp: float = 1.15, seed: int = 202607):
        self.n_states = int(n_states)
        self.beta = float(beta_inv_temp)
        self.seed = int(seed)
        self.target_pi = self._build_target_distribution()
        self.transition_matrix = self._build_mh_transition_matrix()

    def _build_target_distribution(self) -> np.ndarray:
        grid = np.linspace(-2.0, 2.0, self.n_states)
        potential = 0.45 * (grid**2 - 1.35) ** 2 + 0.12 * grid
        unnorm = np.exp(-self.beta * potential)
        return unnorm / np.sum(unnorm)

    def _build_mh_transition_matrix(self) -> np.ndarray:
        """Construct reversible Metropolis-Hastings transition matrix P on a ring-plus-chord graph."""
        n = self.n_states
        pi = self.target_pi
        Q = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            neighbors = [(i - 1) % n, (i + 1) % n, (i + 3) % n, (i - 3) % n]
            for j in neighbors:
                Q[i, j] += 0.25

        P = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j and Q[i, j] > 0.0:
                    alpha_ij = min(1.0, (pi[j] * Q[j, i]) / (pi[i] * Q[i, j]))
                    P[i, j] = Q[i, j] * alpha_ij
            P[i, i] = 1.0 - np.sum(P[i, :])
        return P

    def analyze_mixing(self, max_steps: int = 50, n_Parallel_chains: int = 40_000) -> PerronMixingResult:
        eigvals = np.linalg.eigvals(self.transition_matrix)
        mods = np.sort(np.abs(eigvals))[::-1]
        lambda_2 = float(mods[1])
        spectral_gap = 1.0 - lambda_2

        steps = np.arange(0, max_steps + 1)
        exact_tv = np.zeros(max_steps + 1, dtype=np.float64)
        P_t = np.eye(self.n_states, dtype=np.float64)

        for t in steps:
            if t > 0:
                P_t = P_t @ self.transition_matrix
            tv_per_start = 0.5 * np.sum(np.abs(P_t - self.target_pi[None, :]), axis=1)
            exact_tv[t] = float(np.max(tv_per_start))

        # Simulate parallel Metropolis-Hastings chains starting from worst-case state
        worst_start = int(np.argmin(self.target_pi))
        rng = np.random.default_rng(self.seed)
        cdf_rows = np.cumsum(self.transition_matrix, axis=1)
        cdf_rows[:, -1] = 1.0

        states = np.full(n_Parallel_chains, worst_start, dtype=np.int32)
        emp_tv = np.zeros(max_steps + 1, dtype=np.float64)

        for t in steps:
            if t > 0:
                u = rng.random(n_Parallel_chains)
                thresholds = cdf_rows[states]
                states = np.argmax(u[:, None] <= thresholds, axis=1)
            counts = np.bincount(states, minlength=self.n_states) / float(n_Parallel_chains)
            emp_tv[t] = 0.5 * float(np.sum(np.abs(counts - self.target_pi)))

        # Theoretical reversible Markov chain bound: ||P^t(x0, .) - pi||_TV <= 0.5 * sqrt((1 - pi(x0))/pi(x0)) * |lambda_2|^t
        pi_min = float(self.target_pi[worst_start])
        prefactor = 0.5 * np.sqrt((1.0 - pi_min) / pi_min)
        pf_bound = np.minimum(1.0, prefactor * np.power(lambda_2, steps))

        return PerronMixingResult(
            steps=steps,
            exact_tv_distance=exact_tv,
            empirical_mcmc_tv=emp_tv,
            perron_frobenius_bound=pf_bound,
            second_eigenvalue_mod=lambda_2,
            spectral_gap=spectral_gap,
            stationary_distribution=self.target_pi.copy(),
        )


class BarrierMonteCarloEngine:
    """Down-and-Out Call option pricer with Antithetic + Control Variates and Brownian Bridge weights."""

    def __init__(
        self,
        s0: float = 100.0,
        strike: float = 100.0,
        barrier: float = 96.26,
        rate: float = 0.045,
        sigma: float = 0.21,
        maturity: float = 1.0,
        n_steps: int = 128,
    ):
        self.s0 = float(s0)
        self.strike = float(strike)
        self.barrier = float(barrier)
        self.rate = float(rate)
        self.sigma = float(sigma)
        self.maturity = float(maturity)
        self.n_steps = int(n_steps)

    @staticmethod
    def black_scholes_call(s0: float, k: float, r: float, sigma: float, t: float) -> float:
        if t <= 0.0:
            return max(s0 - k, 0.0)
        vol_t = sigma * np.sqrt(t)
        d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / vol_t
        d2 = d1 - vol_t
        return float(s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))

    def analytical_down_and_out_call(self) -> float:
        if self.barrier <= 1e-8:
            return self.black_scholes_call(self.s0, self.strike, self.rate, self.sigma, self.maturity)
        if self.s0 <= self.barrier:
            return 0.0

        s0, k, b, r, sigma, t = self.s0, self.strike, self.barrier, self.rate, self.sigma, self.maturity
        vol_t = sigma * np.sqrt(t)
        lam = (r + 0.5 * sigma**2) / (sigma**2)
        vanilla = self.black_scholes_call(s0, k, r, sigma, t)

        if b <= k:
            y = np.log((b * b) / (s0 * k)) / vol_t + lam * vol_t
            c_di = (
                s0 * ((b / s0) ** (2.0 * lam)) * norm.cdf(y)
                - k * np.exp(-r * t) * ((b / s0) ** (2.0 * lam - 2.0)) * norm.cdf(y - vol_t)
            )
            return float(max(vanilla - c_di, 0.0))
        else:
            x1 = np.log(s0 / b) / vol_t + lam * vol_t
            y1 = np.log(b / s0) / vol_t + lam * vol_t
            c_do = (
                s0 * norm.cdf(x1)
                - k * np.exp(-r * t) * norm.cdf(x1 - vol_t)
                - s0 * ((b / s0) ** (2.0 * lam)) * norm.cdf(y1)
                + k * np.exp(-r * t) * ((b / s0) ** (2.0 * lam - 2.0)) * norm.cdf(y1 - vol_t)
            )
            return float(max(c_do, 0.0))

    def price(self, n_paths: int = 120_000, seed: int = 18) -> BarrierOptionPythonResult:
        rng = np.random.default_rng(seed)
        n_pairs = max(1, n_paths // 2)
        dt = self.maturity / self.n_steps
        drift = (self.rate - 0.5 * self.sigma**2) * dt
        vol_dt = self.sigma * np.sqrt(dt)
        var_dt = (self.sigma**2) * dt
        disc = np.exp(-self.rate * self.maturity)

        bs_vanilla = self.black_scholes_call(self.s0, self.strike, self.rate, self.sigma, self.maturity)
        bs_barrier = self.analytical_down_and_out_call()

        z = rng.standard_normal((n_pairs, self.n_steps))

        def eval_batch(z_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            log_inc = drift + vol_dt * z_mat
            log_s = np.empty((n_pairs, self.n_steps + 1), dtype=np.float64)
            log_s[:, 0] = np.log(self.s0)
            log_s[:, 1:] = np.log(self.s0) + np.cumsum(log_inc, axis=1)
            s_paths = np.exp(log_s)

            s_T = s_paths[:, -1]
            vanilla_payoff = disc * np.maximum(s_T - self.strike, 0.0)

            if self.barrier <= 1e-8:
                return vanilla_payoff.copy(), vanilla_payoff

            b = self.barrier
            alive = np.all(s_paths > b, axis=1)
            survival = np.zeros(n_pairs, dtype=np.float64)

            if np.any(alive):
                log_ratio = np.log(s_paths[alive] / b)
                log_prev = log_ratio[:, :-1]
                log_curr = log_ratio[:, 1:]
                p_cross = np.exp(-2.0 * log_prev * log_curr / var_dt)
                survival[alive] = np.prod(np.maximum(0.0, 1.0 - p_cross), axis=1)

            return survival * vanilla_payoff, vanilla_payoff

        bar_pos, van_pos = eval_batch(z)
        bar_neg, van_neg = eval_batch(-z)

        crude_samples = bar_pos
        av_bar = 0.5 * (bar_pos + bar_neg)
        av_van = 0.5 * (van_pos + van_neg)

        cov_mat = np.cov(av_bar, av_van, ddof=1)
        beta_opt = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] > 1e-14 else 0.0
        cv_samples = av_bar - beta_opt * (av_van - bs_vanilla)

        crude_var = float(np.var(crude_samples, ddof=1))
        av_var = float(np.var(av_bar, ddof=1))
        cv_var = float(np.var(cv_samples, ddof=1))

        crude_se = float(np.sqrt(crude_var / n_pairs))
        av_se = float(np.sqrt(av_var / n_pairs))
        cv_se = float(np.sqrt(cv_var / n_pairs))

        cv_mean = float(np.mean(cv_samples))
        vr_factor = crude_var / max(cv_var, 1e-14)
        rel_err_pct = 100.0 * abs(cv_mean - bs_barrier) / max(bs_barrier, 1e-12)

        sample_grid = np.unique(np.logspace(2.5, np.log10(n_pairs), 18).astype(int))
        crude_curve = np.sqrt(crude_var / sample_grid)
        av_curve = np.sqrt(av_var / sample_grid)
        cv_curve = np.sqrt(cv_var / sample_grid)

        return BarrierOptionPythonResult(
            analytical_vanilla=bs_vanilla,
            analytical_barrier=bs_barrier,
            crude_price=float(np.mean(crude_samples)),
            crude_se=crude_se,
            antithetic_price=float(np.mean(av_bar)),
            antithetic_se=av_se,
            controlled_price=cv_mean,
            controlled_se=cv_se,
            variance_reduction_factor=vr_factor,
            relative_error_pct=rel_err_pct,
            sample_sizes=sample_grid,
            crude_se_curve=crude_curve,
            antithetic_se_curve=av_curve,
            controlled_se_curve=cv_curve,
        )


def run_markov_and_barrier_benchmark() -> Dict[str, object]:
    ruin = GamblersRuinSimulator(target_wealth=20, p_up=0.48, seed=202605).run_simulation(12_000)
    mixing = MetropolisHastingsMCMC(n_states=16, beta_inv_temp=1.15, seed=202607).analyze_mixing(
        max_steps=50, n_Parallel_chains=40_000
    )
    barrier = BarrierMonteCarloEngine().price(n_paths=120_000, seed=18)
    return {
        "gamblers_ruin": ruin,
        "mcmc_mixing": mixing,
        "barrier_mc": barrier,
    }
