"""Continuous-time stochastic dynamics, Langevin SDEs, and Fokker-Planck PDE solver."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm


@dataclass
class DiffusionMSDResult:
    time_grid: np.ndarray
    empirical_msd: np.ndarray
    theoretical_msd: np.ndarray
    slope_relative_error_pct: float
    lag1_autocorrelation: float
    sample_trajectories: np.ndarray


@dataclass
class FokkerPlanckComparisonResult:
    x_grid: np.ndarray
    pde_density_transient: np.ndarray
    pde_density_stationary: np.ndarray
    analytical_transient: np.ndarray
    analytical_stationary: np.ndarray
    sde_terminal_samples: np.ndarray
    sde_mid_samples: np.ndarray
    l1_density_error: float


@dataclass
class JumpDiffusionResult:
    time_grid: np.ndarray
    diffusion_returns: np.ndarray
    jump_diffusion_returns: np.ndarray
    diffusion_kurtosis: float
    jump_diffusion_kurtosis: float


class LangevinSDEEngine:
    """Vectorized Euler-Maruyama simulator for Brownian, Ornstein-Uhlenbeck, and Jump-Diffusion SDEs."""

    def __init__(self, n_paths: int = 100_000, seed: int = 202606):
        self.n_paths = int(n_paths)
        self.seed = int(seed)

    def simulate_free_brownian_msd(
        self, diffusion_coeff: float = 0.5, maturity: float = 1.0, n_steps: int = 100
    ) -> DiffusionMSDResult:
        """Simulate dX_t = sqrt(2D) dW_t across n_paths to verify <X_t^2> = 2 D t."""
        rng = np.random.default_rng(self.seed)
        dt = maturity / n_steps
        scale = np.sqrt(2.0 * diffusion_coeff * dt)

        dX = rng.normal(0.0, scale, size=(self.n_paths, n_steps))
        X = np.zeros((self.n_paths, n_steps + 1), dtype=np.float64)
        np.cumsum(dX, axis=1, out=X[:, 1:])

        t_grid = np.linspace(0.0, maturity, n_steps + 1)
        emp_msd = np.mean(X**2, axis=0)
        theo_msd = 2.0 * diffusion_coeff * t_grid

        # Ordinary least squares slope of empirical MSD vs time
        slope_emp = float(np.dot(t_grid[1:], emp_msd[1:]) / np.dot(t_grid[1:], t_grid[1:]))
        slope_theo = 2.0 * diffusion_coeff
        rel_err_pct = 100.0 * abs(slope_emp - slope_theo) / slope_theo

        # White-noise increment autocorrelation at lag 1 across trajectories
        inc_0 = dX[:, :-1].ravel()
        inc_1 = dX[:, 1:].ravel()
        lag1_corr = float(np.corrcoef(inc_0[:200_000], inc_1[:200_000])[0, 1])

        return DiffusionMSDResult(
            time_grid=t_grid,
            empirical_msd=emp_msd,
            theoretical_msd=theo_msd,
            slope_relative_error_pct=rel_err_pct,
            lag1_autocorrelation=lag1_corr,
            sample_trajectories=X[:25, :].copy(),
        )

    def simulate_ornstein_uhlenbeck(
        self,
        theta: float = 2.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        x0: float = 1.2,
        maturity: float = 2.0,
        n_steps: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate Ornstein-Uhlenbeck Langevin dynamics dX_t = -theta(X_t - mu)dt + sigma dW_t."""
        rng = np.random.default_rng(self.seed + 17)
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)
        mid_step = n_steps // 4

        x = np.full(self.n_paths, x0, dtype=np.float64)
        x_mid = np.empty(self.n_paths, dtype=np.float64)

        for step in range(1, n_steps + 1):
            z = rng.standard_normal(self.n_paths)
            x = x - theta * (x - mu) * dt + sigma * sqrt_dt * z
            if step == mid_step:
                x_mid = x.copy()

        t_grid = np.linspace(0.0, maturity, n_steps + 1)
        return t_grid, x_mid, x

    def simulate_drift_diffusion_jump(
        self,
        drift: float = 0.05,
        sigma: float = 0.12,
        jump_lambda: float = 1.2,
        jump_mean: float = -0.04,
        jump_std: float = 0.14,
        maturity: float = 1.0,
        n_steps: int = 252,
    ) -> JumpDiffusionResult:
        """Simulate differential Chapman-Kolmogorov drift-diffusion-jump process."""
        rng = np.random.default_rng(self.seed + 31)
        dt = maturity / n_steps
        sqrt_dt = np.sqrt(dt)

        z = rng.standard_normal((self.n_paths, n_steps))
        diff_steps = (drift - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z
        diff_terminal = np.sum(diff_steps, axis=1)

        n_jumps = rng.poisson(jump_lambda * maturity, size=self.n_paths)
        jump_totals = rng.normal(
            loc=n_jumps * jump_mean,
            scale=np.sqrt(np.maximum(n_jumps, 0)) * jump_std,
        )
        jd_terminal = diff_terminal + jump_totals

        def excess_kurtosis(arr: np.ndarray) -> float:
            centered = arr - np.mean(arr)
            v = np.mean(centered**2)
            return float(np.mean(centered**4) / (v**2) - 3.0)

        return JumpDiffusionResult(
            time_grid=np.linspace(0.0, maturity, n_steps + 1),
            diffusion_returns=diff_terminal,
            jump_diffusion_returns=jd_terminal,
            diffusion_kurtosis=excess_kurtosis(diff_terminal),
            jump_diffusion_kurtosis=excess_kurtosis(jd_terminal),
        )


class FokkerPlanckSolver:
    """Crank-Nicolson finite-difference solver and analytical solutions for 1D Fokker-Planck PDE."""

    def __init__(self, theta: float = 2.0, mu: float = 0.0, sigma: float = 1.0):
        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def analytical_density(self, x: np.ndarray, t: float, x0: float) -> np.ndarray:
        """Exact Gaussian fundamental solution of the Ornstein-Uhlenbeck Fokker-Planck equation."""
        mean_t = self.mu + (x0 - self.mu) * np.exp(-self.theta * t)
        var_t = (self.sigma**2 / (2.0 * self.theta)) * (1.0 - np.exp(-2.0 * self.theta * t))
        return norm.pdf(x, loc=mean_t, scale=np.sqrt(max(var_t, 1e-12)))

    def stationary_density(self, x: np.ndarray) -> np.ndarray:
        """Stationary Boltzmann-Gibbs equilibrium distribution as t -> infinity."""
        var_inf = self.sigma**2 / (2.0 * self.theta)
        return norm.pdf(x, loc=self.mu, scale=np.sqrt(var_inf))

    def solve_crank_nicolson(
        self,
        x0: float = 1.2,
        t_init: float = 0.05,
        t_mid: float = 0.50,
        t_final: float = 2.0,
        n_x: int = 401,
        n_t: int = 400,
        x_lim: float = 4.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve dp/dt = d/dx[theta(x - mu)p] + (sigma^2 / 2) d^2p/dx^2 via Crank-Nicolson."""
        x = np.linspace(-x_lim, x_lim, n_x)
        dx = x[1] - x[0]
        dt = (t_final - t_init) / n_t
        D = 0.5 * self.sigma**2

        p = self.analytical_density(x, t_init, x0)
        p[0] = 0.0
        p[-1] = 0.0

        # Spatial operator L(p)_j = theta * (d/dx[(x_j - mu) p_j]) + D * d^2p_j/dx^2
        drift_vel = self.theta * (x - self.mu)
        alpha = D / (dx**2)

        sub_L = -0.5 * drift_vel[:-1] / dx + alpha
        main_L = self.theta - 2.0 * alpha * np.ones(n_x)
        sup_L = 0.5 * drift_vel[1:] / dx + alpha

        # Enforce zero Dirichlet boundary conditions at +-x_lim
        main_L[0] = 0.0
        sup_L[0] = 0.0
        main_L[-1] = 0.0
        sub_L[-1] = 0.0

        ab = np.zeros((3, n_x), dtype=np.float64)
        ab[0, 1:] = -0.5 * dt * sup_L
        ab[1, :] = 1.0 - 0.5 * dt * main_L
        ab[2, :-1] = -0.5 * dt * sub_L

        mid_step = int(round((t_mid - t_init) / dt))
        p_mid = p.copy()

        for step in range(1, n_t + 1):
            rhs = p.copy()
            rhs[1:-1] = (
                p[1:-1]
                + 0.5 * dt * (
                    sub_L[:-1] * p[:-2]
                    + main_L[1:-1] * p[1:-1]
                    + sup_L[1:] * p[2:]
                )
            )
            rhs[0] = 0.0
            rhs[-1] = 0.0
            p = solve_banded((1, 1), ab, rhs)
            p = np.maximum(p, 0.0)
            norm_const = np.trapz(p, x)
            if norm_const > 0.0:
                p /= norm_const
            if step == mid_step:
                p_mid = p.copy()

        return x, p_mid, p


def run_sde_fokker_planck_benchmark(n_paths: int = 100_000, seed: int = 202606) -> Dict[str, object]:
    """Execute full Brownian MSD, Ornstein-Uhlenbeck Fokker-Planck, and Jump-Diffusion suite."""
    sde = LangevinSDEEngine(n_paths=n_paths, seed=seed)
    msd_res = sde.simulate_free_brownian_msd(diffusion_coeff=0.5, maturity=1.0, n_steps=100)

    theta, mu, sigma, x0 = 2.0, 0.0, 1.0, 1.2
    _, x_mid, x_final = sde.simulate_ornstein_uhlenbeck(
        theta=theta, mu=mu, sigma=sigma, x0=x0, maturity=2.0, n_steps=200
    )

    fp = FokkerPlanckSolver(theta=theta, mu=mu, sigma=sigma)
    x_grid, pde_mid, pde_final = fp.solve_crank_nicolson(
        x0=x0, t_init=0.05, t_mid=0.50, t_final=2.0, n_x=401, n_t=400
    )
    ana_mid = fp.analytical_density(x_grid, t=0.50, x0=x0)
    ana_final = fp.analytical_density(x_grid, t=2.0, x0=x0)
    l1_err = float(np.trapz(np.abs(pde_final - ana_final), x_grid))

    fp_res = FokkerPlanckComparisonResult(
        x_grid=x_grid,
        pde_density_transient=pde_mid,
        pde_density_stationary=pde_final,
        analytical_transient=ana_mid,
        analytical_stationary=ana_final,
        sde_terminal_samples=x_final,
        sde_mid_samples=x_mid,
        l1_density_error=l1_err,
    )

    jd_res = sde.simulate_drift_diffusion_jump()

    return {
        "msd": msd_res,
        "fokker_planck": fp_res,
        "jump_diffusion": jd_res,
    }
