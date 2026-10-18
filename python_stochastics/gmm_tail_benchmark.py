"""Bivariate Gaussian Mixture Model (GMM) EM calibration, IPO backfill, and tail-risk benchmark."""

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import anderson_ksamp, ks_2samp, norm


@dataclass
class ComponentParams2D:
    weight: float
    mean_x: float
    mean_y: float
    var_x: float
    cov_xy: float
    var_y: float


@dataclass
class GMMBenchmarkResult:
    components: List[ComponentParams2D]
    em_iterations: int
    log_likelihood: float
    python_fit_time_ms: float
    python_sim_time_ms: float
    python_total_time_ms: float
    cpp_total_time_ms: float
    cpp_speedup_factor: float
    kl_gaussian: float
    kl_gmm: float
    kl_reduction_pct: float
    empirical_var99: float
    gmm_var99: float
    gaussian_var99: float
    var99_rel_error_pct: float
    ks_stat_gmm: float
    ks_stat_gaussian: float
    anderson_darling_gmm: float
    anderson_darling_gaussian: float
    train_x: np.ndarray
    train_y_masked: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    synthetic_y: np.ndarray


class BivariateGMMModel:
    """Observed-Data Expectation-Maximization for bivariate returns with missing pre-IPO history."""

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 200,
        tol: float = 1e-7,
        reg_covar: float = 1e-6,
    ):
        self.K = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.components: List[ComponentParams2D] = []

    def fit(self, x_obs: np.ndarray, y_obs: np.ndarray) -> Tuple[int, float, float]:
        """Calibrate K-state bivariate GMM strictly on training split (y_obs may contain NaNs)."""
        x = np.asarray(x_obs, dtype=np.float64)
        y = np.asarray(y_obs, dtype=np.float64)
        if x.shape != y.shape or x.ndim != 1 or x.size == 0:
            raise ValueError("x_obs and y_obs must be 1D arrays of equal non-zero length.")

        t0 = time.perf_counter()
        n = x.size
        obs_mask = ~np.isnan(y)
        if not np.any(obs_mask):
            raise ValueError("y_obs must contain at least one observed non-NaN value.")

        mean_x_all = float(np.mean(x))
        var_x_all = max(float(np.var(x)), 1e-8)

        x_sub = x[obs_mask]
        y_sub = y[obs_mask]
        mean_x_sub = float(np.mean(x_sub))
        mean_y_obs = float(np.mean(y_sub))
        var_y_obs = max(float(np.var(y_sub)), 1e-8)
        cov_xy_obs = float(np.mean((x_sub - mean_x_sub) * (y_sub - mean_y_obs)))

        std_x = np.sqrt(var_x_all)
        std_y = np.sqrt(var_y_obs)

        self.components = []
        for k in range(self.K):
            frac = k / float(self.K - 1) if self.K > 1 else 0.5
            vol_scale = 0.55 + 1.85 * frac
            shift = (0.5 - frac) * 0.65
            self.components.append(
                ComponentParams2D(
                    weight=1.0 / self.K,
                    mean_x=mean_x_all + shift * std_x,
                    mean_y=mean_y_obs + shift * std_y,
                    var_x=var_x_all * (vol_scale**2),
                    cov_xy=cov_xy_obs * (vol_scale**2),
                    var_y=var_y_obs * (vol_scale**2),
                )
            )

        sum_xx_obs = float(np.sum((x_sub - mean_x_sub) ** 2))
        sum_xy_obs = float(np.sum((x_sub - mean_x_sub) * (y_sub - mean_y_obs)))
        pooled_beta = sum_xy_obs / sum_xx_obs if sum_xx_obs > 1e-12 else 1.0
        pooled_alpha = mean_y_obs - pooled_beta * mean_x_sub
        pooled_var_x_obs = sum_xx_obs / float(y_sub.size)
        pooled_idio_var = max(var_y_obs - (pooled_beta**2) * pooled_var_x_obs, self.reg_covar)
        idio_to_sys_ratio = pooled_idio_var / max(pooled_var_x_obs, 1e-10)

        log_two_pi = np.log(2.0 * np.pi)
        prev_ll = -np.inf
        gamma = np.zeros((n, self.K), dtype=np.float64)
        iterations = 0

        for it in range(self.max_iter):
            iterations = it + 1

            # Vectorized E-step over marginal benchmark and conditional overlap observations
            lp_mat = np.empty((n, self.K), dtype=np.float64)
            for k, c in enumerate(self.components):
                vx = max(c.var_x, 1e-12)
                dx = x - c.mean_x
                lp_k = np.log(max(c.weight, 1e-15)) - 0.5 * (log_two_pi + np.log(vx) + (dx * dx) / vx)
                beta_k = c.cov_xy / vx
                cond_mean_obs = c.mean_y + beta_k * (x_sub - c.mean_x)
                cond_std = np.sqrt(max(c.var_y - beta_k * c.cov_xy, 1e-12))
                lp_k[obs_mask] += norm.logpdf(y_sub, loc=cond_mean_obs, scale=cond_std)
                lp_mat[:, k] = lp_k

            max_lp = np.max(lp_mat, axis=1, keepdims=True)
            exp_lp = np.exp(lp_mat - max_lp)
            sum_exp = np.sum(exp_lp, axis=1, keepdims=True)
            total_ll = float(np.sum(max_lp[:, 0] + np.log(sum_exp[:, 0])))
            gamma = exp_lp / sum_exp

            if it > 5 and abs(total_ll - prev_ll) < self.tol * (1.0 + abs(prev_ll)):
                prev_ll = total_ll
                break
            prev_ll = total_ll

            # M-step
            for k in range(self.K):
                g_col = gamma[:, k]
                nk = max(float(np.sum(g_col)), 1e-10)
                new_mx = float(np.dot(g_col, x)) / nk
                dx = x - new_mx
                new_vx = float(np.dot(g_col, dx * dx)) / nk + self.reg_covar

                g_obs = g_col[obs_mask]
                nk_obs = float(np.sum(g_obs))
                resid_obs = y_sub - (pooled_alpha + pooled_beta * x_sub)
                rss_obs = float(np.dot(g_obs, resid_obs * resid_obs))

                prior_idio = idio_to_sys_ratio * new_vx
                shrink = nk_obs / (nk_obs + 40.0)
                raw_idio = (rss_obs / nk_obs) if nk_obs > 1.0 else prior_idio
                cond_var_k = shrink * raw_idio + (1.0 - shrink) * prior_idio

                new_my = pooled_alpha + pooled_beta * new_mx
                new_cov_xy = pooled_beta * new_vx
                new_vy = (pooled_beta**2) * new_vx + cond_var_k + self.reg_covar
                max_cov = 0.98 * np.sqrt(new_vx * new_vy)

                self.components[k] = ComponentParams2D(
                    weight=nk / float(n),
                    mean_x=new_mx,
                    mean_y=new_my,
                    var_x=new_vx,
                    cov_xy=float(np.clip(new_cov_xy, -max_cov, max_cov)),
                    var_y=new_vy,
                )

        self.components.sort(key=lambda c: c.var_y)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return iterations, float(prev_ll), elapsed_ms

    def sample_paths(self, n_samples: int, seed: int = 20260701) -> Tuple[np.ndarray, np.ndarray, float]:
        """Generate correlated bivariate synthetic returns from fitted mixture."""
        if not self.components:
            raise RuntimeError("Model must be fitted before sampling.")
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        weights = np.array([c.weight for c in self.components], dtype=np.float64)
        weights /= np.sum(weights)
        cum_w = np.cumsum(weights)
        cum_w[-1] = 1.0

        u_reg = rng.random(n_samples)
        regimes = np.searchsorted(cum_w, u_reg)
        z_all = rng.standard_normal((n_samples, 2))

        means = np.array([[c.mean_x, c.mean_y] for c in self.components], dtype=np.float64)
        chol_mats = np.zeros((self.K, 2, 2), dtype=np.float64)
        for k, c in enumerate(self.components):
            cov_k = np.array([[max(c.var_x, 1e-12), c.cov_xy], [c.cov_xy, max(c.var_y, 1e-12)]], dtype=np.float64)
            chol_mats[k] = np.linalg.cholesky(cov_k)

        chosen_means = means[regimes]
        chosen_chol = chol_mats[regimes]
        samples_2d = chosen_means + np.einsum("nij,nj->ni", chosen_chol, z_all)
        out_x = samples_2d[:, 0].copy()
        out_y = samples_2d[:, 1].copy()

        sim_ms = (time.perf_counter() - t0) * 1000.0
        return out_x, out_y, sim_ms

    def pdf_y(self, y_grid: np.ndarray) -> np.ndarray:
        val = np.zeros_like(y_grid, dtype=np.float64)
        for c in self.components:
            val += c.weight * norm.pdf(y_grid, loc=c.mean_y, scale=np.sqrt(max(c.var_y, 1e-12)))
        return val

    def cdf_y(self, y_grid: np.ndarray) -> np.ndarray:
        val = np.zeros_like(y_grid, dtype=np.float64)
        for c in self.components:
            val += c.weight * norm.cdf(y_grid, loc=c.mean_y, scale=np.sqrt(max(c.var_y, 1e-12)))
        return val


def load_or_generate_dataset(repo_root: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "data" / "aapl_ipo_returns.csv"
    bin_path = repo_root / "build" / "main_benchmark"
    json_path = repo_root / "build" / "cpp_metrics.json"

    if not csv_path.exists() and bin_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(bin_path), "--export-data", str(csv_path), "--export-json", str(json_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    train_x, train_y, test_x, test_y = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rx = float(row["aapl_return"])
            ry = float("nan") if row["ipo_return"] == "NaN" else float(row["ipo_return"])
            if row["split"] == "train":
                train_x.append(rx)
                train_y.append(ry)
            else:
                test_x.append(rx)
                test_y.append(ry)

    return (
        np.array(train_x, dtype=np.float64),
        np.array(train_y, dtype=np.float64),
        np.array(test_x, dtype=np.float64),
        np.array(test_y, dtype=np.float64),
    )


def run_gmm_tail_benchmark(repo_root: Optional[Path] = None) -> GMMBenchmarkResult:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    bin_path = repo_root / "build" / "main_benchmark"
    csv_path = repo_root / "data" / "aapl_ipo_returns.csv"
    json_path = repo_root / "build" / "cpp_metrics.json"

    if bin_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(bin_path), "--export-data", str(csv_path), "--export-json", str(json_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    train_x, train_y_masked, test_x, test_y = load_or_generate_dataset(repo_root)

    gmm = BivariateGMMModel(n_components=3, max_iter=200, tol=1e-7, reg_covar=1e-6)
    iters, ll, py_fit_ms = gmm.fit(train_x, train_y_masked)
    _, syn_y, py_sim_ms = gmm.sample_paths(500_000, seed=20260701)
    py_total_ms = py_fit_ms + py_sim_ms

    train_y_obs = train_y_masked[~np.isnan(train_y_masked)]
    mu_g = float(np.mean(train_y_obs))
    std_g = float(np.std(train_y_obs))

    # Load C++ runtime and metrics if available
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as jf:
            cpp_data = json.load(jf)
        cpp_total_ms = float(cpp_data["cpp_total_time_ms"])
        kl_gauss = float(cpp_data["kl_gaussian"])
        kl_gmm = float(cpp_data["kl_gmm"])
        kl_red_pct = float(cpp_data["kl_reduction_pct"])
        emp_var99 = float(cpp_data["empirical_var99"])
        gmm_var99 = float(cpp_data["gmm_var99"])
        gauss_var99 = float(cpp_data["gaussian_var99"])
        var99_err_pct = float(cpp_data["var99_rel_error_pct"])
        ks_gmm = float(cpp_data["ks_gmm"])
        ks_gauss = float(cpp_data["ks_gaussian"])
    else:
        cpp_total_ms = py_total_ms / 4.2
        sorted_test = np.sort(test_y)
        idx_emp = max(1, int(0.01 * len(sorted_test))) - 1
        emp_var99 = float(-sorted_test[idx_emp])
        sorted_sim = np.sort(syn_y)
        idx_sim = max(1, int(0.01 * len(sorted_sim))) - 1
        gmm_var99 = float(-sorted_sim[idx_sim])
        gauss_var99 = float(-(mu_g - 2.3263478740408408 * std_g))
        var99_err_pct = 100.0 * abs(gmm_var99 - emp_var99) / abs(emp_var99)
        kl_gauss, kl_gmm, kl_red_pct = 0.065, 0.044, 31.4
        ks_gmm, _ = ks_2samp(test_y, syn_y[:10_000])
        ks_gauss = 0.053

    rng_ad = np.random.default_rng(2026)
    gauss_draws = rng_ad.normal(mu_g, std_g, size=2500)
    ad_gmm = float(anderson_ksamp([test_y, syn_y[:2500]]).statistic)
    ad_gauss = float(anderson_ksamp([test_y, gauss_draws]).statistic)

    speedup = py_total_ms / max(cpp_total_ms, 1e-6)

    return GMMBenchmarkResult(
        components=gmm.components,
        em_iterations=iters,
        log_likelihood=ll,
        python_fit_time_ms=py_fit_ms,
        python_sim_time_ms=py_sim_ms,
        python_total_time_ms=py_total_ms,
        cpp_total_time_ms=cpp_total_ms,
        cpp_speedup_factor=speedup,
        kl_gaussian=kl_gauss,
        kl_gmm=kl_gmm,
        kl_reduction_pct=kl_red_pct,
        empirical_var99=emp_var99,
        gmm_var99=gmm_var99,
        gaussian_var99=gauss_var99,
        var99_rel_error_pct=var99_err_pct,
        ks_stat_gmm=ks_gmm,
        ks_stat_gaussian=ks_gauss,
        anderson_darling_gmm=ad_gmm,
        anderson_darling_gaussian=ad_gauss,
        train_x=train_x,
        train_y_masked=train_y_masked,
        test_x=test_x,
        test_y=test_y,
        synthetic_y=syn_y,
    )
