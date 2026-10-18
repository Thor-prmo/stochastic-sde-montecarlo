#!/usr/bin/env python3
"""Generate all four publication-grade figures for the stochastic SDE and Monte Carlo suite."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from python_stochastics import (
    CreditMigrationModel,
    run_credit_migration_benchmark,
    run_gmm_tail_benchmark,
    run_markov_and_barrier_benchmark,
    run_sde_fokker_planck_benchmark,
)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_fig1_gmm_tail(fig_dir: Path) -> dict:
    res = run_gmm_tail_benchmark(REPO_ROOT)
    train_y_obs = res.train_y_masked[~np.isnan(res.train_y_masked)]
    mu_g = float(np.mean(train_y_obs))
    std_g = float(np.std(train_y_obs))

    from python_stochastics.gmm_tail_benchmark import BivariateGMMModel

    gmm = BivariateGMMModel(n_components=3)
    gmm.components = res.components

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.35))

    # Left panel: Log-density tail comparison & 99% VaR thresholds
    y_grid = np.linspace(-0.11, 0.09, 500)
    pdf_gmm = gmm.pdf_y(y_grid)
    pdf_gauss = norm.pdf(y_grid, loc=mu_g, scale=std_g)

    ax1.hist(
        res.test_y,
        bins=38,
        density=True,
        alpha=0.35,
        color="#4C72B0",
        edgecolor="#2b4770",
        label="Out-of-Sample Empirical Returns",
    )
    ax1.plot(y_grid, pdf_gmm, color="#8A0076", lw=2.2, label="3-State EM Backfilled GMM")
    ax1.plot(
        y_grid,
        pdf_gauss,
        color="#D55E00",
        lw=1.8,
        ls="--",
        label="Single-Gaussian MLE Baseline",
    )

    ax1.axvline(-res.empirical_var99, color="#111111", lw=1.5, ls="-", label=f"Empirical 99% VaR ({100*res.empirical_var99:.2f}%)")
    ax1.axvline(-res.gmm_var99, color="#008C2D", lw=1.6, ls=":", label=f"GMM 99% VaR ({100*res.gmm_var99:.2f}%, err {res.var99_rel_error_pct:.2f}%)")
    ax1.axvline(-res.gaussian_var99, color="#D55E00", lw=1.3, ls="-.", label=f"Gaussian 99% VaR ({100*res.gaussian_var99:.2f}%)")

    ax1.set_yscale("log")
    ax1.set_ylim(2e-2, 45.0)
    ax1.set_xlim(-0.105, 0.085)
    ax1.set_xlabel("Daily Log-Return $y_t$")
    ax1.set_ylabel("Probability Density $p(y_t)$ [log scale]")
    ax1.set_title(f"(a) Out-of-Sample Tail Density (KL Reduction: -{res.kl_reduction_pct:.1f}%)")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92)

    # Right panel: Quantile-Quantile plot across percentiles
    probs = np.linspace(0.004, 0.996, 180)
    emp_q = np.quantile(res.test_y, probs)
    gmm_q = np.quantile(res.synthetic_y, probs)
    gauss_q = norm.ppf(probs, loc=mu_g, scale=std_g)

    q_lim = [min(emp_q.min(), gmm_q.min()) * 1.08, max(emp_q.max(), gmm_q.max()) * 1.08]
    ax2.plot(q_lim, q_lim, color="#222222", lw=1.3, ls="--", label="Exact $45^\\circ$ Reference")
    ax2.scatter(
        emp_q,
        gauss_q,
        s=15,
        alpha=0.65,
        color="#D55E00",
        marker="^",
        label=f"Single Gaussian (KS = {res.ks_stat_gaussian:.4f})",
    )
    ax2.scatter(
        emp_q,
        gmm_q,
        s=16,
        alpha=0.80,
        color="#8A0076",
        marker="o",
        label=f"3-State GMM Backfill (KS = {res.ks_stat_gmm:.4f})",
    )
    ax2.set_xlim(q_lim)
    ax2.set_ylim(q_lim)
    ax2.set_xlabel("Empirical Out-of-Sample Quantiles")
    ax2.set_ylabel("Model-Implied Quantiles")
    ax2.set_title(
        f"(b) Tail Q-Q Calibration & C++ Speedup ({res.cpp_speedup_factor:.1f}x Faster)"
    )
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92)

    fig.tight_layout()
    out_path = fig_dir / "fig1_gmm_qq_and_tail_var.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "kl_reduction_pct": res.kl_reduction_pct,
        "var99_rel_error_pct": res.var99_rel_error_pct,
        "cpp_speedup_factor": res.cpp_speedup_factor,
        "cpp_total_time_ms": res.cpp_total_time_ms,
        "python_total_time_ms": res.python_total_time_ms,
        "ks_gmm": res.ks_stat_gmm,
        "ks_gaussian": res.ks_stat_gaussian,
    }


def plot_fig2_mc_and_mixing(fig_dir: Path) -> dict:
    mk = run_markov_and_barrier_benchmark()
    bar = mk["barrier_mc"]
    mix = mk["mcmc_mixing"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.35))

    # Left panel: Monte Carlo Standard Error convergence
    ax1.loglog(
        bar.sample_sizes,
        bar.crude_se_curve,
        "o-",
        color="#D55E00",
        lw=1.8,
        ms=4.0,
        label=f"Crude Monte Carlo (SE = {bar.crude_se:.4f})",
    )
    ax1.loglog(
        bar.sample_sizes,
        bar.antithetic_se_curve,
        "s--",
        color="#4C72B0",
        lw=1.7,
        ms=4.0,
        label=f"Antithetic Variates (SE = {bar.antithetic_se:.4f})",
    )
    ax1.loglog(
        bar.sample_sizes,
        bar.controlled_se_curve,
        "^-",
        color="#008C2D",
        lw=2.1,
        ms=4.5,
        label=f"Antithetic + Control Variate (SE = {bar.controlled_se:.4f})",
    )
    ax1.set_xlabel("Number of Simulated Path Pairs $N$")
    ax1.set_ylabel("Monte Carlo Standard Error ($\\mathcal{O}(N^{-1/2})$)")
    ax1.set_title(
        f"(a) Barrier Option MC Convergence ({bar.variance_reduction_factor:.2f}x Var. Reduction, "
        f"Err {bar.relative_error_pct:.2f}%)"
    )
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92)

    # Right panel: Metropolis-Hastings TV distance vs Perron-Frobenius spectral bound
    ax2.semilogy(
        mix.steps,
        mix.perron_frobenius_bound,
        color="#8A0076",
        lw=2.0,
        ls="--",
        label=f"Perron-Frobenius Spectral Bound $\\propto |\\lambda_2|^t$ ($|\\lambda_2|={mix.second_eigenvalue_mod:.3f}$)",
    )
    ax2.semilogy(
        mix.steps,
        mix.exact_tv_distance,
        color="#1932E6",
        lw=2.0,
        label="Exact Matrix-Power $\\|P^t(x_0, \\cdot) - \\pi\\|_{\\mathrm{TV}}$",
    )
    ax2.semilogy(
        mix.steps,
        mix.empirical_mcmc_tv,
        "o",
        color="#008C2D",
        ms=3.8,
        alpha=0.8,
        label="Empirical Metropolis-Hastings MCMC (40k Chains)",
    )
    ax2.set_xlabel("Markov Chain Step $t$")
    ax2.set_ylabel("Total Variation Distance to Stationary $\\pi$ [log scale]")
    ax2.set_title(
        f"(b) MCMC Mixing & Spectral Gap ($\\gamma_\\star = 1 - |\\lambda_2| = {mix.spectral_gap:.3f}$)"
    )
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92)

    fig.tight_layout()
    out_path = fig_dir / "fig2_mc_variance_reduction_convergence.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "variance_reduction_factor": bar.variance_reduction_factor,
        "barrier_rel_error_pct": bar.relative_error_pct,
        "spectral_gap": mix.spectral_gap,
        "ruin_max_err": mk["gamblers_ruin"].max_abs_prob_error,
    }


def plot_fig3_fokker_planck_and_msd(fig_dir: Path) -> dict:
    sde_data = run_sde_fokker_planck_benchmark(n_paths=100_000, seed=202606)
    fp = sde_data["fokker_planck"]
    msd = sde_data["msd"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.35))

    # Left panel: Fokker-Planck PDE vs Euler-Maruyama SDE densities
    ax1.hist(
        fp.sde_mid_samples,
        bins=60,
        density=True,
        alpha=0.28,
        color="#D55E00",
        label="Euler-Maruyama SDE ($t = 0.50$, 100k Paths)",
    )
    ax1.hist(
        fp.sde_terminal_samples,
        bins=60,
        density=True,
        alpha=0.30,
        color="#4C72B0",
        label="Euler-Maruyama SDE ($t = 2.00$, 100k Paths)",
    )
    ax1.plot(
        fp.x_grid,
        fp.pde_density_transient,
        color="#D55E00",
        lw=2.1,
        label="Crank-Nicolson Fokker-Planck PDE ($t = 0.50$)",
    )
    ax1.plot(
        fp.x_grid,
        fp.pde_density_stationary,
        color="#1932E6",
        lw=2.2,
        label="Crank-Nicolson Fokker-Planck PDE ($t = 2.00$)",
    )
    ax1.plot(
        fp.x_grid,
        fp.analytical_stationary,
        color="#111111",
        lw=1.4,
        ls=":",
        label=f"Analytical OU Equilibrium ($L_1$ err = {fp.l1_density_error:.4f})",
    )
    ax1.set_xlim(-2.2, 2.6)
    ax1.set_xlabel("State Variable $x$")
    ax1.set_ylabel("Probability Density $p(x, t)$")
    ax1.set_title("(a) Fokker-Planck PDE vs. Ornstein-Uhlenbeck Langevin SDE")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92)

    # Right panel: Linear growth of Mean-Squared Displacement <X_t^2> = 2Dt
    ax2.plot(
        msd.time_grid,
        msd.theoretical_msd,
        color="#111111",
        lw=2.0,
        ls="--",
        label="Theoretical Diffusion Law $\\langle X_t^2 \\rangle = 2Dt$",
    )
    ax2.plot(
        msd.time_grid,
        msd.empirical_msd,
        color="#8A0076",
        lw=2.1,
        label=f"Empirical 100k-Path MSD (Error: {msd.slope_relative_error_pct:.2f}%)",
    )

    ax2_twin = ax2.twinx()
    for path_idx in range(10):
        ax2_twin.plot(
            msd.time_grid,
            msd.sample_trajectories[path_idx],
            color="#888888",
            alpha=0.18,
            lw=0.8,
        )
    ax2_twin.set_ylabel("Sample Brownian Paths $X_t$ (Faint)", color="#666666")
    ax2_twin.tick_params(axis="y", labelcolor="#666666")
    ax2_twin.spines["top"].set_visible(False)
    ax2_twin.grid(False)

    ax2.set_xlabel("Time $t$ (Years)")
    ax2.set_ylabel("Mean-Squared Displacement $\\langle X_t^2 \\rangle$")
    ax2.set_title(
        f"(b) Brownian Diffusion MSD Scaling (Slope Error {msd.slope_relative_error_pct:.2f}% < 0.4%)"
    )
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92)

    fig.tight_layout()
    out_path = fig_dir / "fig3_fokker_planck_density_and_msd.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "msd_error_pct": msd.slope_relative_error_pct,
        "fp_l1_error": fp.l1_density_error,
    }


def plot_fig4_credit_migration(fig_dir: Path) -> dict:
    model = CreditMigrationModel()
    res = run_credit_migration_benchmark()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.35))

    # Left panel: Continuous-time cumulative default probability curves
    fine_t = np.linspace(0.25, 10.0, 40)
    fine_curves = model.cumulative_default_curve(fine_t)
    palette = {
        "AAA": "#1b9e77",
        "AA": "#2b5c8f",
        "A": "#4C72B0",
        "BBB": "#7570b3",
        "BB": "#d95f02",
        "B": "#e7298a",
        "CCC": "#a61c1c",
    }
    for rating in ["AA", "A", "BBB", "BB", "B", "CCC"]:
        ax1.plot(
            fine_t,
            100.0 * fine_curves[rating],
            lw=2.0,
            color=palette[rating],
            label=f"Rating {rating}",
        )
    ax1.set_yscale("log")
    ax1.set_xlabel("Horizon $T$ (Years)")
    ax1.set_ylabel("Cumulative Default Probability $P_D(T)$ [%, log scale]")
    ax1.set_title("(a) Sovereign Credit Absorption Probabilities $[\\exp(QT)]_{i,D}$")
    ax1.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.92)

    # Right panel: Model vs Empirical CDS Par Spreads (bps)
    for rating in ["AA", "A", "BBB", "BB", "B"]:
        ax2.plot(
            res.maturities,
            res.model_cds_spreads_bps[rating],
            "-",
            lw=1.9,
            color=palette[rating],
            label=f"{rating} Generator Model",
        )
        ax2.scatter(
            res.maturities,
            res.empirical_cds_spreads_bps[rating],
            s=28,
            color=palette[rating],
            marker="o",
            edgecolors="#111111",
            linewidths=0.6,
            zorder=4,
        )

    ax2.scatter([], [], s=28, color="#555555", marker="o", edgecolors="#111111", label="Empirical Market CDS Quotes")
    ax2.set_yscale("log")
    ax2.set_xlabel("Maturity $T$ (Years)")
    ax2.set_ylabel("Par CDS Spread $s_{\\mathrm{par}}(T)$ [bps, log scale]")
    ax2.set_title(
        f"(b) CDS Term-Structure Fit (Max Error: {res.max_abs_spread_error_bps:.1f} bps < 15 bps)"
    )
    ax2.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.92)

    fig.tight_layout()
    out_path = fig_dir / "fig4_credit_migration_term_structure.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "max_cds_error_bps": res.max_abs_spread_error_bps,
        "mean_cds_error_bps": res.mean_abs_spread_error_bps,
    }


def main() -> None:
    configure_plot_style()
    fig_dir = REPO_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    m1 = plot_fig1_gmm_tail(fig_dir)
    m2 = plot_fig2_mc_and_mixing(fig_dir)
    m3 = plot_fig3_fokker_planck_and_msd(fig_dir)
    m4 = plot_fig4_credit_migration(fig_dir)

    print("Generated all 4 figures in figures/:")
    print(f"  - fig1_gmm_qq_and_tail_var.png           (KL Reduction: -{m1['kl_reduction_pct']:.1f}%, VaR99 Err: {m1['var99_rel_error_pct']:.2f}%, C++ Speedup: {m1['cpp_speedup_factor']:.2f}x)")
    print(f"  - fig2_mc_variance_reduction_convergence.png (Variance Reduction: {m2['variance_reduction_factor']:.2f}x, BS Rel Err: {m2['barrier_rel_error_pct']:.3f}%)")
    print(f"  - fig3_fokker_planck_density_and_msd.png (MSD Slope Err: {m3['msd_error_pct']:.3f}%, FP L1 Err: {m3['fp_l1_error']:.5f})")
    print(f"  - fig4_credit_migration_term_structure.png (Max CDS Spread Err: {m4['max_cds_error_bps']:.2f} bps, Mean: {m4['mean_cds_error_bps']:.2f} bps)")


if __name__ == "__main__":
    main()

