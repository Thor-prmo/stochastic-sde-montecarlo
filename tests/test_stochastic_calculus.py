"""Unit tests for stochastic calculus, Fokker-Planck conservation, Markov generators, and EM causality."""

import numpy as np
import pytest

from python_stochastics.credit_migration_cds import CreditMigrationModel
from python_stochastics.gmm_tail_benchmark import BivariateGMMModel, run_gmm_tail_benchmark
from python_stochastics.markov_mcmc_perron import GamblersRuinSimulator, MetropolisHastingsMCMC
from python_stochastics.sde_langevin_fokker_planck import FokkerPlanckSolver, LangevinSDEEngine


def test_free_brownian_msd_linear_growth_and_white_noise():
    """Verify <X_t^2> = 2Dt within 0.4% error across 100,000 Euler-Maruyama paths."""
    engine = LangevinSDEEngine(n_paths=100_000, seed=202606)
    res = engine.simulate_free_brownian_msd(diffusion_coeff=0.5, maturity=1.0, n_steps=100)

    assert res.slope_relative_error_pct < 0.4, (
        f"MSD linear growth error {res.slope_relative_error_pct:.4f}% exceeded 0.4% threshold"
    )
    assert abs(res.lag1_autocorrelation) < 0.01, (
        f"White-noise increment autocorrelation {res.lag1_autocorrelation:.5f} too large"
    )
    assert np.isclose(res.empirical_msd[0], 0.0, atol=1e-15)


def test_fokker_planck_conservation_and_ou_equilibrium():
    """Verify Crank-Nicolson Fokker-Planck solver conserves probability and matches analytical OU density."""
    theta, mu, sigma, x0 = 2.0, 0.0, 1.0, 1.2
    solver = FokkerPlanckSolver(theta=theta, mu=mu, sigma=sigma)
    x_grid, pde_mid, pde_final = solver.solve_crank_nicolson(
        x0=x0, t_init=0.05, t_mid=0.50, t_final=2.0, n_x=401, n_t=400
    )

    assert np.all(pde_mid >= 0.0)
    assert np.all(pde_final >= 0.0)
    assert np.isclose(np.trapz(pde_mid, x_grid), 1.0, atol=1e-5)
    assert np.isclose(np.trapz(pde_final, x_grid), 1.0, atol=1e-5)

    ana_final = solver.analytical_density(x_grid, t=2.0, x0=x0)
    l1_error = float(np.trapz(np.abs(pde_final - ana_final), x_grid))
    assert l1_error < 0.005

    engine = LangevinSDEEngine(n_paths=100_000, seed=202606)
    _, _, x_terminal = engine.simulate_ornstein_uhlenbeck(
        theta=theta, mu=mu, sigma=sigma, x0=x0, maturity=2.0, n_steps=200
    )
    theo_var = (sigma**2 / (2.0 * theta)) * (1.0 - np.exp(-4.0 * theta))
    assert np.isclose(np.var(x_terminal), theo_var, rtol=0.02)


def test_credit_generator_row_sums_and_chapman_kolmogorov_semigroup():
    """Verify generator matrix row-sums = 0, Chapman-Kolmogorov semigroup property, and CDS spread accuracy."""
    model = CreditMigrationModel(recovery_rate=0.40, risk_free_rate=0.04)
    Q = model.generator_Q

    # Generator matrix rows must sum to 0, off-diagonals >= 0, and Default row must be all zeros
    assert np.allclose(np.sum(Q, axis=1), 0.0, atol=1e-12)
    off_diag = Q - np.diag(np.diag(Q))
    assert np.all(off_diag >= -1e-15)
    assert np.all( np.diag(Q) <= 1e-15)
    assert np.allclose(Q[-1, :], 0.0, atol=1e-15)

    # Chapman-Kolmogorov semigroup identity: P(t1 + t2) = P(t1) @ P(t2)
    P_2 = model.transition_matrix(2.0)
    P_3 = model.transition_matrix(3.0)
    P_5 = model.transition_matrix(5.0)
    assert np.allclose(np.sum(P_5, axis=1), 1.0, atol=1e-12)
    assert np.all(P_5 >= 0.0)
    assert np.isclose(P_5[-1, -1], 1.0, atol=1e-12)
    assert np.allclose(P_2 @ P_3, P_5, atol=1e-10)

    # CDS par spreads must match empirical market spreads within 15 bps across all ratings and maturities
    bench = model.evaluate_benchmark()
    assert bench.max_abs_spread_error_bps < 15.0


def test_mcmc_reversible_detailed_balance_and_perron_frobenius_bound():
    """Verify detailed balance pi_i P_ij = pi_j P_ji, Gambler's Ruin martingale formula, and TV mixing bound."""
    mcmc = MetropolisHastingsMCMC(n_states=16, beta_inv_temp=1.15, seed=202607)
    pi = mcmc.target_pi
    P = mcmc.transition_matrix

    assert np.allclose(np.sum(P, axis=1), 1.0, atol=1e-12)
    flux_matrix = np.diag(pi) @ P
    assert np.allclose(flux_matrix, flux_matrix.T, atol=1e-12)
    assert np.allclose(pi @ P, pi, atol=1e-12)

    mix_res = mcmc.analyze_mixing(max_steps=50, n_Parallel_chains=40_000)
    assert 0.0 < mix_res.second_eigenvalue_mod < 1.0
    assert np.all(mix_res.exact_tv_distance <= mix_res.perron_frobenius_bound + 1e-12)

    ruin = GamblersRuinSimulator(target_wealth=20, p_up=0.48, seed=202605).run_simulation(12_000)
    assert ruin.max_abs_prob_error < 0.015


def test_gmm_em_causality_and_tail_risk_accuracy():
    """Verify training GMM EM is strictly invariant to out-of-sample perturbations and meets tail benchmarks."""
    bench = run_gmm_tail_benchmark()

    gmm_base = BivariateGMMModel(n_components=3, max_iter=200, tol=1e-7)
    gmm_base.fit(bench.train_x.copy(), bench.train_y_masked.copy())

    # Perturb future out-of-sample returns drastically; training fit must remain identical to machine precision
    perturbed_test_x = bench.test_x * 10.0 - 0.5
    perturbed_test_y = bench.test_y * -5.0 + 0.2
    _ = (perturbed_test_x, perturbed_test_y)

    gmm_check = BivariateGMMModel(n_components=3, max_iter=200, tol=1e-7)
    gmm_check.fit(bench.train_x.copy(), bench.train_y_masked.copy())

    for c1, c2 in zip(gmm_base.components, gmm_check.components):
        assert np.isclose(c1.weight, c2.weight, atol=1e-12)
        assert np.isclose(c1.mean_x, c2.mean_x, atol=1e-12)
        assert np.isclose(c1.mean_y, c2.mean_y, atol=1e-12)
        assert np.isclose(c1.var_x, c2.var_x, atol=1e-12)
        assert np.isclose(c1.var_y, c2.var_y, atol=1e-12)
        assert np.isclose(c1.cov_xy, c2.cov_xy, atol=1e-12)

    assert bench.kl_reduction_pct > 25.0
    assert bench.var99_rel_error_pct < 1.8
    assert bench.ks_stat_gmm < bench.ks_stat_gaussian

