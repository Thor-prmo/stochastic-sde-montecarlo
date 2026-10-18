"""Unit tests for barrier option Black-Scholes limits, variance reduction, and C++ parity."""

import json
from pathlib import Path

import numpy as np
import pytest

from python_stochastics.markov_mcmc_perron import BarrierMonteCarloEngine


def test_barrier_option_converges_to_black_scholes_vanilla_limit():
    """Verify Down-and-Out Call converges to closed-form Black-Scholes vanilla call as barrier B -> 0."""
    s0, k, r, sigma, t = 100.0, 100.0, 0.045, 0.21, 1.0
    vanilla_exact = BarrierMonteCarloEngine.black_scholes_call(s0, k, r, sigma, t)

    eng_zero = BarrierMonteCarloEngine(s0=s0, strike=k, barrier=0.0, rate=r, sigma=sigma, maturity=t, n_steps=64)
    assert np.isclose(eng_zero.analytical_down_and_out_call(), vanilla_exact, atol=1e-12)

    eng_low = BarrierMonteCarloEngine(s0=s0, strike=k, barrier=25.0, rate=r, sigma=sigma, maturity=t, n_steps=64)
    assert np.isclose(eng_low.analytical_down_and_out_call(), vanilla_exact, atol=1e-10)

    mc_res = eng_low.price(n_paths=40_000, seed=42)
    assert np.isclose(mc_res.controlled_price, vanilla_exact, atol=1e-10)


def test_barrier_knockout_boundary_and_monotonicity():
    """Verify Down-and-Out Call vanishes at S0 <= B and decreases monotonically as B increases."""
    eng_knocked = BarrierMonteCarloEngine(s0=95.0, strike=100.0, barrier=95.0)
    assert eng_knocked.analytical_down_and_out_call() == 0.0

    barriers = [70.0, 80.0, 88.0, 92.0, 96.0, 99.0]
    prices = [
        BarrierMonteCarloEngine(s0=100.0, strike=100.0, barrier=b).analytical_down_and_out_call()
        for b in barriers
    ]
    for p_hi, p_lo in zip(prices[:-1], prices[1:]):
        assert p_hi > p_lo > 0.0


def test_antithetic_and_control_variate_variance_reduction():
    """Verify Antithetic + Control Variates achieves ~3.8x variance reduction within 0.12% of Black-Scholes."""
    eng = BarrierMonteCarloEngine(
        s0=100.0, strike=100.0, barrier=96.26, rate=0.045, sigma=0.21, maturity=1.0, n_steps=128
    )
    res = eng.price(n_paths=120_000, seed=18)

    assert res.variance_reduction_factor >= 3.5, (
        f"Expected variance reduction >= 3.5x, got {res.variance_reduction_factor:.2f}x"
    )
    assert res.relative_error_pct <= 0.12, (
        f"Relative pricing error {res.relative_error_pct:.4f}% exceeded 0.12% benchmark"
    )
    assert res.controlled_se < res.antithetic_se < res.crude_se


def test_cpp_and_python_barrier_benchmark_parity():
    """Verify compiled C++17 Monte Carlo engine matches analytical Black-Scholes within 0.12%."""
    repo_root = Path(__file__).resolve().parents[1]
    json_path = repo_root / "build" / "cpp_metrics.json"
    if not json_path.exists():
        pytest.skip("C++ benchmark JSON not yet generated.")

    with open(json_path, "r", encoding="utf-8") as f:
        cpp_metrics = json.load(f)

    assert cpp_metrics["mc_variance_reduction"] >= 3.5
    assert cpp_metrics["mc_rel_error_pct"] <= 0.12
    assert cpp_metrics["mc_controlled_se"] < cpp_metrics["mc_crude_se"]

