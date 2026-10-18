#ifndef STOCH_BARRIER_MC_HPP
#define STOCH_BARRIER_MC_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stoch {

struct BarrierOptionParams {
    double s0;          // Initial spot price
    double strike;      // Strike price K
    double barrier;     // Down-and-out lower barrier B < s0
    double rate;        // Risk-free rate r
    double sigma;       // Volatility sigma
    double maturity;    // Time to maturity T (years)
    int n_steps;        // Number of discrete monitoring steps
};

struct BarrierMCResult {
    double analytical_bs_vanilla;
    double analytical_bs_barrier;
    double crude_mc_price;
    double crude_mc_std_error;
    double antithetic_mc_price;
    double antithetic_mc_std_error;
    double controlled_mc_price;
    double controlled_mc_std_error;
    double variance_reduction_factor;
    double relative_error_pct;
    double optimal_control_beta;
    double runtime_ms;
};

struct BellmanMDPResult {
    double optimal_expected_value;
    std::vector<int> optimal_policy_t0;
    double bellman_residual;
};

class BarrierOptionEngine {
public:
    explicit BarrierOptionEngine(const BarrierOptionParams& params);

    // Closed-form Black-Scholes vanilla call price
    static double black_scholes_call(double s0, double k, double r, double sigma, double t);

    // Closed-form Merton / Reiner-Rubinstein continuous Down-and-Out Call price
    static double analytical_down_and_out_call(const BarrierOptionParams& p);

    // Run Crude, Antithetic, and Antithetic + Control Variate Monte Carlo pricing
    BarrierMCResult run_monte_carlo(std::size_t n_paths, uint64_t seed = 42) const;

    // Solve finite-horizon Bellman MDP for dynamic barrier hedging / execution
    BellmanMDPResult solve_bellman_execution_mdp(int n_inventory_levels = 11,
                                                 int n_price_states = 21,
                                                 int horizon_steps = 20) const;

private:
    BarrierOptionParams params_;
};

} // namespace stoch

#endif // STOCH_BARRIER_MC_HPP

