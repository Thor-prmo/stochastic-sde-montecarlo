#include "barrier_mc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace stoch {

namespace {
constexpr double kInvSqrt2 = 0.7071067811865475244008443621048;

inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * kInvSqrt2);
}
} // namespace

BarrierOptionEngine::BarrierOptionEngine(const BarrierOptionParams& params) : params_(params) {
    if (params_.s0 <= 0.0 || params_.strike <= 0.0 || params_.sigma <= 0.0 || params_.maturity <= 0.0) {
        throw std::invalid_argument("Option spot, strike, volatility, and maturity must be positive.");
    }
}

double BarrierOptionEngine::black_scholes_call(double s0, double k, double r, double sigma, double t) {
    if (t <= 0.0) {
        return std::max(s0 - k, 0.0);
    }
    const double vol_sqrt_t = sigma * std::sqrt(t);
    const double d1 = (std::log(s0 / k) + (r + 0.5 * sigma * sigma) * t) / vol_sqrt_t;
    const double d2 = d1 - vol_sqrt_t;
    return s0 * norm_cdf(d1) - k * std::exp(-r * t) * norm_cdf(d2);
}

double BarrierOptionEngine::analytical_down_and_out_call(const BarrierOptionParams& p) {
    if (p.barrier <= 1e-8) {
        return black_scholes_call(p.s0, p.strike, p.rate, p.sigma, p.maturity);
    }
    if (p.s0 <= p.barrier) {
        return 0.0;
    }

    const double s0 = p.s0;
    const double k = p.strike;
    const double b = p.barrier;
    const double r = p.rate;
    const double sigma = p.sigma;
    const double t = p.maturity;
    const double vol_sqrt_t = sigma * std::sqrt(t);
    const double lambda = (r + 0.5 * sigma * sigma) / (sigma * sigma);

    if (b <= k) {
        const double vanilla = black_scholes_call(s0, k, r, sigma, t);
        const double y = std::log((b * b) / (s0 * k)) / vol_sqrt_t + lambda * vol_sqrt_t;
        const double power_term = std::pow(b / s0, 2.0 * lambda);
        const double power_term_2 = std::pow(b / s0, 2.0 * lambda - 2.0);
        const double c_di = s0 * power_term * norm_cdf(y)
                          - k * std::exp(-r * t) * power_term_2 * norm_cdf(y - vol_sqrt_t);
        return std::max(vanilla - c_di, 0.0);
    } else {
        const double x1 = std::log(s0 / b) / vol_sqrt_t + lambda * vol_sqrt_t;
        const double y1 = std::log(b / s0) / vol_sqrt_t + lambda * vol_sqrt_t;
        const double power_term = std::pow(b / s0, 2.0 * lambda);
        const double power_term_2 = std::pow(b / s0, 2.0 * lambda - 2.0);
        const double c_do = s0 * norm_cdf(x1) - k * std::exp(-r * t) * norm_cdf(x1 - vol_sqrt_t)
                          - s0 * power_term * norm_cdf(y1)
                          + k * std::exp(-r * t) * power_term_2 * norm_cdf(y1 - vol_sqrt_t);
        return std::max(c_do, 0.0);
    }
}

BarrierMCResult BarrierOptionEngine::run_monte_carlo(std::size_t n_paths, uint64_t seed) const {
    const auto t_start = std::chrono::steady_clock::now();

    const std::size_t n_pairs = std::max<std::size_t>(1, n_paths / 2);
    const std::size_t total_paths = n_pairs * 2;
    const double dt = params_.maturity / static_cast<double>(params_.n_steps);
    const double drift = (params_.rate - 0.5 * params_.sigma * params_.sigma) * dt;
    const double vol_dt = params_.sigma * std::sqrt(dt);
    const double var_dt = params_.sigma * params_.sigma * dt;
    const double disc = std::exp(-params_.rate * params_.maturity);
    const double b = params_.barrier;

    const double bs_vanilla = black_scholes_call(params_.s0, params_.strike, params_.rate,
                                                 params_.sigma, params_.maturity);
    const double bs_barrier = analytical_down_and_out_call(params_);

    std::vector<double> crude_payoffs(total_paths, 0.0);
    std::vector<double> av_barrier_payoffs(n_pairs, 0.0);
    std::vector<double> av_control_payoffs(n_pairs, 0.0);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    auto simulate_single_path = [&](const std::vector<double>& z_steps,
                                    double sign,
                                    double& out_barrier_payoff,
                                    double& out_vanilla_payoff) {
        double s_prev = params_.s0;
        double survival_prob = (b > 0.0 && s_prev <= b) ? 0.0 : 1.0;

        for (int m = 0; m < params_.n_steps; ++m) {
            const double z = sign * z_steps[m];
            const double s_curr = s_prev * std::exp(drift + vol_dt * z);

            if (b > 0.0) {
                if (s_curr <= b || s_prev <= b) {
                    survival_prob = 0.0;
                } else if (survival_prob > 0.0) {
                    // Brownian bridge survival probability within (t_m, t_{m+1})
                    const double log_prev = std::log(s_prev / b);
                    const double log_curr = std::log(s_curr / b);
                    const double p_cross = std::exp(-2.0 * log_prev * log_curr / var_dt);
                    survival_prob *= std::max(0.0, 1.0 - p_cross);
                }
            }
            s_prev = s_curr;
        }

        const double terminal_call = disc * std::max(s_prev - params_.strike, 0.0);
        out_vanilla_payoff = terminal_call;
        out_barrier_payoff = survival_prob * terminal_call;
    };

    std::vector<double> z_vec(params_.n_steps);
    for (std::size_t i = 0; i < n_pairs; ++i) {
        for (int m = 0; m < params_.n_steps; ++m) {
            z_vec[m] = norm(rng);
        }

        double bar_pos = 0.0, van_pos = 0.0;
        double bar_neg = 0.0, van_neg = 0.0;
        simulate_single_path(z_vec, +1.0, bar_pos, van_pos);
        simulate_single_path(z_vec, -1.0, bar_neg, van_neg);

        crude_payoffs[2 * i] = bar_pos;
        crude_payoffs[2 * i + 1] = bar_neg;

        av_barrier_payoffs[i] = 0.5 * (bar_pos + bar_neg);
        av_control_payoffs[i] = 0.5 * (van_pos + van_neg);
    }

    // Crude Monte Carlo statistics (over first n_pairs independent paths for fair per-draw variance comparison)
    double crude_mean = 0.0;
    for (std::size_t i = 0; i < n_pairs; ++i) {
        crude_mean += crude_payoffs[2 * i];
    }
    crude_mean /= static_cast<double>(n_pairs);

    double crude_var = 0.0;
    for (std::size_t i = 0; i < n_pairs; ++i) {
        const double d = crude_payoffs[2 * i] - crude_mean;
        crude_var += d * d;
    }
    crude_var /= static_cast<double>(std::max<std::size_t>(1, n_pairs - 1));
    const double crude_se = std::sqrt(crude_var / static_cast<double>(n_pairs));

    // Antithetic Variates statistics
    double av_mean = 0.0;
    double ctrl_mean = 0.0;
    for (std::size_t i = 0; i < n_pairs; ++i) {
        av_mean += av_barrier_payoffs[i];
        ctrl_mean += av_control_payoffs[i];
    }
    av_mean /= static_cast<double>(n_pairs);
    ctrl_mean /= static_cast<double>(n_pairs);

    double av_var = 0.0;
    double ctrl_var = 0.0;
    double cov_bc = 0.0;
    for (std::size_t i = 0; i < n_pairs; ++i) {
        const double db = av_barrier_payoffs[i] - av_mean;
        const double dc = av_control_payoffs[i] - ctrl_mean;
        av_var += db * db;
        ctrl_var += dc * dc;
        cov_bc += db * dc;
    }
    const double denom = static_cast<double>(std::max<std::size_t>(1, n_pairs - 1));
    av_var /= denom;
    ctrl_var /= denom;
    cov_bc /= denom;

    const double av_se = std::sqrt(av_var / static_cast<double>(n_pairs));
    const double beta_opt = (ctrl_var > 1e-14) ? (cov_bc / ctrl_var) : 0.0;

    // Antithetic + Control Variate estimator
    double cv_mean = 0.0;
    std::vector<double> cv_samples(n_pairs, 0.0);
    for (std::size_t i = 0; i < n_pairs; ++i) {
        cv_samples[i] = av_barrier_payoffs[i] - beta_opt * (av_control_payoffs[i] - bs_vanilla);
        cv_mean += cv_samples[i];
    }
    cv_mean /= static_cast<double>(n_pairs);

    double cv_var = 0.0;
    for (std::size_t i = 0; i < n_pairs; ++i) {
        const double d = cv_samples[i] - cv_mean;
        cv_var += d * d;
    }
    cv_var /= denom;
    const double cv_se = std::sqrt(cv_var / static_cast<double>(n_pairs));

    const double vr_factor = (cv_var > 1e-14) ? (crude_var / cv_var) : 1.0;
    const double rel_err_pct = (bs_barrier > 1e-12)
        ? (100.0 * std::abs(cv_mean - bs_barrier) / bs_barrier)
        : 0.0;

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return BarrierMCResult{
        bs_vanilla,
        bs_barrier,
        crude_mean,
        crude_se,
        av_mean,
        av_se,
        cv_mean,
        cv_se,
        vr_factor,
        rel_err_pct,
        beta_opt,
        elapsed_ms
    };
}

BellmanMDPResult BarrierOptionEngine::solve_bellman_execution_mdp(int n_inventory_levels,
                                                                  int n_price_states,
                                                                  int horizon_steps) const {
    const double dt = params_.maturity / static_cast<double>(horizon_steps);
    const double gamma = std::exp(-params_.rate * dt);
    const double p_min = params_.barrier;
    const double p_max = params_.s0 * 1.45;
    const double dp = (p_max - p_min) / static_cast<double>(n_price_states - 1);

    std::vector<double> v_next(n_inventory_levels * n_price_states, 0.0);
    std::vector<double> v_curr(n_inventory_levels * n_price_states, 0.0);
    std::vector<int> policy_t0(n_inventory_levels * n_price_states, 0);

    // Terminal condition at maturity T: liquidate remaining hedge inventory against option payoff
    for (int q = 0; q < n_inventory_levels; ++q) {
        const double inv = static_cast<double>(q) / static_cast<double>(n_inventory_levels - 1);
        for (int s = 0; s < n_price_states; ++s) {
            const double spot = (s == 0) ? p_min : (p_min + s * dp);
            const double payoff = (s == 0) ? 0.0 : std::max(spot - params_.strike, 0.0);
            const double tracking_penalty = 0.5 * (inv * spot - payoff) * (inv * spot - payoff) * 0.01;
            v_next[q * n_price_states + s] = -tracking_penalty;
        }
    }

    double last_max_diff = 0.0;
    for (int t = horizon_steps - 1; t >= 0; --t) {
        last_max_diff = 0.0;
        for (int q = 0; q < n_inventory_levels; ++q) {
            for (int s = 0; s < n_price_states; ++s) {
                if (s == 0) {
                    // Knocked-out absorbing barrier state
                    v_curr[q * n_price_states + s] = -0.02 * static_cast<double>(q * q);
                    if (t == 0) policy_t0[q * n_price_states + s] = 0;
                    continue;
                }

                const double spot = p_min + s * dp;
                double best_val = -1e18;
                int best_q_next = q;

                for (int q_next = std::max(0, q - 2); q_next <= std::min(n_inventory_levels - 1, q + 2); ++q_next) {
                    const double trade = static_cast<double>(q_next - q) / static_cast<double>(n_inventory_levels - 1);
                    const double exec_cost = 0.0012 * spot * std::abs(trade) + 0.004 * spot * trade * trade;
                    const double s_up = std::min(n_price_states - 1, s + 1);
                    const double s_dn = std::max(0, s - 1);
                    const double cont_val = 0.28 * v_next[q_next * n_price_states + s_up]
                                          + 0.44 * v_next[q_next * n_price_states + s]
                                          + 0.28 * v_next[q_next * n_price_states + s_dn];
                    const double candidate = -exec_cost + gamma * cont_val;
                    if (candidate > best_val) {
                        best_val = candidate;
                        best_q_next = q_next;
                    }
                }

                const double diff = std::abs(best_val - v_next[q * n_price_states + s]);
                if (diff > last_max_diff) last_max_diff = diff;
                v_curr[q * n_price_states + s] = best_val;
                if (t == 0) {
                    policy_t0[q * n_price_states + s] = best_q_next - q;
                }
            }
        }
        v_next = v_curr;
    }

    const int mid_q = n_inventory_levels / 2;
    const int s0_idx = std::clamp(static_cast<int>(std::round((params_.s0 - p_min) / dp)), 1, n_price_states - 1);
    return BellmanMDPResult{
        v_curr[mid_q * n_price_states + s0_idx],
        policy_t0,
        last_max_diff
    };
}

} // namespace stoch

