#include "barrier_mc.hpp"
#include "gmm_em.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    std::string export_csv_path;
    std::string export_json_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--export-data" && i + 1 < argc) {
            export_csv_path = argv[++i];
        } else if (arg == "--export-json" && i + 1 < argc) {
            export_json_path = argv[++i];
        }
    }

    // 1. Generate joint AAPL + short-history IPO return series & run Observed-Data GMM EM
    const std::size_t n_train = 1800;
    const std::size_t n_test = 750;
    const std::size_t n_ipo_obs = 420;
    const uint64_t data_seed = 20260585ULL;

    std::vector<double> train_x, train_y_masked, test_x, test_y;
    stoch::generate_joint_market_dataset(n_train, n_test, n_ipo_obs, data_seed,
                                         train_x, train_y_masked, test_x, test_y);

    if (!export_csv_path.empty()) {
        std::ofstream ofs(export_csv_path);
        ofs << "split,day,aapl_return,ipo_return\n";
        ofs << std::fixed << std::setprecision(8);
        for (std::size_t t = 0; t < n_train; ++t) {
            ofs << "train," << t << "," << train_x[t] << ",";
            if (std::isnan(train_y_masked[t])) {
                ofs << "NaN\n";
            } else {
                ofs << train_y_masked[t] << "\n";
            }
        }
        for (std::size_t t = 0; t < n_test; ++t) {
            ofs << "test," << t << "," << test_x[t] << "," << test_y[t] << "\n";
        }
    }

    stoch::BivariateGMM gmm(3, 200, 1e-7, 1e-6);
    const auto fit_res = gmm.fit_em(train_x, train_y_masked);

    std::vector<double> train_y_obs;
    for (double yv : train_y_masked) {
        if (!std::isnan(yv)) train_y_obs.push_back(yv);
    }

    const auto tail_res = stoch::evaluate_tail_benchmark(gmm, train_y_obs, test_y, 500000, 20260705ULL);

    // 2. Run Variance-Reduced Barrier Option Monte Carlo & Bellman MDP Execution
    stoch::BarrierOptionParams opt_params{
        100.0,  // S0
        100.0,  // K
        96.26,   // Down-and-out barrier B
        0.045,  // Risk-free rate r
        0.21,   // Volatility sigma
        1.0,    // Maturity T = 1.0 yr
        128      // Monitoring steps
    };
    stoch::BarrierOptionEngine barrier_engine(opt_params);
    const auto mc_res = barrier_engine.run_monte_carlo(120000, 202ULL);
    const auto mdp_res = barrier_engine.solve_bellman_execution_mdp(11, 21, 20);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "======================================================================\n";
    std::cout << " C++17 Stochastic SDE, Monte Carlo & Generative Market Benchmark\n";
    std::cout << "======================================================================\n";
    std::cout << "[1] Bivariate GMM EM Calibration (AAPL + Short-History IPO Backfill)\n";
    std::cout << "    EM Iterations / Log-Likelihood : " << fit_res.iterations
              << " / " << fit_res.final_log_likelihood << "\n";
    std::cout << "    C++ EM Fit + 500k Path Sim Time: "
              << (fit_res.fit_time_ms + tail_res.sim_time_ms) << " ms\n";
    std::cout << "    Out-of-Sample KL (Gaussian)    : " << tail_res.kl_divergence_gaussian << "\n";
    std::cout << "    Out-of-Sample KL (3-State GMM) : " << tail_res.kl_divergence_gmm
              << " (-" << std::setprecision(1) << tail_res.kl_reduction_pct << "%)\n";
    std::cout << std::setprecision(4);
    std::cout << "    Empirical vs GMM 99% Tail VaR  : " << (100.0 * tail_res.empirical_var99) << "% vs "
              << (100.0 * tail_res.gmm_var99) << "% (Rel. Err: "
              << std::setprecision(2) << tail_res.var99_rel_error_pct << "%)\n";
    std::cout << "    KS Statistic (GMM vs Gaussian) : " << std::setprecision(4)
              << tail_res.ks_statistic_gmm << " vs " << tail_res.ks_statistic_gaussian << "\n\n";

    std::cout << "[2] Barrier Option Monte Carlo (Antithetic + Control Variates)\n";
    std::cout << "    Analytical Black-Scholes Price : " << mc_res.analytical_bs_barrier << "\n";
    std::cout << "    Crude Monte Carlo Price (SE)   : " << mc_res.crude_mc_price
              << " (+/- " << mc_res.crude_mc_std_error << ")\n";
    std::cout << "    AV + Control Variate Price (SE): " << mc_res.controlled_mc_price
              << " (+/- " << mc_res.controlled_mc_std_error << ")\n";
    std::cout << "    Variance Reduction Factor      : " << std::setprecision(2)
              << mc_res.variance_reduction_factor << "x\n";
    std::cout << "    Relative Pricing Error vs BS   : " << std::setprecision(3)
              << mc_res.relative_error_pct << "%\n";
    std::cout << "    Bellman MDP Optimal Value (t=0): " << std::setprecision(4)
              << mdp_res.optimal_expected_value << " (Residual: " << mdp_res.bellman_residual << ")\n";
    std::cout << "======================================================================\n";

    if (!export_json_path.empty()) {
        std::ofstream jfs(export_json_path);
        jfs << std::fixed << std::setprecision(6);
        jfs << "{\n"
            << "  \"gmm_em_iterations\": " << fit_res.iterations << ",\n"
            << "  \"gmm_log_likelihood\": " << fit_res.final_log_likelihood << ",\n"
            << "  \"cpp_fit_time_ms\": " << fit_res.fit_time_ms << ",\n"
            << "  \"cpp_sim_time_ms\": " << tail_res.sim_time_ms << ",\n"
            << "  \"cpp_total_time_ms\": " << (fit_res.fit_time_ms + tail_res.sim_time_ms) << ",\n"
            << "  \"kl_gaussian\": " << tail_res.kl_divergence_gaussian << ",\n"
            << "  \"kl_gmm\": " << tail_res.kl_divergence_gmm << ",\n"
            << "  \"kl_reduction_pct\": " << tail_res.kl_reduction_pct << ",\n"
            << "  \"empirical_var99\": " << tail_res.empirical_var99 << ",\n"
            << "  \"gmm_var99\": " << tail_res.gmm_var99 << ",\n"
            << "  \"gaussian_var99\": " << tail_res.gaussian_var99 << ",\n"
            << "  \"var99_rel_error_pct\": " << tail_res.var99_rel_error_pct << ",\n"
            << "  \"ks_gmm\": " << tail_res.ks_statistic_gmm << ",\n"
            << "  \"ks_gaussian\": " << tail_res.ks_statistic_gaussian << ",\n"
            << "  \"bs_barrier_analytical\": " << mc_res.analytical_bs_barrier << ",\n"
            << "  \"mc_crude_price\": " << mc_res.crude_mc_price << ",\n"
            << "  \"mc_crude_se\": " << mc_res.crude_mc_std_error << ",\n"
            << "  \"mc_controlled_price\": " << mc_res.controlled_mc_price << ",\n"
            << "  \"mc_controlled_se\": " << mc_res.controlled_mc_std_error << ",\n"
            << "  \"mc_variance_reduction\": " << mc_res.variance_reduction_factor << ",\n"
            << "  \"mc_rel_error_pct\": " << mc_res.relative_error_pct << ",\n"
            << "  \"mc_runtime_ms\": " << mc_res.runtime_ms << ",\n"
            << "  \"mdp_optimal_value\": " << mdp_res.optimal_expected_value << "\n"
            << "}\n";
    }

    return 0;
}
