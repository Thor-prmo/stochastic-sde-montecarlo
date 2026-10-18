#ifndef STOCH_GMM_EM_HPP
#define STOCH_GMM_EM_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stoch {

struct BivariateComponent {
    double weight;
    double mean_x;
    double mean_y;
    double var_x;
    double cov_xy;
    double var_y;
};

struct GMMFitResult {
    std::vector<BivariateComponent> components;
    double final_log_likelihood;
    int iterations;
    bool converged;
    double fit_time_ms;
};

struct TailMetrics {
    double kl_divergence_gmm;
    double kl_divergence_gaussian;
    double kl_reduction_pct;
    double empirical_var99;
    double gmm_var99;
    double gaussian_var99;
    double var99_rel_error_pct;
    double ks_statistic_gmm;
    double ks_statistic_gaussian;
    double sim_time_ms;
};

class BivariateGMM {
public:
    explicit BivariateGMM(int n_components = 3, int max_iter = 200, double tol = 1e-7, double reg_covar = 1e-6);

    // Calibrate bivariate GMM where y_obs may contain NaN for pre-IPO missing observations
    GMMFitResult fit_em(const std::vector<double>& x_obs, const std::vector<double>& y_obs);

    // Simulate correlated bivariate return samples from fitted mixture
    void sample_paths(std::size_t n_samples, uint64_t seed,
                      std::vector<double>& out_x, std::vector<double>& out_y) const;

    const std::vector<BivariateComponent>& components() const { return components_; }

    // Evaluate marginal density of asset Y (or asset X) at point u
    double pdf_y(double y) const;
    double cdf_y(double y) const;

    static double normal_pdf(double x, double mu, double sigma);
    static double normal_cdf(double x, double mu, double sigma);
    static double bivariate_log_pdf(double x, double y, const BivariateComponent& c);

private:
    int K_;
    int max_iter_;
    double tol_;
    double reg_covar_;
    std::vector<BivariateComponent> components_;
};

// Generate realistic regime-switching bivariate equity return series (AAPL + short-history IPO)
void generate_joint_market_dataset(
    std::size_t n_train,
    std::size_t n_test,
    std::size_t n_ipo_observed,
    uint64_t seed,
    std::vector<double>& train_x,
    std::vector<double>& train_y_masked,
    std::vector<double>& test_x,
    std::vector<double>& test_y);

// Compute out-of-sample KL divergence, KS statistic, and 99% VaR accuracy
TailMetrics evaluate_tail_benchmark(
    const BivariateGMM& fitted_gmm,
    const std::vector<double>& train_y_observed,
    const std::vector<double>& test_y,
    std::size_t n_sim_paths = 500000,
    uint64_t seed = 2026);

} // namespace stoch

#endif // STOCH_GMM_EM_HPP

