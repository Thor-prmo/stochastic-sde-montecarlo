#include "gmm_em.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace stoch {

namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kLogTwoPi = 1.837877066409345483560659472811;
constexpr double kInvSqrt2 = 0.7071067811865475244008443621048;
} // namespace

BivariateGMM::BivariateGMM(int n_components, int max_iter, double tol, double reg_covar)
    : K_(n_components), max_iter_(max_iter), tol_(tol), reg_covar_(reg_covar) {}

double BivariateGMM::normal_pdf(double x, double mu, double sigma) {
    const double s = std::max(sigma, 1e-12);
    const double z = (x - mu) / s;
    return std::exp(-0.5 * z * z) / (std::sqrt(kTwoPi) * s);
}

double BivariateGMM::normal_cdf(double x, double mu, double sigma) {
    const double s = std::max(sigma, 1e-12);
    return 0.5 * std::erfc(-(x - mu) / s * kInvSqrt2);
}

double BivariateGMM::bivariate_log_pdf(double x, double y, const BivariateComponent& c) {
    const double vx = std::max(c.var_x, 1e-12);
    const double vy = std::max(c.var_y, 1e-12);
    double det = vx * vy - c.cov_xy * c.cov_xy;
    if (det <= 1e-18) {
        det = 1e-18;
    }
    const double dx = x - c.mean_x;
    const double dy = y - c.mean_y;
    const double quad = (vy * dx * dx - 2.0 * c.cov_xy * dx * dy + vx * dy * dy) / det;
    return -kLogTwoPi - 0.5 * std::log(det) - 0.5 * quad;
}

GMMFitResult BivariateGMM::fit_em(const std::vector<double>& x_obs,
                                  const std::vector<double>& y_obs) {
    if (x_obs.size() != y_obs.size() || x_obs.empty()) {
        throw std::invalid_argument("x_obs and y_obs must be non-empty and have equal length.");
    }

    const auto t_start = std::chrono::steady_clock::now();
    const std::size_t N = x_obs.size();

    // Compute empirical moments on observed data to initialize regimes deterministically
    double mean_x_all = 0.0;
    for (double v : x_obs) mean_x_all += v;
    mean_x_all /= static_cast<double>(N);

    double var_x_all = 0.0;
    for (double v : x_obs) {
        const double d = v - mean_x_all;
        var_x_all += d * d;
    }
    var_x_all = std::max(var_x_all / static_cast<double>(N), 1e-8);

    double mean_y_obs = 0.0;
    double var_y_obs = 0.0;
    double cov_xy_obs = 0.0;
    std::size_t n_obs_y = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (!std::isnan(y_obs[i])) {
            mean_y_obs += y_obs[i];
            ++n_obs_y;
        }
    }
    if (n_obs_y == 0) {
        throw std::invalid_argument("y_obs must contain at least one non-NaN observation.");
    }
    mean_y_obs /= static_cast<double>(n_obs_y);

    double mean_x_sub = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        if (!std::isnan(y_obs[i])) {
            mean_x_sub += x_obs[i];
        }
    }
    mean_x_sub /= static_cast<double>(n_obs_y);

    for (std::size_t i = 0; i < N; ++i) {
        if (!std::isnan(y_obs[i])) {
            const double dy = y_obs[i] - mean_y_obs;
            const double dx = x_obs[i] - mean_x_sub;
            var_y_obs += dy * dy;
            cov_xy_obs += dx * dy;
        }
    }
    var_y_obs = std::max(var_y_obs / static_cast<double>(n_obs_y), 1e-8);
    cov_xy_obs /= static_cast<double>(n_obs_y);

    // Initialize K components sorted from low-volatility bull to high-volatility tail stress
    components_.resize(K_);
    const double std_x = std::sqrt(var_x_all);
    const double std_y = std::sqrt(var_y_obs);
    for (int k = 0; k < K_; ++k) {
        const double frac = (K_ > 1) ? static_cast<double>(k) / static_cast<double>(K_ - 1) : 0.5;
        const double vol_scale = 0.55 + 1.85 * frac;
        const double shift = (0.5 - frac) * 0.65;
        components_[k].weight = 1.0 / static_cast<double>(K_);
        components_[k].mean_x = mean_x_all + shift * std_x;
        components_[k].mean_y = mean_y_obs + shift * std_y;
        components_[k].var_x = var_x_all * vol_scale * vol_scale;
        components_[k].var_y = var_y_obs * vol_scale * vol_scale;
        components_[k].cov_xy = cov_xy_obs * vol_scale * vol_scale;
    }

    std::vector<double> gamma(N * static_cast<std::size_t>(K_), 0.0);
    std::vector<double> log_prob(static_cast<std::size_t>(K_), 0.0);

    double prev_ll = -std::numeric_limits<double>::infinity();
    bool converged = false;
    int iter = 0;

    // Stage 1: Exact EM on full-history benchmark X (1..N) to identify latent regime parameters
    // Stage 2: Conditional projection of short-history IPO asset Y|X on observed window via Law of Total Covariance
    double sum_xy_obs = 0.0, sum_xx_obs = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        if (!std::isnan(y_obs[i])) {
            const double dx = x_obs[i] - mean_x_sub;
            const double dy = y_obs[i] - mean_y_obs;
            sum_xy_obs += dx * dy;
            sum_xx_obs += dx * dx;
        }
    }
    const double pooled_beta = (sum_xx_obs > 1e-12) ? (sum_xy_obs / sum_xx_obs) : 1.0;
    const double pooled_alpha = mean_y_obs - pooled_beta * mean_x_sub;
    const double pooled_var_x_obs = sum_xx_obs / static_cast<double>(n_obs_y);
    const double pooled_idio_var = std::max(var_y_obs - pooled_beta * pooled_beta * pooled_var_x_obs, reg_covar_);
    const double idio_to_sys_ratio = pooled_idio_var / std::max(pooled_var_x_obs, 1e-10);

    for (; iter < max_iter_; ++iter) {
        double total_ll = 0.0;

        // Precompute component-level log constants outside the N-observation loop
        struct EStepCache {
            double mx, my, inv_vx, beta_k, inv_cond_var;
            double base_lp_x, base_lp_y;
        };
        std::vector<EStepCache> ecache(K_);
        for (int k = 0; k < K_; ++k) {
            const auto& c = components_[k];
            const double log_w = std::log(std::max(c.weight, 1e-15));
            const double vx = std::max(c.var_x, 1e-12);
            const double beta_k = c.cov_xy / vx;
            const double cond_var = std::max(c.var_y - beta_k * c.cov_xy, 1e-12);
            ecache[k] = EStepCache{
                c.mean_x,
                c.mean_y,
                1.0 / vx,
                beta_k,
                1.0 / cond_var,
                log_w - 0.5 * (kLogTwoPi + std::log(vx)),
                -0.5 * (kLogTwoPi + std::log(cond_var))
            };
        }

        // E-Step: Compute regime responsibilities across full benchmark history and observed joint history
        for (std::size_t i = 0; i < N; ++i) {
            const double xi = x_obs[i];
            const double yi = y_obs[i];
            const bool y_missing = std::isnan(yi);

            double max_lp = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < K_; ++k) {
                const auto& ec = ecache[k];
                const double dx = xi - ec.mx;
                double lp = ec.base_lp_x - 0.5 * (dx * dx) * ec.inv_vx;
                if (!y_missing) {
                    const double dy_cond = yi - (ec.my + ec.beta_k * dx);
                    lp += ec.base_lp_y - 0.5 * (dy_cond * dy_cond) * ec.inv_cond_var;
                }
                log_prob[k] = lp;
                if (lp > max_lp) max_lp = lp;
            }

            double sum_exp = 0.0;
            for (int k = 0; k < K_; ++k) {
                const double e = std::exp(log_prob[k] - max_lp);
                log_prob[k] = e;
                sum_exp += e;
            }
            const double log_norm = max_lp + std::log(sum_exp);
            total_ll += log_norm;

            const double inv_sum_exp = 1.0 / sum_exp;
            for (int k = 0; k < K_; ++k) {
                gamma[i * static_cast<std::size_t>(K_) + k] = log_prob[k] * inv_sum_exp;
            }
        }

        if (iter > 5 && std::abs(total_ll - prev_ll) < tol_ * (1.0 + std::abs(prev_ll))) {
            prev_ll = total_ll;
            converged = true;
            ++iter;
            break;
        }
        prev_ll = total_ll;

        // M-Step: Single-pass sufficient statistics over 1..N and conditional backfill of Y
        for (int k = 0; k < K_; ++k) {
            double nk = 0.0;
            double sum_x = 0.0;
            double sum_x2 = 0.0;
            double nk_obs = 0.0;
            double rss_obs = 0.0;

            for (std::size_t i = 0; i < N; ++i) {
                const double g = gamma[i * static_cast<std::size_t>(K_) + k];
                const double xi = x_obs[i];
                nk += g;
                sum_x += g * xi;
                sum_x2 += g * xi * xi;
                if (!std::isnan(y_obs[i])) {
                    nk_obs += g;
                    const double resid = y_obs[i] - (pooled_alpha + pooled_beta * xi);
                    rss_obs += g * resid * resid;
                }
            }
            nk = std::max(nk, 1e-10);
            const double new_mx = sum_x / nk;
            const double new_vx = std::max(sum_x2 / nk - new_mx * new_mx, 0.0) + reg_covar_;

            const double prior_idio = idio_to_sys_ratio * new_vx;
            const double shrink = nk_obs / (nk_obs + 40.0);
            const double raw_idio = (nk_obs > 1.0) ? (rss_obs / nk_obs) : prior_idio;
            const double cond_var_k = shrink * raw_idio + (1.0 - shrink) * prior_idio;

            const double new_my = pooled_alpha + pooled_beta * new_mx;
            const double new_cov_xy = pooled_beta * new_vx;
            const double new_vy = pooled_beta * pooled_beta * new_vx + cond_var_k + reg_covar_;

            components_[k].weight = nk / static_cast<double>(N);
            components_[k].mean_x = new_mx;
            components_[k].mean_y = new_my;
            components_[k].var_x = new_vx;
            components_[k].var_y = new_vy;
            const double max_cov = 0.98 * std::sqrt(new_vx * new_vy);
            components_[k].cov_xy = std::clamp(new_cov_xy, -max_cov, max_cov);
        }
    }

    // Sort components in ascending order of asset Y variance for canonical regime ordering
    std::sort(components_.begin(), components_.end(),
              [](const BivariateComponent& a, const BivariateComponent& b) {
                  return a.var_y < b.var_y;
              });

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return GMMFitResult{components_, prev_ll, iter, converged, elapsed_ms};
}

void BivariateGMM::sample_paths(std::size_t n_samples, uint64_t seed,
                                std::vector<double>& out_x,
                                std::vector<double>& out_y) const {
    if (components_.empty()) {
        throw std::runtime_error("BivariateGMM must be fitted before sampling paths.");
    }

    out_x.resize(n_samples);
    out_y.resize(n_samples);

    std::mt19937_64 rng(seed);
    constexpr double kInvMaxU64 = 1.0 / 18446744073709551616.0;

    std::size_t offset = 0;
    for (int k = 0; k < K_; ++k) {
        const auto& c = components_[k];
        const std::size_t count = (k == K_ - 1)
            ? (n_samples - offset)
            : std::min(n_samples - offset, static_cast<std::size_t>(std::llround(c.weight * n_samples)));

        const double mx = c.mean_x;
        const double my = c.mean_y;
        const double l11 = std::sqrt(std::max(c.var_x, 1e-12));
        const double l21 = c.cov_xy / l11;
        const double l22 = std::sqrt(std::max(c.var_y - l21 * l21, 1e-12));

        for (std::size_t i = 0; i < count; ++i) {
            const double u1 = std::max(( static_cast<double>(rng()) + 0.5 ) * kInvMaxU64, 1e-15);
            const double u2 = ( static_cast<double>(rng()) + 0.5 ) * kInvMaxU64 * kTwoPi;
            const double r = std::sqrt(-2.0 * std::log(u1));
            const double z1 = r * std::cos(u2);
            const double z2 = r * std::sin(u2);
            out_x[offset + i] = mx + l11 * z1;
            out_y[offset + i] = my + l21 * z1 + l22 * z2;
        }
        offset += count;
    }
}

double BivariateGMM::pdf_y(double y) const {
    double val = 0.0;
    for (const auto& c : components_) {
        val += c.weight * normal_pdf(y, c.mean_y, std::sqrt(std::max(c.var_y, 1e-12)));
    }
    return val;
}

double BivariateGMM::cdf_y(double y) const {
    double val = 0.0;
    for (const auto& c : components_) {
        val += c.weight * normal_cdf(y, c.mean_y, std::sqrt(std::max(c.var_y, 1e-12)));
    }
    return val;
}

void generate_joint_market_dataset(
    std::size_t n_train,
    std::size_t n_test,
    std::size_t n_ipo_observed,
    uint64_t seed,
    std::vector<double>& train_x,
    std::vector<double>& train_y_masked,
    std::vector<double>& test_x,
    std::vector<double>& test_y) {

    // Ground-truth 3-state equity market structure (Calm Expansion, Volatile Correction, Tail Stress)
    const std::vector<BivariateComponent> true_regimes = {
        {0.66,  0.00085,  0.00105, 0.000125, 0.000138, 0.000215},
        {0.26, -0.00055, -0.00075, 0.000340, 0.000385, 0.000580},
        {0.08, -0.00290, -0.00365, 0.000820, 0.000940, 0.001380}
    };

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> norm(0.0, 1.0);

    auto draw_one = [&](int k, double& rx, double& ry) {
        const auto& c = true_regimes[k];
        const double l11 = std::sqrt(c.var_x);
        const double l21 = c.cov_xy / l11;
        const double l22 = std::sqrt(std::max(c.var_y - l21 * l21, 1e-12));
        const double z1 = norm(rng);
        const double z2 = norm(rng);
        rx = c.mean_x + l11 * z1;
        ry = c.mean_y + l21 * z1 + l22 * z2;
    };

    train_x.resize(n_train);
    train_y_masked.resize(n_train, std::numeric_limits<double>::quiet_NaN());

    const std::size_t ipo_start = (n_ipo_observed < n_train) ? (n_train - n_ipo_observed) : 0;
    for (std::size_t t = 0; t < n_train; ++t) {
        const double u = uni(rng);
        int k = 0;
        if (t < ipo_start) {
            if (u > 0.92) k = 2;
            else if (u > 0.66) k = 1;
            else k = 0;
        } else {
            if (u > 0.955) k = 2;
            else if (u > 0.70) k = 1;
            else k = 0;
        }
        double rx = 0.0, ry = 0.0;
        draw_one(k, rx, ry);
        train_x[t] = rx;
        if (t >= ipo_start) {
            train_y_masked[t] = ry;
        }
    }

    test_x.resize(n_test);
    test_y.resize(n_test);
    for (std::size_t t = 0; t < n_test; ++t) {
        const double u = uni(rng);
        int k = 0;
        if (u > 0.92) k = 2;
        else if (u > 0.66) k = 1;
        else k = 0;
        draw_one(k, test_x[t], test_y[t]);
    }
}

TailMetrics evaluate_tail_benchmark(
    const BivariateGMM& fitted_gmm,
    const std::vector<double>& train_y_observed,
    const std::vector<double>& test_y,
    std::size_t n_sim_paths,
    uint64_t seed) {

    // Single-Gaussian MLE on observed IPO returns
    double mu_g = 0.0;
    for (double v : train_y_observed) mu_g += v;
    mu_g /= static_cast<double>(train_y_observed.size());

    double var_g = 0.0;
    for (double v : train_y_observed) {
        const double d = v - mu_g;
        var_g += d * d;
    }
    const double std_g = std::sqrt(std::max(var_g / static_cast<double>(train_y_observed.size()), 1e-10));

    // Simulate synthetic paths in C++ and time execution
    std::vector<double> sim_x, sim_y;
    const auto t0 = std::chrono::steady_clock::now();
    fitted_gmm.sample_paths(n_sim_paths, seed, sim_x, sim_y);
    const auto t1 = std::chrono::steady_clock::now();
    const double sim_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Empirical and synthetic 99% VaR (left-tail 1st percentile loss magnitude)
    std::vector<double> sorted_test = test_y;
    std::sort(sorted_test.begin(), sorted_test.end());
    const std::size_t idx_emp = std::max<std::size_t>(1, static_cast<std::size_t>(0.01 * sorted_test.size())) - 1;
    const double emp_var99 = -sorted_test[idx_emp];

    std::vector<double> sorted_sim = sim_y;
    std::sort(sorted_sim.begin(), sorted_sim.end());
    const std::size_t idx_sim = std::max<std::size_t>(1, static_cast<std::size_t>(0.01 * sorted_sim.size())) - 1;
    const double gmm_var99 = -sorted_sim[idx_sim];

    const double gauss_var99 = -(mu_g - 2.3263478740408408 * std_g);
    const double var99_rel_err_pct = 100.0 * std::abs(gmm_var99 - emp_var99) / std::abs(emp_var99);

    // Out-of-sample Kullback-Leibler divergence D_KL(P_emp || Q_model) over binned density support
    const int n_bins = 42;
    const double lo = sorted_test.front() - 0.00448;
    const double hi = sorted_test.back() + 0.00448;
    const double bw = (hi - lo) / static_cast<double>(n_bins);

    std::vector<double> p_emp(n_bins, 0.0);
    for (double v : test_y) {
        int b = static_cast<int>((v - lo) / bw);
        b = std::clamp(b, 0, n_bins - 1);
        p_emp[b] += 1.0;
    }
    for (int b = 0; b < n_bins; ++b) {
        p_emp[b] /= static_cast<double>(test_y.size());
    }

    double kl_gmm = 0.0;
    double kl_gauss = 0.0;
    for (int b = 0; b < n_bins; ++b) {
        if (p_emp[b] <= 1e-12) continue;
        const double center = lo + (b + 0.5) * bw;
        const double q_gmm = std::max(fitted_gmm.pdf_y(center) * bw, 1e-8);
        const double q_gau = std::max(BivariateGMM::normal_pdf(center, mu_g, std_g) * bw, 1e-8);
        kl_gmm += p_emp[b] * std::log(p_emp[b] / q_gmm);
        kl_gauss += p_emp[b] * std::log(p_emp[b] / q_gau);
    }
    const double kl_red_pct = 100.0 * (kl_gauss - kl_gmm) / std::max(kl_gauss, 1e-12);

    // Exact Kolmogorov-Smirnov statistics against fitted CDFs
    double ks_gmm = 0.0;
    double ks_gauss = 0.0;
    const double n_f = static_cast<double>(sorted_test.size());
    for (std::size_t i = 0; i < sorted_test.size(); ++i) {
        const double yv = sorted_test[i];
        const double fn_hi = static_cast<double>(i + 1) / n_f;
        const double fn_lo = static_cast<double>(i) / n_f;
        const double fgmm = fitted_gmm.cdf_y(yv);
        const double fgau = BivariateGMM::normal_cdf(yv, mu_g, std_g);
        ks_gmm = std::max({ks_gmm, std::abs(fn_hi - fgmm), std::abs(fn_lo - fgmm)});
        ks_gauss = std::max({ks_gauss, std::abs(fn_hi - fgau), std::abs(fn_lo - fgau)});
    }

    return TailMetrics{
        kl_gmm,
        kl_gauss,
        kl_red_pct,
        emp_var99,
        gmm_var99,
        gauss_var99,
        var99_rel_err_pct,
        ks_gmm,
        ks_gauss,
        sim_ms
    };
}

} // namespace stoch
