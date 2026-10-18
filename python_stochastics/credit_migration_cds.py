"""Continuous-time sovereign credit-rating migration dynamics and CDS term-structure pricing."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.linalg import expm


@dataclass
class CreditMigrationResult:
    ratings: List[str]
    generator_matrix: np.ndarray
    one_year_transition_matrix: np.ndarray
    maturities: np.ndarray
    cumulative_default_probs: Dict[str, np.ndarray]
    model_cds_spreads_bps: Dict[str, np.ndarray]
    empirical_cds_spreads_bps: Dict[str, np.ndarray]
    max_abs_spread_error_bps: float
    mean_abs_spread_error_bps: float


class CreditMigrationModel:
    """Time-homogeneous generator matrix model for sovereign rating transitions and CDS spreads."""

    RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]

    def __init__(self, recovery_rate: float = 0.40, risk_free_rate: float = 0.04):
        if not (0.0 <= recovery_rate < 1.0):
            raise ValueError("recovery_rate must lie in [0, 1).")
        self.recovery_rate = float(recovery_rate)
        self.risk_free_rate = float(risk_free_rate)
        self.generator_Q = self._construct_generator_matrix()

    @classmethod
    def _construct_generator_matrix(cls) -> np.ndarray:
        """Build 8x8 continuous-time intensity matrix Q with absorbing default state D."""
        Q = np.array(
            [
                #   AAA       AA        A       BBB       BB        B      CCC        D
                [ 0.0000,  0.0680,  0.0045,  0.0010,  0.0003,  0.0001,  0.0000,  0.0001],  # AAA
                [ 0.0140,  0.0000,  0.0720,  0.0055,  0.0012,  0.0004,  0.0001,  0.0004],  # AA
                [ 0.0008,  0.0220,  0.0000,  0.0640,  0.0065,  0.0018,  0.0003,  0.0012],  # A
                [ 0.0003,  0.0020,  0.0380,  0.0000,  0.0520,  0.0095,  0.0018,  0.0032],  # BBB
                [ 0.0001,  0.0008,  0.0042,  0.0560,  0.0000,  0.0780,  0.0140,  0.0135],  # BB
                [ 0.0000,  0.0003,  0.0015,  0.0060,  0.0620,  0.0000,  0.0880,  0.0420],  # B
                [ 0.0000,  0.0000,  0.0006,  0.0025,  0.0110,  0.0540,  0.0000,  0.1950],  # CCC
                [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],  # D (Absorbing)
            ],
            dtype=np.float64,
        )
        for i in range(Q.shape[0]):
            Q[i, i] = -np.sum(Q[i, :])
        return Q

    def transition_matrix(self, horizon_years: float) -> np.ndarray:
        """Compute P(t) = exp(Q t) via Padé matrix exponential."""
        if horizon_years < 0.0:
            raise ValueError("horizon_years must be non-negative.")
        P = expm(self.generator_Q * float(horizon_years))
        P = np.maximum(P, 0.0)
        row_sums = P.sum(axis=1, keepdims=True)
        return P / np.where(row_sums > 0.0, row_sums, 1.0)

    def cumulative_default_curve(self, maturities: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute absorption probability into Default state D across maturities for each rating."""
        curves = {r: np.zeros(len(maturities), dtype=np.float64) for r in self.RATINGS[:-1]}
        default_idx = len(self.RATINGS) - 1
        for m_idx, t in enumerate(maturities):
            P_t = self.transition_matrix(float(t))
            for r_idx, rating in enumerate(self.RATINGS[:-1]):
                curves[rating][m_idx] = float(P_t[r_idx, default_idx])
        return curves

    def price_cds_par_spread_bps(self, rating: str, maturity_years: float, n_quad: int = 80) -> float:
        """Compute par CDS spread in basis points from continuous-time survival probabilities."""
        if rating not in self.RATINGS[:-1]:
            raise ValueError(f"Unknown non-default rating: {rating}")
        r_idx = self.RATINGS.index(rating)
        default_idx = len(self.RATINGS) - 1

        t_grid = np.linspace(0.0, float(maturity_years), n_quad + 1)
        dt = t_grid[1] - t_grid[0]
        P_dt = expm(self.generator_Q * dt)

        P_curr = np.eye(len(self.RATINGS), dtype=np.float64)
        pd_grid = np.zeros(n_quad + 1, dtype=np.float64)
        for k in range(1, n_quad + 1):
            P_curr = P_curr @ P_dt
            pd_grid[k] = float(P_curr[r_idx, default_idx])

        surv_grid = 1.0 - pd_grid
        t_mid = 0.5 * (t_grid[:-1] + t_grid[1:])
        disc_mid = np.exp(-self.risk_free_rate * t_mid)
        dpd = np.maximum(pd_grid[1:] - pd_grid[:-1], 0.0)
        surv_mid = 0.5 * (surv_grid[:-1] + surv_grid[1:])

        protection_leg = (1.0 - self.recovery_rate) * float(np.sum(disc_mid * dpd))
        premium_annuity = float(np.sum(disc_mid * surv_mid * dt))
        return 10_000.0 * protection_leg / max(premium_annuity, 1e-12)

    def evaluate_benchmark(self) -> CreditMigrationResult:
        maturities = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=np.float64)
        pd_curves = self.cumulative_default_curve(maturities)

        tracked_ratings = ["AA", "A", "BBB", "BB", "B"]
        model_spreads: Dict[str, np.ndarray] = {}
        for rating in tracked_ratings:
            model_spreads[rating] = np.array(
                [self.price_cds_par_spread_bps(rating, float(t)) for t in maturities],
                dtype=np.float64,
            )

        # Empirical sovereign CDS term-structure benchmark spreads (in bps) across maturities [1, 2, 3, 5, 7, 10]Y
        empirical_spreads = {
            "AA":  np.array([  3.1,   3.6,   4.1,   5.2,   6.5,   8.6], dtype=np.float64),
            "A":   np.array([  8.6,   9.9,  11.2,  14.1,  17.0,  21.8], dtype=np.float64),
            "BBB": np.array([ 23.5,  27.2,  31.0,  38.4,  45.2,  54.1], dtype=np.float64),
            "BB":  np.array([ 90.8, 101.5, 111.0, 126.5, 138.2, 150.4], dtype=np.float64),
            "B":   np.array([274.5, 296.0, 312.5, 332.0, 342.5, 347.8], dtype=np.float64),
        }

        all_diffs = []
        for rating in tracked_ratings:
            diffs = np.abs(model_spreads[rating] - empirical_spreads[rating])
            all_diffs.append(diffs)
        all_diffs_arr = np.concatenate(all_diffs)

        return CreditMigrationResult(
            ratings=list(self.RATINGS),
            generator_matrix=self.generator_Q.copy(),
            one_year_transition_matrix=self.transition_matrix(1.0),
            maturities=maturities,
            cumulative_default_probs=pd_curves,
            model_cds_spreads_bps=model_spreads,
            empirical_cds_spreads_bps=empirical_spreads,
            max_abs_spread_error_bps=float(np.max(all_diffs_arr)),
            mean_abs_spread_error_bps=float(np.mean(all_diffs_arr)),
        )


def run_credit_migration_benchmark() -> CreditMigrationResult:
    return CreditMigrationModel().evaluate_benchmark()
