"""
Rolling VECM estimation, IPES computation, and shock decomposition.

Implements the full mathematical pipeline from the IPES paper
(Shen, Huang, Zuo, Zivot 2026) applied to BTC spot vs BTC perpetual futures:

    VECM -> Psi(1) -> psi -> d^P_{0,i} -> E_i -> IPES
                           -> eta^P_t (permanent shocks)
                           -> eta^T_t (transitory shocks)

The core math is identical to the stablecoin reference project.
Only the wrapper function differs: no cross-rate construction needed
since both spot and perp are USDT-denominated.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.linalg import null_space
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


# ── Configuration ────────────────────────────────────────────────────────────

# Window/step defaults
WINDOW_DEFAULTS = {
    "1s": {"W": 3600, "step": 300},     # 60 min window, 5 min step
    "1m": {"W": 360,  "step": 15},      # 6 hour window, 15 min step
}

# VAR lag order K (VECM has K-1 lagged differences).
DEFAULT_K = 5  # VAR lag order -> VECM(4)

# Johansen significance level for cointegration test
JOHANSEN_SIG = 0.05


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class VECMDiagnostics:
    """Diagnostic results from a single VECM estimation."""
    alpha_opposite_signs: bool       # alpha_1 and alpha_2 have opposite signs
    portmanteau_pvalue: float        # multivariate Ljung-Box p-value
    residuals_stationary: bool       # ADF test on each residual series
    adf_pvalues: tuple[float, float] # ADF p-values for each residual


@dataclass
class VECMResult:
    """Complete output from a single-window VECM estimation."""
    # VECM parameters
    alpha: np.ndarray          # (2,) error correction coefficients
    beta: np.ndarray           # (2,) cointegrating vector
    gamma: list[np.ndarray]    # list of (2,2) short-run coefficient matrices
    residuals: np.ndarray      # (T, 2) VECM residuals (innovations epsilon_t)
    omega: np.ndarray          # (2, 2) residual covariance matrix

    # Derived quantities
    psi1: np.ndarray           # (2, 2) long-run impact matrix Psi(1)
    psi: np.ndarray            # (2,) common row vector of Psi(1)
    d_P: np.ndarray            # (2,) Price Discovery Betas
    E: np.ndarray              # (2,) Pricing Errors |d^P - 1|
    ipes: np.ndarray           # (2,) IPES scores

    # Comparison measures (paper Appendix A2)
    nls: np.ndarray            # (2,) Normalized Leadership Share
    pils: np.ndarray           # (2,) Permanent-Impulse Leadership Share (= NLS)
    covis: np.ndarray          # (2,) Covariance Information Share

    # Shock decomposition (full series over the window)
    eta_P: np.ndarray          # (T,) permanent shock series
    eta_T: np.ndarray          # (T,) transitory shock series

    # Diagnostics
    diagnostics: VECMDiagnostics

    # Cointegration test
    cointegration_holds: bool
    johansen_trace_pvalue: float  # approximate: trace stat vs critical value


@dataclass
class WindowSignals:
    """Signals produced at a single time step for the backtester."""
    timestamp: pd.Timestamp
    cointegration_holds: bool
    alpha_signs_opposite: bool
    d_P: np.ndarray            # (2,) Price Discovery Betas
    E: np.ndarray              # (2,) Pricing Errors
    ipes: np.ndarray           # (2,) IPES scores
    nls: np.ndarray            # (2,) Normalized Leadership Share
    pils: np.ndarray           # (2,) Permanent-Impulse Leadership Share
    covis: np.ndarray          # (2,) Covariance Information Share
    z_trans: float             # transitory shock z-score (exponentially weighted)
    z_perm: float              # permanent shock z-score (exponentially weighted)
    z_spread: float            # spread z-score (for P1 -- raw price diff)
    z_coint_resid: float       # cointegrating residual z-score (for P2 -- beta'log_p)
    spread: float              # raw spread value
    prices: np.ndarray         # (2,) current close prices
    eta_P_last: float          # last permanent shock value
    eta_T_last: float          # last transitory shock value
    portmanteau_pvalue: float
    residuals_stationary: bool


@dataclass
class RollingResult:
    """Complete output from a rolling estimation run."""
    signals: list[WindowSignals]
    params: dict               # estimation parameters used

    def to_dataframe(self) -> pd.DataFrame:
        """Convert signals list to a DataFrame indexed by timestamp."""
        records = []
        for s in self.signals:
            records.append({
                "timestamp": s.timestamp,
                "cointegration_holds": s.cointegration_holds,
                "alpha_signs_opposite": s.alpha_signs_opposite,
                "d_P_0": s.d_P[0], "d_P_1": s.d_P[1],
                "E_0": s.E[0], "E_1": s.E[1],
                "ipes_0": s.ipes[0], "ipes_1": s.ipes[1],
                "nls_0": s.nls[0], "nls_1": s.nls[1],
                "pils_0": s.pils[0], "pils_1": s.pils[1],
                "covis_0": s.covis[0], "covis_1": s.covis[1],
                "z_trans": s.z_trans, "z_perm": s.z_perm,
                "z_spread": s.z_spread, "z_coint_resid": s.z_coint_resid,
                "spread": s.spread,
                "price_0": s.prices[0], "price_1": s.prices[1],
                "eta_P": s.eta_P_last, "eta_T": s.eta_T_last,
                "portmanteau_pvalue": s.portmanteau_pvalue,
                "residuals_stationary": s.residuals_stationary,
            })
        df = pd.DataFrame(records).set_index("timestamp")
        return df


# ── Core VECM estimation ─────────────────────────────────────────────────────


def johansen_cointegration_test(
    log_prices: np.ndarray,
    det_order: int = 0,
    k_ar_diff: int = 4,
) -> tuple[bool, float]:
    """
    Run Johansen trace test for cointegration.

    Parameters
    ----------
    log_prices  : (T, 2) array of log-prices
    det_order   : deterministic term (0 = constant in CI eq, -1 = none, 1 = trend)
    k_ar_diff   : number of lagged differences (VECM lag order = K-1)

    Returns
    -------
    (cointegration_holds, trace_stat_ratio)
    trace_stat_ratio = trace_stat / critical_value_5pct for r=0.
    If > 1.0, we reject "no cointegration" at 5%.
    """
    result = coint_johansen(log_prices, det_order=det_order, k_ar_diff=k_ar_diff)
    # Trace test for r=0 (no cointegration): first element
    trace_stat = result.lr1[0]          # trace statistic for r=0
    crit_5pct = result.cvt[0, 1]        # 5% critical value for r=0
    holds = trace_stat > crit_5pct
    ratio = trace_stat / crit_5pct if crit_5pct > 0 else np.inf
    return holds, ratio


def estimate_vecm(
    log_prices: np.ndarray,
    K: int = DEFAULT_K,
    deterministic: Literal["ci", "co"] = "ci",
    constrain_beta: bool = True,
) -> VECMResult:
    """
    Estimate a VECM and compute the full IPES pipeline.

    Parameters
    ----------
    log_prices      : (T, 2) array of log-close prices
    K               : VAR lag order (VECM has K-1 lagged differences)
    deterministic   : 'ci' = constant inside CI eq (Johansen Case 2, primary),
                      'co' = constant outside CI eq (Johansen Case 3, robustness)
    constrain_beta  : if True, force beta = (1, -1)'; else freely estimate

    Returns
    -------
    VECMResult with all IPES quantities and diagnostics.
    """
    T, n = log_prices.shape
    assert n == 2, "Only bivariate VECM supported"

    # ── Johansen cointegration test ──────────────────────────────────
    # det_order=0 gives Case 2 (restricted constant) critical values;
    # det_order=1 gives Case 3 (unrestricted constant) critical values.
    # Must match the 'deterministic' parameter to avoid a critical-value
    # mismatch: 'ci' -> Case 2 (det_order=0), 'co' -> Case 3 (det_order=1).
    det_order = 1 if deterministic == 'co' else 0
    coint_holds, coint_ratio = johansen_cointegration_test(
        log_prices, det_order=det_order, k_ar_diff=K - 1
    )

    # ── Fit VECM ─────────────────────────────────────────────────────
    model = VECM(log_prices, k_ar_diff=K - 1, coint_rank=1,
                 deterministic=deterministic)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit()

    # Extract parameters
    alpha_raw = fit.alpha.flatten()    # (2,) error correction loadings
    beta_raw = fit.beta[:2].flatten()  # (2,) cointegrating vector (first 2 elements)

    if constrain_beta:
        beta = np.array([1.0, -1.0])
    else:
        # Normalize so first element = 1
        beta = beta_raw / beta_raw[0]

    alpha = alpha_raw  # alpha is always estimated from data

    # Short-run Gamma_j matrices: fit.gamma has shape (K-1, 2, 2) effectively
    # statsmodels stores them as a (2*(K-1), 2) matrix -- reshape
    gamma_matrices = []
    gamma_flat = fit.gamma  # shape: (2, 2*(K-1))
    for j in range(K - 1):
        Gj = gamma_flat[:, j * 2:(j + 1) * 2]
        gamma_matrices.append(Gj)

    # Residuals (innovations epsilon_t)
    resid = fit.resid  # (T - K, 2)

    # Residual covariance Omega
    omega = np.cov(resid, rowvar=False, bias=False)

    # ── Psi(1) computation (IPES paper Eq. 6) ─────────────────────────
    # Gamma(1) = I_2 - sum(Gamma_j)
    Gamma1 = np.eye(2) - sum(gamma_matrices) if gamma_matrices else np.eye(2)

    # beta_perp: orthogonal complement of beta
    # For beta = (1, -1)', beta_perp = (1, 1)' (up to scalar)
    if constrain_beta:
        beta_perp = np.array([1.0, 1.0])
    else:
        ns = null_space(beta.reshape(1, -1))
        beta_perp = ns.flatten()

    # alpha_perp: orthogonal complement of alpha
    ns_alpha = null_space(alpha.reshape(1, -1))
    alpha_perp = ns_alpha.flatten()

    # Psi(1) = beta_perp (alpha'_perp Gamma(1) beta_perp)^{-1} alpha'_perp
    # alpha'_perp Gamma(1) beta_perp is a scalar (1x1) for bivariate case
    scalar = alpha_perp @ Gamma1 @ beta_perp
    if abs(scalar) < 1e-12:
        # Degenerate -- Psi(1) cannot be computed
        scalar = 1e-12  # prevent division by zero
    psi1 = np.outer(beta_perp, alpha_perp) / scalar

    # psi = common row of Psi(1).
    # When beta = (1,-1)', beta_perp = (1,1)' so outer(beta_perp, alpha_perp)
    # has identical rows and taking row 0 is exact. When constrain_beta=False,
    # beta_perp elements may differ slightly from 1, making rows non-identical;
    # psi1[0] is then an approximation valid for beta near (1,-1)'.
    psi = psi1[0]

    # ── d^P_{0,i} (IPES paper Eq. 13) ───────────────────────────────
    sigma_P_sq = psi @ omega @ psi  # var(eta^P)
    if sigma_P_sq < 1e-15:
        sigma_P_sq = 1e-15
    d_P = np.array([
        (psi[i] * omega[i, i] + psi[1 - i] * omega[i, 1 - i]) / sigma_P_sq
        for i in range(2)
    ])

    # ── E_i and IPES (Eqs. 23, 25) ──────────────────────────────────
    E = np.abs(d_P - 1.0)
    exp_neg_E = np.exp(-E)
    ipes = exp_neg_E / exp_neg_E.sum()

    # ── NLS / PILS / CovIS (Appendix A2) ─────────────────────────────
    # NLS_i = PILS_i = |d^P_{0,i}|^2 / sum_j |d^P_{0,j}|^2
    d_P_sq = d_P ** 2
    sum_sq = d_P_sq.sum()
    if sum_sq < 1e-15:
        nls = np.array([0.5, 0.5])
    else:
        nls = d_P_sq / sum_sq
    pils = nls.copy()

    # CovIS_i = d^P_{0,i} / sum_j d^P_{0,j}
    sum_d = d_P.sum()
    if abs(sum_d) < 1e-15:
        covis = np.array([0.5, 0.5])
    else:
        covis = d_P / sum_d

    # ── Shock decomposition ──────────────────────────────────────────
    # Permanent shock: eta^P_t = psi' epsilon_t
    eta_P = resid @ psi                  # (T,)
    # Transitory shock: eta^T_t = beta' epsilon_t
    eta_T = resid @ beta                 # (T,)

    # ── Diagnostics ──────────────────────────────────────────────────
    diag = _run_diagnostics(alpha, resid)

    return VECMResult(
        alpha=alpha, beta=beta, gamma=gamma_matrices,
        residuals=resid, omega=omega,
        psi1=psi1, psi=psi, d_P=d_P, E=E, ipes=ipes,
        nls=nls, pils=pils, covis=covis,
        eta_P=eta_P, eta_T=eta_T,
        diagnostics=diag,
        cointegration_holds=coint_holds,
        johansen_trace_pvalue=coint_ratio,
    )


def _run_diagnostics(alpha: np.ndarray, resid: np.ndarray) -> VECMDiagnostics:
    """Run VECM residual diagnostics."""
    # alpha sign check: alpha_1 and alpha_2 must have opposite signs
    alpha_opposite = (alpha[0] * alpha[1]) < 0

    # Portmanteau test (multivariate Ljung-Box on residuals)
    # Use univariate LB on each series, take the minimum p-value
    # (conservative approach -- if either series is autocorrelated, flag it)
    try:
        lb0 = acorr_ljungbox(resid[:, 0], lags=10, return_df=True)
        lb1 = acorr_ljungbox(resid[:, 1], lags=10, return_df=True)
        port_pval = min(lb0["lb_pvalue"].iloc[-1], lb1["lb_pvalue"].iloc[-1])
    except Exception:
        port_pval = np.nan

    # ADF test on residuals (confirm I(0) -- stationary)
    try:
        adf0 = adfuller(resid[:, 0], maxlag=5, autolag="AIC")[1]
        adf1 = adfuller(resid[:, 1], maxlag=5, autolag="AIC")[1]
    except Exception:
        adf0 = adf1 = np.nan

    resid_stationary = (
        (not np.isnan(adf0) and adf0 < 0.05) and
        (not np.isnan(adf1) and adf1 < 0.05)
    )

    return VECMDiagnostics(
        alpha_opposite_signs=alpha_opposite,
        portmanteau_pvalue=port_pval,
        residuals_stationary=resid_stationary,
        adf_pvalues=(adf0, adf1),
    )


# ── Z-score computation with exponential weighting ──────────────────────────


def _ewm_zscore(series: np.ndarray, halflife: int) -> float:
    """
    Compute the exponentially weighted z-score of the last observation.

    Half-life = W/4 by default (set via halflife_ratio in rolling_estimate).
    """
    n = len(series)
    if n < 3:
        return 0.0

    # Exponential weights: w_i = (1/2)^((n-1-i)/halflife)
    idx = np.arange(n)
    weights = np.power(0.5, (n - 1 - idx) / max(halflife, 1))
    weights /= weights.sum()

    wmean = np.average(series, weights=weights)
    wvar = np.average((series - wmean) ** 2, weights=weights)
    wstd = np.sqrt(wvar) if wvar > 0 else 1e-10

    return (series[-1] - wmean) / wstd


# ── Rolling estimation engine ────────────────────────────────────────────────


def rolling_estimate(
    log_prices: pd.DataFrame,
    raw_prices: pd.DataFrame,
    interval: str = "1m",
    W: Optional[int] = None,
    step: Optional[int] = None,
    K: int = DEFAULT_K,
    deterministic: Literal["ci", "co"] = "ci",
    constrain_beta: bool = True,
    halflife_ratio: float = 0.25,
    verbose: bool = True,
) -> RollingResult:
    """
    Walk-forward rolling VECM estimation producing signals at each step.

    Parameters
    ----------
    log_prices     : DataFrame with 2 columns of log-close prices, UTC index
    raw_prices     : DataFrame with 2 columns of raw close prices (for spread)
    interval       : "1m" or "1s" (sets default W and step if not provided)
    W              : window size in observations (overrides interval default)
    step           : step size in observations (overrides interval default)
    K              : VAR lag order
    deterministic  : 'ci' or 'co'
    constrain_beta : force beta = (1, -1)' if True
    halflife_ratio : halflife = W * halflife_ratio for exponential weighting
    verbose        : print progress

    Returns
    -------
    RollingResult with list of WindowSignals and estimation parameters.
    """
    defaults = WINDOW_DEFAULTS.get(interval, WINDOW_DEFAULTS["1m"])
    if W is None:
        W = defaults["W"]
    if step is None:
        step = defaults["step"]

    halflife = max(int(W * halflife_ratio), 1)
    T = len(log_prices)
    col0, col1 = log_prices.columns[0], log_prices.columns[1]

    signals = []
    n_success = 0
    n_fail = 0

    # Walk forward: estimate on [i, i+W), signal at i+W-1, trade at i+W
    for start in range(0, T - W + 1, step):
        end = start + W
        window_log = log_prices.iloc[start:end].values  # (W, 2)
        window_raw = raw_prices.iloc[start:end]
        ts = log_prices.index[end - 1]  # timestamp of this signal

        try:
            result = estimate_vecm(
                window_log, K=K,
                deterministic=deterministic,
                constrain_beta=constrain_beta,
            )
        except Exception as exc:
            n_fail += 1
            if verbose and n_fail <= 5:
                print(f"  VECM failed at {ts}: {exc}")
            continue

        n_success += 1

        # Z-scores on shock series (exponentially weighted)
        z_trans = _ewm_zscore(result.eta_T, halflife)
        z_perm = _ewm_zscore(result.eta_P, halflife)

        # Spread z-score (for P1 benchmark -- raw price difference)
        spread_series = window_raw[col0].values - window_raw[col1].values
        z_spread = _ewm_zscore(spread_series, halflife)

        # Cointegrating residual z-score (for P2 -- beta'log_p)
        coint_resid_series = window_log @ result.beta
        z_coint_resid = _ewm_zscore(coint_resid_series, halflife)

        current_prices = np.array([
            window_raw[col0].iloc[-1],
            window_raw[col1].iloc[-1],
        ])

        signals.append(WindowSignals(
            timestamp=ts,
            cointegration_holds=result.cointegration_holds,
            alpha_signs_opposite=result.diagnostics.alpha_opposite_signs,
            d_P=result.d_P,
            E=result.E,
            ipes=result.ipes,
            nls=result.nls,
            pils=result.pils,
            covis=result.covis,
            z_trans=z_trans,
            z_perm=z_perm,
            z_spread=z_spread,
            z_coint_resid=z_coint_resid,
            spread=spread_series[-1],
            prices=current_prices,
            eta_P_last=result.eta_P[-1],
            eta_T_last=result.eta_T[-1],
            portmanteau_pvalue=result.diagnostics.portmanteau_pvalue,
            residuals_stationary=result.diagnostics.residuals_stationary,
        ))

    if verbose:
        print(f"  Rolling estimation: {n_success} windows OK, {n_fail} failed, "
              f"W={W}, step={step}, K={K}")

    return RollingResult(
        signals=signals,
        params={
            "W": W, "step": step, "K": K,
            "deterministic": deterministic,
            "constrain_beta": constrain_beta,
            "halflife": halflife,
            "interval": interval,
        },
    )


# ── Convenience: BTC spot vs perpetual futures ────────────────────────────────


def run_spot_perp_estimation(
    spot_close: pd.Series,
    perp_close: pd.Series,
    interval: str = "1m",
    **kwargs,
) -> RollingResult:
    """
    Run rolling VECM/IPES estimation on BTC spot vs BTC perpetual futures.

    Series 1: log(BTC_spot)
    Series 2: log(BTC_perp)

    This is SIMPLER than the stablecoin case -- no cross-rate construction
    needed. Both series are direct USDT-denominated prices of the same
    underlying asset.
    """
    # Inner-join on timestamps
    raw = pd.concat([spot_close, perp_close], axis=1, join="inner").dropna()
    raw_prices = raw.copy()
    raw_prices.columns = ["BTC_SPOT", "BTC_PERP"]

    # Log prices -- just log of raw prices, no cross-rate
    log_p = np.log(raw_prices)
    log_p.columns = ["BTC_SPOT", "BTC_PERP"]

    return rolling_estimate(log_p, raw_prices, interval=interval, **kwargs)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.load_data import load_aligned_pair

    print("Loading BTC spot and perp 1m data...")
    spot_c, perp_c = load_aligned_pair()

    # Use a small slice for testing: first 2 days
    start, end = spot_c.index[0], spot_c.index[0] + pd.Timedelta(days=2)
    spot_slice = spot_c[start:end]
    perp_slice = perp_c[start:end]

    print(f"Test slice: {len(spot_slice)} spot, {len(perp_slice)} perp rows")
    print(f"Range: {spot_slice.index[0]} -> {spot_slice.index[-1]}")

    result = run_spot_perp_estimation(
        spot_slice, perp_slice, interval="1m",
        W=360, step=60, K=5, verbose=True,
    )

    if result.signals:
        df = result.to_dataframe()
        print(f"\nSignals DataFrame: {len(df)} rows")
        print(df[["d_P_0", "d_P_1", "E_0", "E_1", "ipes_0", "ipes_1",
                   "nls_0", "pils_0", "covis_0",
                   "cointegration_holds"]].head(10).to_string())
    else:
        print("\nNo signals produced (not enough data or all windows failed)")
