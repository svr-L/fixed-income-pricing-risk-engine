"""Vasicek and CIR models under risk-neutral and physical measures.

The implementations follow the standard affine term-structure formulas and
exact transition laws of the two models.  Calibration and estimation examples
use only synthetic fixtures bundled with this repository.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import ncx2


@dataclass(frozen=True)
class ShortRateParams:
    r0: float
    kappa: float
    theta: float
    sigma: float

    @property
    def feller_margin(self) -> float:
        return 2.0 * self.kappa * self.theta - self.sigma * self.sigma


@dataclass(frozen=True)
class CurveCalibrationResult:
    model: str
    params: ShortRateParams
    fitted_zero_rates: np.ndarray
    rmse: float
    max_abs_error: float
    success: bool
    nfev: int
    message: str


@dataclass(frozen=True)
class PhysicalFitResult:
    model: str
    params: ShortRateParams
    log_likelihood: float
    success: bool
    n_obs: int
    message: str


def _curve_inputs(
    maturities: Sequence[float], zero_rates: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(maturities, dtype=float)
    z = np.asarray(zero_rates, dtype=float)
    if t.ndim != 1 or z.ndim != 1 or t.shape != z.shape or t.size < 4:
        raise ValueError("maturities and zero_rates must be matching 1D arrays with >= 4 points")
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(z)) or np.any(t <= 0.0):
        raise ValueError("maturities must be positive and all curve inputs finite")
    return t, z


def _series_inputs(rates: Sequence[float], dt: float) -> np.ndarray:
    r = np.asarray(rates, dtype=float)
    if r.ndim != 1 or r.size < 10:
        raise ValueError("rates must be a 1D series with at least 10 observations")
    if dt <= 0.0 or np.any(~np.isfinite(r)):
        raise ValueError("dt must be positive and rates finite")
    return r


def vasicek_zero_rate(
    maturity: float | np.ndarray,
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> float | np.ndarray:
    """Continuously compounded zero rate implied by Vasicek under Q."""
    t = np.asarray(maturity, dtype=float)
    if np.any(t <= 0.0) or kappa <= 0.0 or sigma < 0.0:
        raise ValueError("maturity and kappa must be positive; sigma must be non-negative")
    B = (1.0 - np.exp(-kappa * t)) / kappa
    variance_adjusted_mean = theta - sigma * sigma / (2.0 * kappa * kappa)
    log_A = variance_adjusted_mean * (B - t) - sigma * sigma * B * B / (4.0 * kappa)
    price_log = log_A - B * r0
    zero = -price_log / t
    return float(zero) if zero.ndim == 0 else zero


def cir_zero_rate(
    maturity: float | np.ndarray,
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> float | np.ndarray:
    """Continuously compounded zero rate implied by CIR under Q."""
    t = np.asarray(maturity, dtype=float)
    if np.any(t <= 0.0) or r0 < 0.0 or min(kappa, theta, sigma) <= 0.0:
        raise ValueError("CIR requires r0>=0 and positive maturity/kappa/theta/sigma")
    gamma = np.sqrt(kappa * kappa + 2.0 * sigma * sigma)
    eg = np.exp(gamma * t)
    denominator = (gamma + kappa) * (eg - 1.0) + 2.0 * gamma
    B = 2.0 * (eg - 1.0) / denominator
    exponent = 2.0 * kappa * theta / (sigma * sigma)
    log_A = exponent * (
        np.log(2.0 * gamma) + 0.5 * (kappa + gamma) * t - np.log(denominator)
    )
    zero = -(log_A - B * r0) / t
    return float(zero) if zero.ndim == 0 else zero


def _normalised_weights(weights: Sequence[float] | None, size: int) -> np.ndarray:
    if weights is None:
        return np.ones(size, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (size,) or np.any(~np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("weights must be positive, finite and match the input length")
    return w / np.mean(w)


def _calibration_result(model: str, opt, fitted: np.ndarray, market: np.ndarray) -> CurveCalibrationResult:
    err = fitted - market
    return CurveCalibrationResult(
        model=model,
        params=ShortRateParams(*map(float, opt.x)),
        fitted_zero_rates=fitted,
        rmse=float(np.sqrt(np.mean(err * err))),
        max_abs_error=float(np.max(np.abs(err))),
        success=bool(opt.success),
        nfev=int(opt.nfev),
        message=str(opt.message),
    )


def calibrate_vasicek_q(
    maturities: Sequence[float],
    market_zero_rates: Sequence[float],
    initial: ShortRateParams | None = None,
    weights: Sequence[float] | None = None,
) -> CurveCalibrationResult:
    t, market = _curve_inputs(maturities, market_zero_rates)
    w = _normalised_weights(weights, t.size)
    guess = initial or ShortRateParams(float(market[0]), 0.30, float(market[-1]), 0.01)
    x0 = np.array([guess.r0, guess.kappa, guess.theta, guess.sigma], dtype=float)
    lo = np.array([-0.10, 1e-4, -0.10, 1e-6])
    hi = np.array([0.30, 10.0, 0.30, 1.00])
    x0 = np.clip(x0, lo + 1e-10, hi - 1e-10)

    def objective(x: np.ndarray) -> np.ndarray:
        return np.sqrt(w) * (np.asarray(vasicek_zero_rate(t, *x)) - market)

    opt = least_squares(
        objective,
        x0=x0,
        bounds=(lo, hi),
        max_nfev=20_000,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        x_scale="jac",
    )
    fitted = np.asarray(vasicek_zero_rate(t, *opt.x))
    return _calibration_result("Vasicek-Q", opt, fitted, market)


def calibrate_cir_q(
    maturities: Sequence[float],
    market_zero_rates: Sequence[float],
    initial: ShortRateParams | None = None,
    weights: Sequence[float] | None = None,
    enforce_feller: bool = True,
) -> CurveCalibrationResult:
    t, market = _curve_inputs(maturities, market_zero_rates)
    if np.any(market <= -0.02):
        raise ValueError("CIR calibration is not intended for deeply negative rate regimes")
    w = _normalised_weights(weights, t.size)
    guess = initial or ShortRateParams(
        max(float(market[0]), 1e-4),
        0.30,
        max(float(market[-1]), 1e-4),
        0.05,
    )
    x0 = np.array([guess.r0, guess.kappa, guess.theta, guess.sigma], dtype=float)
    lo = np.array([1e-8, 1e-4, 1e-8, 1e-5])
    hi = np.array([0.30, 10.0, 0.30, 1.00])
    x0 = np.clip(x0, lo + 1e-10, hi - 1e-10)

    def objective(x: np.ndarray) -> np.ndarray:
        residual = np.sqrt(w) * (np.asarray(cir_zero_rate(t, *x)) - market)
        if enforce_feller:
            breach = max(x[3] * x[3] - 2.0 * x[1] * x[2], 0.0)
            residual = np.append(residual, 25.0 * breach)
        return residual

    opt = least_squares(
        objective,
        x0=x0,
        bounds=(lo, hi),
        max_nfev=30_000,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        x_scale="jac",
    )
    fitted = np.asarray(cir_zero_rate(t, *opt.x))
    return _calibration_result("CIR-Q", opt, fitted, market)


def fit_vasicek_p(rates: Sequence[float], dt: float) -> PhysicalFitResult:
    """Conditional MLE using the exact Gaussian Vasicek transition."""
    r = _series_inputs(rates, dt)
    x = r[:-1]
    y = r[1:]
    design = np.column_stack([np.ones_like(x), x])
    intercept, phi = np.linalg.lstsq(design, y, rcond=None)[0]
    if not 0.0 < phi < 1.0:
        raise ValueError("estimated Vasicek persistence must lie in (0,1)")
    kappa = -np.log(phi) / dt
    theta = intercept / (1.0 - phi)
    errors = y - (intercept + phi * x)
    innovation_var = float(np.mean(errors * errors))
    sigma = np.sqrt(innovation_var * 2.0 * kappa / (1.0 - phi * phi))
    exact_var = sigma * sigma * (1.0 - phi * phi) / (2.0 * kappa)
    loglik = -0.5 * np.sum(np.log(2.0 * np.pi * exact_var) + errors * errors / exact_var)
    return PhysicalFitResult(
        model="Vasicek-P",
        params=ShortRateParams(float(r[0]), float(kappa), float(theta), float(sigma)),
        log_likelihood=float(loglik),
        success=True,
        n_obs=int(r.size),
        message="exact Gaussian transition / conditional MLE",
    )


def _cir_starting_guess(r: np.ndarray, dt: float) -> np.ndarray:
    x = r[:-1]
    dr = r[1:] - x
    design = np.column_stack([np.ones_like(x), x])
    intercept_dt, slope_dt = np.linalg.lstsq(design, dr, rcond=None)[0]
    kappa = max(float(-slope_dt / dt), 0.05)
    theta = max(float(intercept_dt / (kappa * dt)), max(float(np.mean(r)), 1e-4))
    residual = dr - kappa * (theta - x) * dt
    denominator = max(float(np.sum(np.maximum(x, 1e-8) * dt)), 1e-12)
    sigma = max(np.sqrt(max(float(np.sum(residual * residual)) / denominator, 1e-8)), 1e-4)
    sigma = min(sigma, np.sqrt(max(1.8 * kappa * theta, 1e-8)))
    return np.array([kappa, theta, sigma], dtype=float)


def fit_cir_p(
    rates: Sequence[float],
    dt: float,
    initial: ShortRateParams | None = None,
    enforce_feller: bool = True,
) -> PhysicalFitResult:
    """Maximum likelihood using the exact non-central-chi-square transition."""
    r = _series_inputs(rates, dt)
    if np.any(r < 0.0):
        raise ValueError("CIR estimation requires non-negative rate observations")
    previous = r[:-1]
    current = r[1:]
    if initial is None:
        x0 = _cir_starting_guess(r, dt)
    else:
        x0 = np.array([initial.kappa, initial.theta, initial.sigma], dtype=float)

    def negative_log_likelihood(v: np.ndarray) -> float:
        kappa, theta, sigma = map(float, v)
        if min(kappa, theta, sigma) <= 0.0:
            return 1e100
        decay = np.exp(-kappa * dt)
        denominator = sigma * sigma * (1.0 - decay)
        if denominator <= 0.0:
            return 1e100
        scale = denominator / (4.0 * kappa)
        dof = 4.0 * kappa * theta / (sigma * sigma)
        noncentrality = 4.0 * kappa * decay * previous / denominator
        transformed = np.maximum(current / scale, 1e-300)
        log_density = ncx2.logpdf(transformed, dof, noncentrality) - np.log(scale)
        if np.any(~np.isfinite(log_density)):
            return 1e100
        return float(-np.sum(log_density))

    bounds = [(1e-4, 10.0), (1e-6, 0.50), (1e-5, 1.00)]
    constraints = []
    method = "L-BFGS-B"
    if enforce_feller:
        method = "SLSQP"
        constraints = [
            {"type": "ineq", "fun": lambda v: 2.0 * v[0] * v[1] - v[2] * v[2]}
        ]
    opt = minimize(
        negative_log_likelihood,
        x0=x0,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 10_000, "ftol": 1e-11},
    )
    kappa, theta, sigma = map(float, opt.x)
    return PhysicalFitResult(
        model="CIR-P",
        params=ShortRateParams(float(r[0]), kappa, theta, sigma),
        log_likelihood=float(-negative_log_likelihood(opt.x)),
        success=bool(opt.success),
        n_obs=int(r.size),
        message=str(opt.message),
    )


def simulate_vasicek(
    params: ShortRateParams,
    n_steps: int,
    dt: float,
    seed: int | None = None,
) -> np.ndarray:
    if n_steps < 1 or dt <= 0.0 or params.kappa <= 0.0 or params.sigma < 0.0:
        raise ValueError("invalid simulation inputs")
    rng = np.random.default_rng(seed)
    path = np.empty(n_steps + 1, dtype=float)
    path[0] = params.r0
    phi = np.exp(-params.kappa * dt)
    std = params.sigma * np.sqrt((1.0 - phi * phi) / (2.0 * params.kappa))
    for i in range(n_steps):
        mean = params.theta + (path[i] - params.theta) * phi
        path[i + 1] = mean + std * rng.standard_normal()
    return path


def simulate_cir(
    params: ShortRateParams,
    n_steps: int,
    dt: float,
    seed: int | None = None,
) -> np.ndarray:
    if (
        n_steps < 1
        or dt <= 0.0
        or params.r0 < 0.0
        or min(params.kappa, params.theta, params.sigma) <= 0.0
    ):
        raise ValueError("invalid CIR simulation inputs")
    rng = np.random.default_rng(seed)
    path = np.empty(n_steps + 1, dtype=float)
    path[0] = params.r0
    decay = np.exp(-params.kappa * dt)
    denominator = params.sigma * params.sigma * (1.0 - decay)
    scale = denominator / (4.0 * params.kappa)
    dof = 4.0 * params.kappa * params.theta / (params.sigma * params.sigma)
    for i in range(n_steps):
        nc = 4.0 * params.kappa * decay * path[i] / denominator
        path[i + 1] = scale * rng.noncentral_chisquare(dof, nc)
    return path
