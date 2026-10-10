"""Shifted-lognormal SABR smile analytics and swaption volatility cubes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from . import core as fc
from .arbitrage_checks import (
    CheckResult,
    check_butterfly_convexity,
    check_total_variance_monotone,
)
from .volatility import (
    interpolate_across_expiry,
    linear_interpolate,
    validate_expiry_interpolation,
)


@dataclass(frozen=True)
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float
    shift: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        if not -0.999 < self.rho < 0.999:
            raise ValueError("rho must lie strictly between -0.999 and 0.999")
        if self.nu < 0.0:
            raise ValueError("nu must be non-negative")


def _z_over_x(z: float, rho: float) -> float:
    if abs(z) < 1e-8:
        return 1.0 - 0.5 * rho * z
    root = math.sqrt(max(1.0 - 2.0 * rho * z + z * z, 0.0))
    xz = math.log((root + z - rho) / (1.0 - rho))
    return 1.0 if abs(xz) < 1e-14 else z / xz


def _validated_vol(vol: float, strike: float, expiry: float) -> float:
    """Reject a Hagan expansion that has broken down.

    The ``(1 + correction * T)`` term is a truncated expansion, not a bound. For
    large ``nu``, strongly negative ``rho`` or long expiries it can drive the
    result to zero or below, which is a model breakdown rather than a
    volatility. Returning it silently would push a negative number into a Black
    formula further downstream.
    """
    if not math.isfinite(vol) or vol <= 0.0:
        raise ValueError(
            f"Hagan SABR expansion broke down at strike {strike:.6g} and expiry "
            f"{expiry:.4g}, returning a non-positive volatility ({vol:.6g}); "
            "reduce nu, move rho away from the bounds, or shorten the expiry"
        )
    return vol


def sabr_lognormal_vol(
    forward: float,
    strike: float,
    expiry: float,
    params: SABRParams,
) -> float:
    """Hagan shifted-lognormal SABR implied-volatility approximation."""
    if expiry < 0.0:
        raise ValueError("expiry must be non-negative")
    f = forward + params.shift
    k = strike + params.shift
    if f <= 0.0 or k <= 0.0:
        required = max(-forward, -strike, 0.0)
        raise ValueError(
            "shifted SABR requires forward+shift and strike+shift > 0; "
            f"use a shift greater than {required:.8g}"
        )
    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu
    omb = 1.0 - beta
    fk = f * k
    fk_beta = fk ** (0.5 * omb)
    log_fk = math.log(f / k)
    correction = (
        (omb * omb / 24.0) * alpha * alpha / (fk**omb)
        + rho * beta * nu * alpha / (4.0 * fk_beta)
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    )
    if abs(log_fk) < 1e-10:
        return _validated_vol(alpha / (f**omb) * (1.0 + correction * expiry), strike, expiry)
    denominator = fk_beta * (
        1.0
        + omb * omb * log_fk**2 / 24.0
        + omb**4 * log_fk**4 / 1920.0
    )
    z = (nu / alpha) * fk_beta * log_fk if nu > 0.0 else 0.0
    return _validated_vol(
        alpha / denominator * _z_over_x(z, rho) * (1.0 + correction * expiry),
        strike,
        expiry,
    )


def sabr_smile_arbitrage_check(
    params: SABRParams,
    forward: float,
    expiry: float,
    strikes: Sequence[float],
    *,
    refinement: int = 4,
) -> CheckResult:
    """Check that the SABR smile prices convex call payoffs in strike.

    Hagan's expansion is not arbitrage-free: at longer expiries, or with a large
    ``nu`` or strongly negative ``rho``, the wings can produce negative
    butterfly spreads even for parameter sets that look entirely ordinary. This
    is a known property of the approximation, not a calibration failure, which
    is why :func:`calibrate_sabr_smile` reports it instead of raising.

    The grid is densified between the supplied strikes, but never extended past
    them: a violation outside the quoted range says something about
    extrapolation rather than about the fitted smile.
    """
    ks = sorted(float(k) for k in strikes)
    if len(ks) < 3:
        raise ValueError("at least three strikes are required")
    if refinement < 1:
        raise ValueError("refinement must be at least 1")
    grid: list[float] = []
    for left, right in zip(ks, ks[1:], strict=False):
        for step in range(refinement):
            grid.append(left + (right - left) * step / refinement)
    grid.append(ks[-1])

    prices: list[float] = []
    for strike in grid:
        vol = sabr_lognormal_vol(forward, strike, expiry, params)
        prices.append(
            fc.forward_option_premium(
                forward, strike, expiry, vol, 1, model="shifted_black", shift=params.shift
            )
        )
    return check_butterfly_convexity(grid, prices, label=f"SABR smile T={expiry:g}")


@dataclass(frozen=True)
class SABRCalibrationResult:
    params: SABRParams
    rmse_vol: float
    max_abs_vol_error: float
    errors: tuple[float, ...]
    success: bool
    message: str
    arbitrage: CheckResult | None = None
    """Butterfly-convexity diagnostic for the fitted smile, or ``None`` when the
    check was disabled. ``arbitrage.passed`` is False for a smile that is fitted
    well but prices concave call payoffs somewhere in the quoted range."""

    @property
    def arbitrage_free(self) -> bool:
        return self.arbitrage is None or self.arbitrage.passed


def calibrate_sabr_smile(
    forward: float,
    expiry: float,
    strikes: Sequence[float],
    market_vols: Sequence[float],
    *,
    beta: float = 0.5,
    shift: float = 0.0,
    weights: Sequence[float] | None = None,
    initial: SABRParams | None = None,
    check_arbitrage: bool = True,
    strict_arbitrage: bool = False,
) -> SABRCalibrationResult:
    """Fit alpha, rho and nu to a volatility smile at fixed beta and shift.

    Unlike the bootstrapper and the caplet stripper, which raise on an
    inadmissible build, this reports arbitrage rather than refusing: a negative
    butterfly here comes from the Hagan expansion, not from bad quotes, and
    refusing by default would make long expiries unusable. Set
    ``strict_arbitrage`` to turn the diagnostic into an error.
    """
    ks = np.asarray(strikes, dtype=float)
    vols = np.asarray(market_vols, dtype=float)
    if len(ks) < 3 or len(ks) != len(vols):
        raise ValueError("at least three strikes with matching market vols are required")
    if np.any(vols <= 0.0) or np.any(~np.isfinite(vols)):
        raise ValueError("market vols must be positive and finite")
    w = (
        np.ones(len(ks), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    if w.shape != vols.shape or np.any(w <= 0.0) or np.any(~np.isfinite(w)):
        raise ValueError("weights must be positive finite values matching strikes")
    if forward + shift <= 0.0 or np.any(ks + shift <= 0.0):
        raise ValueError("shift is too small for the supplied forward/strikes")
    start = initial or SABRParams(
        max(float(np.median(vols)) * (forward + shift) ** (1.0 - beta), 1e-4),
        beta,
        0.0,
        0.40,
        shift,
    )
    if abs(start.beta - beta) > 1e-14 or abs(start.shift - shift) > 1e-14:
        raise ValueError("initial beta/shift must match fixed calibration beta/shift")

    def residuals(x: np.ndarray) -> np.ndarray:
        try:
            trial = SABRParams(float(x[0]), beta, float(x[1]), float(x[2]), shift)
            fitted = np.array(
                [sabr_lognormal_vol(forward, float(k), expiry, trial) for k in ks]
            )
        except (ValueError, OverflowError, FloatingPointError):
            return np.full(len(ks), 1e6)
        return np.sqrt(w) * (fitted - vols)

    opt = least_squares(
        residuals,
        np.array([start.alpha, start.rho, start.nu]),
        bounds=([1e-8, -0.995, 0.0], [5.0, 0.995, 5.0]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=5000,
    )
    params = SABRParams(float(opt.x[0]), beta, float(opt.x[1]), float(opt.x[2]), shift)
    fitted = np.array(
        [sabr_lognormal_vol(forward, float(k), expiry, params) for k in ks]
    )
    errors = fitted - vols
    arbitrage = (
        sabr_smile_arbitrage_check(params, forward, expiry, ks)
        if check_arbitrage
        else None
    )
    if strict_arbitrage and arbitrage is not None:
        arbitrage.raise_if_failed()
    return SABRCalibrationResult(
        params=params,
        rmse_vol=float(np.sqrt(np.mean(errors * errors))),
        max_abs_vol_error=float(np.max(np.abs(errors))),
        errors=tuple(float(x) for x in errors),
        success=bool(opt.success),
        message=str(opt.message),
        arbitrage=arbitrage,
    )


@dataclass(frozen=True)
class SwaptionVolCube:
    """Expiry x tenor x strike swaption volatility cube.

    Strike and tenor interpolation are linear in volatility. The expiry rule is
    explicit and must match how the cube was built, for the same reason it is
    explicit on :class:`~fi_pricing.volatility.VolSurface2D`: linear-in-vol
    across expiry can put negative forward variance between two admissible
    nodes. The two objects share one implementation so they cannot diverge.
    """

    expiries: tuple[float, ...]
    tenors: tuple[float, ...]
    strikes: tuple[float, ...]
    vols: tuple[tuple[tuple[float, ...], ...], ...]
    expiry_interpolation: str = "linear_vol"

    def __init__(
        self,
        expiries: Sequence[float],
        tenors: Sequence[float],
        strikes: Sequence[float],
        vols: Sequence[Sequence[Sequence[float]]],
        expiry_interpolation: str = "linear_vol",
    ) -> None:
        exps = tuple(map(float, expiries))
        tens = tuple(map(float, tenors))
        ks = tuple(map(float, strikes))
        grid = tuple(
            tuple(tuple(map(float, row)) for row in slab) for slab in vols
        )
        if len(exps) < 2 or len(tens) < 2 or len(ks) < 3:
            raise ValueError("cube needs >=2 expiries, >=2 tenors and >=3 strikes")
        if any(b <= a for a, b in zip(exps, exps[1:], strict=False)):
            raise ValueError("expiries must be strictly increasing")
        if any(b <= a for a, b in zip(tens, tens[1:], strict=False)):
            raise ValueError("tenors must be strictly increasing")
        if any(b <= a for a, b in zip(ks, ks[1:], strict=False)):
            raise ValueError("strikes must be strictly increasing")
        if len(grid) != len(exps) or any(
            len(slab) != len(tens) or any(len(row) != len(ks) for row in slab)
            for slab in grid
        ):
            raise ValueError("vol cube shape must be expiry x tenor x strike")
        if any(v <= 0.0 for slab in grid for row in slab for v in row):
            raise ValueError("cube volatilities must be positive")
        object.__setattr__(self, "expiries", exps)
        object.__setattr__(self, "tenors", tens)
        object.__setattr__(self, "strikes", ks)
        object.__setattr__(self, "vols", grid)
        object.__setattr__(
            self, "expiry_interpolation", validate_expiry_interpolation(expiry_interpolation)
        )

    def smile(self, expiry_index: int, tenor_index: int) -> tuple[float, ...]:
        return self.vols[expiry_index][tenor_index]

    def vol(self, expiry: float, tenor: float, strike: float) -> float:
        by_expiry: list[float] = []
        for slab in self.vols:
            by_tenor = [linear_interpolate(strike, self.strikes, row) for row in slab]
            by_expiry.append(linear_interpolate(tenor, self.tenors, by_tenor))
        return interpolate_across_expiry(
            self.expiry_interpolation, expiry, self.expiries, by_expiry
        )

    def calendar_arbitrage_check(self, tenor: float) -> CheckResult:
        """Total variance must not fall with expiry at a fixed tenor."""
        rows = [
            [linear_interpolate(tenor, self.tenors, [slab[j][k] for j in range(len(self.tenors))])
             for k in range(len(self.strikes))]
            for slab in self.vols
        ]
        return check_total_variance_monotone(self.expiries, rows, self.strikes)


@dataclass(frozen=True)
class SABRCubeCalibration:
    results: tuple[tuple[SABRCalibrationResult, ...], ...]
    rmse_vol: float
    max_abs_vol_error: float
    arbitrage_violations: tuple[tuple[int, int], ...] = ()
    """(expiry index, tenor index) of every slice whose fitted smile fails the
    butterfly check. Empty means the whole cube is admissible."""

    @property
    def arbitrage_free(self) -> bool:
        return not self.arbitrage_violations


def calibrate_sabr_cube(
    cube: SwaptionVolCube,
    forwards: Sequence[Sequence[float]],
    *,
    beta: float = 0.5,
    shift: float = 0.0,
    check_arbitrage: bool = True,
) -> SABRCubeCalibration:
    fwd = np.asarray(forwards, dtype=float)
    if fwd.shape != (len(cube.expiries), len(cube.tenors)):
        raise ValueError("forwards shape must be expiry x tenor")
    rows: list[tuple[SABRCalibrationResult, ...]] = []
    errors: list[float] = []
    violations: list[tuple[int, int]] = []
    for i, expiry in enumerate(cube.expiries):
        row: list[SABRCalibrationResult] = []
        for j, _tenor in enumerate(cube.tenors):
            result = calibrate_sabr_smile(
                float(fwd[i, j]),
                expiry,
                cube.strikes,
                cube.smile(i, j),
                beta=beta,
                shift=shift,
                check_arbitrage=check_arbitrage,
            )
            row.append(result)
            if not result.arbitrage_free:
                violations.append((i, j))
            errors.extend(result.errors)
        rows.append(tuple(row))
    err = np.asarray(errors, dtype=float)
    return SABRCubeCalibration(
        results=tuple(rows),
        rmse_vol=float(np.sqrt(np.mean(err * err))),
        max_abs_vol_error=float(np.max(np.abs(err))),
        arbitrage_violations=tuple(violations),
    )
