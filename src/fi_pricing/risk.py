"""Finite-difference risk analytics for fixed-income instruments.

The module deliberately separates *market risk* sensitivities (curve buckets,
curve factors, volatility-surface buckets) from *model risk* sensitivities to
short-rate parameters.  All functions accept small pricing callables so the
same machinery can be reused across IRS, cap/floor, swaption and future
instruments.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .core import DiscountCurve
from .curves import CurveSet
from .short_rate import ShortRateParams, cir_zero_rate, vasicek_zero_rate
from .volatility import VolSurface2D

CurvePricer = Callable[[DiscountCurve], float]
DatedCurvePricer = Callable[[date, DiscountCurve], float]
SurfacePricer = Callable[[VolSurface2D], float]


@dataclass(frozen=True)
class BucketRisk:
    """Key-rate bucket sensitivity.

    ``convexity`` is the *diagonal* second derivative for this bucket alone.
    Unlike ``dv01``, diagonal convexities are NOT additive to the parallel
    convexity: the cross terms are missing. Use
    :func:`key_rate_convexity_matrix` when the second order must reconcile.
    """

    tenor_years: float
    dv01: float
    convexity: float


@dataclass(frozen=True)
class FactorRisk:
    name: str
    dv01: float
    convexity: float


@dataclass(frozen=True)
class VegaBucket:
    expiry: float
    strike: float
    dv_dvol: float
    vega_1vol_point: float
    vega_1vol_bp: float


@dataclass(frozen=True)
class CarryRollResult:
    horizon_days: int
    base_pv: float
    horizon_pv: float
    cashflows: float
    financing_carry: float
    roll_down: float
    total_pnl: float
    funding_rate: float


@dataclass(frozen=True)
class PnLExplainResult:
    base_pv: float
    shocked_pv: float
    actual_pnl: float
    first_order: float
    second_order: float
    residual: float
    node_delta_per_bp: tuple[float, ...]
    node_gamma_per_bp2: tuple[float, ...]
    cross_gamma_per_bp2: Mapping[tuple[int, int], float]


@dataclass(frozen=True)
class ModelParameterGreek:
    parameter: str
    base_value: float
    bump: float
    delta: float
    gamma: float
    up_pnl: float
    down_pnl: float


def _effective_rates(curve: DiscountCurve) -> np.ndarray:
    return np.asarray(curve.rates, dtype=float) + float(curve.shift)



def parallel_curve_risk(
    pricer: CurvePricer,
    curve: DiscountCurve,
    bump_bp: float = 1.0,
) -> FactorRisk:
    """Parallel DV01 and curve convexity for an arbitrary curve-priced instrument."""
    if bump_bp <= 0.0:
        raise ValueError("bump_bp must be positive")
    base = float(pricer(curve))
    up = float(pricer(curve.bumped(+bump_bp)))
    down = float(pricer(curve.bumped(-bump_bp)))
    return FactorRisk(
        name="parallel",
        dv01=(up - down) / (2.0 * bump_bp),
        convexity=(up - 2.0 * base + down) / (bump_bp * bump_bp),
    )

def key_rate_weights(node_years: Sequence[float], key_tenors: Sequence[float]) -> np.ndarray:
    """Triangular key-rate shock weights with partition-of-unity property.

    The first/last key absorb the tails. Between adjacent keys, their weights
    interpolate linearly and sum to one. This makes the sum of key-rate DV01s
    approximately equal to the parallel DV01 for small bumps.
    """
    nodes = np.asarray(node_years, dtype=float)
    keys = np.asarray(key_tenors, dtype=float)
    if nodes.ndim != 1 or keys.ndim != 1 or keys.size < 2:
        raise ValueError("node_years and key_tenors must be 1D; at least two keys are required")
    if np.any(np.diff(keys) <= 0.0):
        raise ValueError("key_tenors must be strictly increasing")

    w = np.zeros((keys.size, nodes.size), dtype=float)
    for j, t in enumerate(nodes):
        if t <= keys[0]:
            w[0, j] = 1.0
        elif t >= keys[-1]:
            w[-1, j] = 1.0
        else:
            i = int(np.searchsorted(keys, t) - 1)
            left, right = keys[i], keys[i + 1]
            alpha = (t - left) / (right - left)
            w[i, j] = 1.0 - alpha
            w[i + 1, j] = alpha
    return w


def key_rate_risk(
    pricer: CurvePricer,
    curve: DiscountCurve,
    key_tenors: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
    bump_bp: float = 1.0,
) -> list[BucketRisk]:
    """Central-difference key-rate DV01 and diagonal curve convexity.

    Key-rate DV01s sum to the parallel DV01 because the triangular weights form
    a partition of unity. The reported convexities are diagonal only and do not
    sum to the parallel convexity; see :func:`key_rate_convexity_matrix`.
    """
    if bump_bp <= 0.0:
        raise ValueError("bump_bp must be positive")
    base = float(pricer(curve))
    node_years = np.asarray(curve.days, dtype=float) / 365.0
    weights = key_rate_weights(node_years, key_tenors)
    out: list[BucketRisk] = []
    for tenor, shock_shape in zip(key_tenors, weights, strict=True):
        up = float(pricer(curve.bumped_nodes(shock_shape * bump_bp)))
        down = float(pricer(curve.bumped_nodes(-shock_shape * bump_bp)))
        dv01 = (up - down) / (2.0 * bump_bp)
        convexity = (up - 2.0 * base + down) / (bump_bp * bump_bp)
        out.append(BucketRisk(float(tenor), dv01, convexity))
    return out


@dataclass(frozen=True)
class KeyRateConvexity:
    """Full key-rate second-order structure.

    ``matrix[i, j]`` is d2V/(dk_i dk_j) per bp^2 in key-rate space. Because the
    key-rate weights are a partition of unity, a parallel move is the sum of all
    key-rate moves, so ``total`` (the full quadratic form with unit shocks)
    reconciles with the parallel convexity up to finite-difference error.
    """

    key_tenors: tuple[float, ...]
    matrix: np.ndarray
    total: float


def key_rate_convexity_matrix(
    pricer: CurvePricer,
    curve: DiscountCurve,
    key_tenors: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
    bump_bp: float = 1.0,
) -> KeyRateConvexity:
    """Diagonal and cross key-rate convexities that reconcile with the parallel one."""
    if bump_bp <= 0.0:
        raise ValueError("bump_bp must be positive")
    base = float(pricer(curve))
    node_years = np.asarray(curve.days, dtype=float) / 365.0
    weights = key_rate_weights(node_years, key_tenors)
    n = len(key_tenors)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        up = float(pricer(curve.bumped_nodes(weights[i] * bump_bp)))
        down = float(pricer(curve.bumped_nodes(-weights[i] * bump_bp)))
        matrix[i, i] = (up - 2.0 * base + down) / (bump_bp * bump_bp)

    for i in range(n):
        for j in range(i + 1, n):
            pp = (weights[i] + weights[j]) * bump_bp
            pm = (weights[i] - weights[j]) * bump_bp
            cross = (
                float(pricer(curve.bumped_nodes(pp)))
                - float(pricer(curve.bumped_nodes(pm)))
                - float(pricer(curve.bumped_nodes(-pm)))
                + float(pricer(curve.bumped_nodes(-pp)))
            ) / (4.0 * bump_bp * bump_bp)
            matrix[i, j] = cross
            matrix[j, i] = cross

    total = float(matrix.sum())
    return KeyRateConvexity(tuple(float(t) for t in key_tenors), matrix, total)


def curve_factor_loadings(curve: DiscountCurve) -> dict[str, np.ndarray]:
    """Return smooth level/slope/curvature loadings on the curve nodes.

    Slope and curvature use Legendre-like shapes on log maturity. Each loading
    is normalized to max absolute magnitude one, so ``bump_bp`` has a common
    interpretation across factors. Custom empirical/PCA loadings can instead be
    passed directly to :func:`curve_factor_risk`.
    """
    t = np.asarray(curve.days, dtype=float) / 365.0
    x = np.log(np.maximum(t, 1.0 / 365.0))
    x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
    level = np.ones_like(x)
    slope = x
    curvature = 0.5 * (3.0 * x * x - 1.0)
    return {"level": level, "slope": slope, "curvature": curvature}


def curve_factor_risk(
    pricer: CurvePricer,
    curve: DiscountCurve,
    factors: Mapping[str, Sequence[float]] | None = None,
    bump_bp: float = 1.0,
) -> list[FactorRisk]:
    """DV01/convexity to smooth or user-supplied curve factors.

    User-supplied factor arrays are interpreted as node shock loadings and are
    normalized to max absolute value one. This makes the function directly
    usable with PCA eigenvectors estimated outside the pricing package.
    """
    if bump_bp <= 0.0:
        raise ValueError("bump_bp must be positive")
    raw = factors or curve_factor_loadings(curve)
    base = float(pricer(curve))
    out: list[FactorRisk] = []
    for name, values in raw.items():
        load = np.asarray(values, dtype=float)
        if load.shape != (len(curve.days),) or np.any(~np.isfinite(load)):
            raise ValueError(f"factor {name!r} must have one finite loading per curve node")
        scale = float(np.max(np.abs(load)))
        if scale == 0.0:
            raise ValueError(f"factor {name!r} is identically zero")
        load = load / scale
        up = float(pricer(curve.bumped_nodes(load * bump_bp)))
        down = float(pricer(curve.bumped_nodes(-load * bump_bp)))
        out.append(
            FactorRisk(
                str(name),
                (up - down) / (2.0 * bump_bp),
                (up - 2.0 * base + down) / (bump_bp * bump_bp),
            )
        )
    return out


def bucketed_surface_vega(
    pricer: SurfacePricer,
    surface: VolSurface2D,
    bump_abs: float = 1e-4,
) -> list[VegaBucket]:
    """Finite-difference Vega by expiry/strike surface node.

    ``dv_dvol`` is derivative per 1.00 absolute volatility. ``vega_1vol_point``
    is P&L for a 1 percentage-point volatility move and ``vega_1vol_bp`` for a
    0.01 percentage-point move.
    """
    if bump_abs <= 0.0:
        raise ValueError("bump_abs must be positive")
    out: list[VegaBucket] = []
    for i, expiry in enumerate(surface.expiries):
        for j, strike in enumerate(surface.strikes):
            up = float(pricer(surface.bumped_node(i, j, bump_abs)))
            down = float(pricer(surface.bumped_node(i, j, -bump_abs)))
            derivative = (up - down) / (2.0 * bump_abs)
            out.append(
                VegaBucket(
                    expiry=expiry,
                    strike=strike,
                    dv_dvol=derivative,
                    vega_1vol_point=derivative * 0.01,
                    vega_1vol_bp=derivative * 0.0001,
                )
            )
    return out


def carry_roll_down(
    pricer: DatedCurvePricer,
    value_date: date,
    curve: DiscountCurve,
    horizon_days: int,
    cashflows: float = 0.0,
    funding_rate: float | None = None,
) -> CarryRollResult:
    """Deterministic carry/roll-down decomposition over a horizon.

    The curve is rolled forward using discount-factor ratios implied by today's
    term structure. ``roll_down`` is the change in mark-to-market on that
    deterministic forward curve. ``financing_carry`` is cash received during
    the horizon minus the funding cost of the opening PV. This is an explicit
    convention, not a unique economic decomposition.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    base = float(pricer(value_date, curve))
    future_date = value_date + timedelta(days=horizon_days)
    rolled = curve.roll_forward(horizon_days)
    future = float(pricer(future_date, rolled))
    rate = float(funding_rate if funding_rate is not None else curve.zero(horizon_days))
    financing_cost = base * (math.exp(rate * horizon_days / 365.0) - 1.0)
    financing_carry = float(cashflows) - financing_cost
    roll = future - base
    total = roll + financing_carry
    return CarryRollResult(
        horizon_days=horizon_days,
        base_pv=base,
        horizon_pv=future,
        cashflows=float(cashflows),
        financing_carry=financing_carry,
        roll_down=roll,
        total_pnl=total,
        funding_rate=rate,
    )


def _node_greeks(
    pricer: CurvePricer,
    curve: DiscountCurve,
    bump_bp: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    base = float(pricer(curve))
    n = len(curve.days)
    delta = np.empty(n, dtype=float)
    gamma = np.empty(n, dtype=float)
    for i in range(n):
        shock = np.zeros(n, dtype=float)
        shock[i] = bump_bp
        up = float(pricer(curve.bumped_nodes(shock)))
        down = float(pricer(curve.bumped_nodes(-shock)))
        delta[i] = (up - down) / (2.0 * bump_bp)
        gamma[i] = (up - 2.0 * base + down) / (bump_bp * bump_bp)
    return base, delta, gamma


def explain_curve_pnl(
    pricer: CurvePricer,
    base_curve: DiscountCurve,
    shocked_curve: DiscountCurve,
    bump_bp: float = 1.0,
    include_cross_gamma: bool = True,
) -> PnLExplainResult:
    """Explain full-revaluation curve P&L with first/second-order node Greeks."""
    if base_curve.days != shocked_curve.days:
        raise ValueError("base and shocked curves must share the same node grid")
    if bump_bp <= 0.0:
        raise ValueError("bump_bp must be positive")
    base, delta, gamma = _node_greeks(pricer, base_curve, bump_bp)
    shocked = float(pricer(shocked_curve))
    shock_bp = (_effective_rates(shocked_curve) - _effective_rates(base_curve)) * 10000.0
    first = float(delta @ shock_bp)
    second = float(0.5 * np.sum(gamma * shock_bp * shock_bp))
    cross: dict[tuple[int, int], float] = {}

    if include_cross_gamma:
        n = len(base_curve.days)
        for i in range(n):
            for j in range(i + 1, n):
                pp = np.zeros(n)
                pp[i] = bump_bp
                pp[j] = bump_bp
                pm = np.zeros(n)
                pm[i] = bump_bp
                pm[j] = -bump_bp
                mp = np.zeros(n)
                mp[i] = -bump_bp
                mp[j] = bump_bp
                mm = np.zeros(n)
                mm[i] = -bump_bp
                mm[j] = -bump_bp
                cross_gamma = (
                    float(pricer(base_curve.bumped_nodes(pp)))
                    - float(pricer(base_curve.bumped_nodes(pm)))
                    - float(pricer(base_curve.bumped_nodes(mp)))
                    + float(pricer(base_curve.bumped_nodes(mm)))
                ) / (4.0 * bump_bp * bump_bp)
                cross[(i, j)] = cross_gamma
                second += cross_gamma * shock_bp[i] * shock_bp[j]

    actual = shocked - base
    return PnLExplainResult(
        base_pv=base,
        shocked_pv=shocked,
        actual_pnl=actual,
        first_order=first,
        second_order=second,
        residual=actual - first - second,
        node_delta_per_bp=tuple(map(float, delta)),
        node_gamma_per_bp2=tuple(map(float, gamma)),
        cross_gamma_per_bp2=cross,
    )


def short_rate_model_curve(
    model: str,
    params: ShortRateParams,
    value_date: date,
    days: Sequence[int],
) -> DiscountCurve:
    """Build a :class:`DiscountCurve` from Vasicek/CIR Q parameters."""
    t = np.asarray(days, dtype=float) / 365.0
    name = model.strip().lower()
    if name in {"vasicek", "ou"}:
        z = vasicek_zero_rate(t, params.r0, params.kappa, params.theta, params.sigma)
    elif name == "cir":
        z = cir_zero_rate(t, params.r0, params.kappa, params.theta, params.sigma)
    else:
        raise ValueError("model must be 'vasicek'/'ou' or 'cir'")
    return DiscountCurve(list(map(int, days)), np.asarray(z, dtype=float).tolist(), value_date)


def short_rate_parameter_greeks(
    pricer: CurvePricer,
    model: str,
    params: ShortRateParams,
    value_date: date,
    curve_days: Sequence[int],
    bumps: Mapping[str, float] | None = None,
) -> list[ModelParameterGreek]:
    """Model-risk sensitivities to Q short-rate parameters.

    These are *parameter* Greeks with the other model parameters held fixed; they
    are not hedgeable market Greeks and do not represent a recalibration of the
    model after a market move.
    """
    bump_map = dict(bumps or {"r0": 1e-4, "kappa": 1e-3, "theta": 1e-4, "sigma": 1e-4})
    names = ("r0", "kappa", "theta", "sigma")
    if set(bump_map) != set(names) or any(x <= 0.0 for x in bump_map.values()):
        raise ValueError("bumps must provide positive r0/kappa/theta/sigma bumps")

    base_curve = short_rate_model_curve(model, params, value_date, curve_days)
    base_pv = float(pricer(base_curve))
    out: list[ModelParameterGreek] = []
    values = {name: getattr(params, name) for name in names}

    for name in names:
        h = float(bump_map[name])
        up_values = dict(values)
        up_values[name] += h
        down_values = dict(values)
        down_values[name] -= h
        if model.strip().lower() == "cir" and name in {"r0", "kappa", "theta", "sigma"}:
            if down_values[name] <= 0.0:
                h = min(h, 0.49 * values[name])
                up_values = dict(values)
                up_values[name] += h
                down_values = dict(values)
                down_values[name] -= h
        up_params = ShortRateParams(**up_values)
        down_params = ShortRateParams(**down_values)
        up_pv = float(pricer(short_rate_model_curve(model, up_params, value_date, curve_days)))
        down_pv = float(pricer(short_rate_model_curve(model, down_params, value_date, curve_days)))
        out.append(
            ModelParameterGreek(
                parameter=name,
                base_value=float(values[name]),
                bump=h,
                delta=(up_pv - down_pv) / (2.0 * h),
                gamma=(up_pv - 2.0 * base_pv + down_pv) / (h * h),
                up_pnl=up_pv - base_pv,
                down_pnl=down_pv - base_pv,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Multi-curve risk
# ---------------------------------------------------------------------------
CurveSetPricer = Callable[[CurveSet], float]


@dataclass(frozen=True)
class MultiCurveRisk:
    """Decomposition of curve delta into discounting and projection components.

    ``total_dv01`` moves every curve together (the classic parallel move).
    ``discount_dv01`` holds projection fixed, ``projection_dv01`` holds
    discounting fixed. For a par swap the two largely offset, which is precisely
    why a single-curve DV01 hides the basis exposure.
    """

    total_dv01: float
    discount_dv01: float
    projection_dv01: dict[str, float]
    basis_dv01: float

    @property
    def cross_check(self) -> float:
        """Residual between the total and the sum of its components."""
        return self.total_dv01 - (self.discount_dv01 + self.basis_dv01)


def multi_curve_risk(
    pricer: CurveSetPricer,
    curves: CurveSet,
    bump_bp: float = 1.0,
) -> MultiCurveRisk:
    """Central-difference discount, projection and basis DV01s."""
    if bump_bp <= 0.0:
        raise ValueError("bump_bp must be positive")

    def central(up: CurveSet, down: CurveSet) -> float:
        return (float(pricer(up)) - float(pricer(down))) / (2.0 * bump_bp)

    total = central(curves.bumped(bump_bp), curves.bumped(-bump_bp))
    discount = central(curves.bumped_discount(bump_bp), curves.bumped_discount(-bump_bp))
    basis = central(curves.bumped_projection(bump_bp), curves.bumped_projection(-bump_bp))

    per_index: dict[str, float] = {}
    for name in curves.projection:
        per_index[name] = central(
            curves.bumped_projection(bump_bp, name),
            curves.bumped_projection(-bump_bp, name),
        )
    return MultiCurveRisk(total, discount, per_index, basis)
