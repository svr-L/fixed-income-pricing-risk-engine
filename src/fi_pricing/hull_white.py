"""Hull-White one-factor short-rate model, fitted to an initial discount curve.

The model is

    dr(t) = (theta(t) - a r(t)) dt + sigma dW(t)

with ``theta(t)`` implied by the initial term structure, so the model reproduces
today's discount factors *exactly* by construction. That is the whole point of
using it for market-consistent curve fitting: constant-parameter Vasicek and
CIR are valid pricing models, but they do not in general reproduce an arbitrary
observed initial term structure exactly unless extended with a deterministic shift.

``theta(t)`` never needs to be evaluated: the affine reconstruction

    P(t, T) = A(t, T) exp(-B(t, T) r(t))

absorbs it through the market discount factors and the market instantaneous
forward, which is what the closed forms below use.

Scope: this is a single-curve model. In a multi-curve setting it is applied to
the discount curve, with the projection basis held deterministic - the standard
practical assumption, but an assumption. Pass the curve you intend to model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import brentq, least_squares

from . import core as fc
from .curves import CurveSet, as_curve_set

_SQRT_EPS = 1e-12


@dataclass(frozen=True)
class HullWhiteParams:
    """Mean reversion ``a`` and short-rate volatility ``sigma``."""

    a: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        if abs(self.a) < 1e-10:
            raise ValueError(
                "mean reversion too close to zero; use a >= 1e-10 "
                "(the a -> 0 limit is Ho-Lee and is not implemented)"
            )


def b_factor(a: float, tau: float) -> float:
    """B(t, T) = (1 - exp(-a*tau)) / a, the bond-price duration factor."""
    if tau <= 0.0:
        return 0.0
    return (1.0 - math.exp(-a * tau)) / a


def a_factor(
    params: HullWhiteParams, curve: fc.DiscountCurve, t: float, maturity: float
) -> float:
    """A(t, T) from the initial curve.

    A(t, T) = [P(0,T)/P(0,t)] exp( B(t,T) f(0,t) - sigma^2/(4a) (1-e^{-2at}) B(t,T)^2 )
    """
    if maturity < t:
        raise ValueError("maturity must be on or after t")
    df_t = curve.df_days(t * 365.0)
    df_maturity = curve.df_days(maturity * 365.0)
    b = b_factor(params.a, maturity - t)
    fwd = curve.instantaneous_forward(t * 365.0)
    variance = 0.0
    if t > 0.0:
        variance = (
            params.sigma
            * params.sigma
            / (4.0 * params.a)
            * (1.0 - math.exp(-2.0 * params.a * t))
            * b
            * b
        )
    return (df_maturity / df_t) * math.exp(b * fwd - variance)


def zero_bond_price(
    params: HullWhiteParams,
    curve: fc.DiscountCurve,
    t: float,
    maturity: float,
    short_rate: float,
) -> float:
    """P(t, T) given the short rate at t."""
    return a_factor(params, curve, t, maturity) * math.exp(
        -b_factor(params.a, maturity - t) * short_rate
    )


def bond_option_vol(params: HullWhiteParams, expiry: float, bond_maturity: float) -> float:
    """Integrated volatility of the T-forward bond price P(T, S)."""
    if expiry <= 0.0 or params.sigma == 0.0:
        return 0.0
    b = b_factor(params.a, bond_maturity - expiry)
    return (
        params.sigma
        * math.sqrt((1.0 - math.exp(-2.0 * params.a * expiry)) / (2.0 * params.a))
        * b
    )


def zero_bond_option(
    params: HullWhiteParams,
    curve: fc.DiscountCurve,
    expiry: float,
    bond_maturity: float,
    strike: float,
    phi: int,
) -> float:
    """European option on a zero-coupon bond. ``phi`` is +1 call, -1 put.

    Closed form: the T-forward price of P(T,S) is lognormal under the T-forward
    measure, so this is Black with the integrated bond volatility.
    """
    if phi not in (1, -1):
        raise ValueError("phi must be +1 (call) or -1 (put)")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if bond_maturity < expiry:
        raise ValueError("bond maturity must be on or after option expiry")

    df_expiry = curve.df_days(expiry * 365.0)
    df_bond = curve.df_days(bond_maturity * 365.0)
    vol = bond_option_vol(params, expiry, bond_maturity)

    if vol < _SQRT_EPS:
        intrinsic = phi * (df_bond - strike * df_expiry)
        return max(intrinsic, 0.0)

    h = math.log(df_bond / (df_expiry * strike)) / vol + 0.5 * vol
    n = fc.norm_cdf
    if phi == 1:
        return df_bond * n(h) - strike * df_expiry * n(h - vol)
    return strike * df_expiry * n(-h + vol) - df_bond * n(-h)


def caplet_price(
    params: HullWhiteParams,
    curve: fc.DiscountCurve,
    reset: float,
    payment: float,
    accrual: float,
    strike: float,
    notional: float = 1.0,
    phi: int = 1,
) -> float:
    """Caplet (phi=+1) or floorlet (phi=-1) as an option on a zero-coupon bond.

    A caplet on the rate set at ``reset`` and paid at ``payment`` is
    ``(1 + K*tau)`` puts on P(reset, payment) struck at ``1/(1 + K*tau)``.
    """
    if accrual <= 0.0:
        raise ValueError("accrual must be positive")
    coefficient = 1.0 + strike * accrual
    if coefficient <= 0.0:
        raise ValueError("1 + strike*accrual must be positive")
    bond_strike = 1.0 / coefficient
    # A cap is a put on the bond, a floor is a call.
    return (
        notional
        * coefficient
        * zero_bond_option(params, curve, reset, payment, bond_strike, -phi)
    )


def jamshidian_strikes(
    params: HullWhiteParams,
    curve: fc.DiscountCurve,
    expiry: float,
    payment_times: Sequence[float],
    cashflows: Sequence[float],
) -> tuple[float, tuple[float, ...]]:
    """Decompose a coupon bond option into zero-bond options.

    Returns the critical short rate ``r*`` at which the coupon bond is worth par
    at ``expiry``, and the per-cashflow bond strikes. Valid because in a
    one-factor model every bond price is monotone in the single state variable.
    """
    if len(payment_times) != len(cashflows):
        raise ValueError("payment_times and cashflows must have equal length")
    if not payment_times:
        raise ValueError("at least one cashflow is required")

    def bond_value(short_rate: float) -> float:
        return (
            sum(
                c * zero_bond_price(params, curve, expiry, t, short_rate)
                for t, c in zip(payment_times, cashflows, strict=True)
            )
            - 1.0
        )

    lo, hi = -1.0, 1.0
    f_lo, f_hi = bond_value(lo), bond_value(hi)
    tries = 0
    while f_lo * f_hi > 0.0 and tries < 60:
        lo, hi = lo * 1.5 - 0.05, hi * 1.5 + 0.05
        f_lo, f_hi = bond_value(lo), bond_value(hi)
        tries += 1
    if f_lo * f_hi > 0.0:
        raise ValueError("could not bracket the Jamshidian critical rate")

    r_star = float(brentq(bond_value, lo, hi, xtol=1e-14, rtol=1e-15))
    strikes = tuple(
        zero_bond_price(params, curve, expiry, t, r_star) for t in payment_times
    )
    return r_star, strikes


def swaption_price(
    params: HullWhiteParams,
    curve: fc.DiscountCurve,
    expiry: float,
    payment_times: Sequence[float],
    accruals: Sequence[float],
    strike: float,
    notional: float = 1.0,
    payer: bool = True,
) -> float:
    """European swaption by Jamshidian decomposition.

    A payer swaption is a put on the fixed-leg coupon bond struck at par, which
    decomposes into a portfolio of puts on the individual zero-coupon bonds.
    """
    if len(payment_times) != len(accruals):
        raise ValueError("payment_times and accruals must have equal length")
    if expiry <= 0.0:
        raise ValueError("expiry must be positive")

    cashflows = [strike * tau for tau in accruals]
    cashflows[-1] += 1.0
    _, strikes = jamshidian_strikes(params, curve, expiry, payment_times, cashflows)

    phi = -1 if payer else 1
    total = 0.0
    for t, c, k in zip(payment_times, cashflows, strikes, strict=True):
        total += c * zero_bond_option(params, curve, expiry, t, k, phi)
    return notional * total


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationInstrument:
    """A caplet or European swaption used as a calibration target."""

    expiry: float
    payment_times: tuple[float, ...]
    accruals: tuple[float, ...]
    strike: float
    target_price: float
    weight: float = 1.0
    kind: str = "swaption"
    market_vega: float | None = None

    def model_price(self, params: HullWhiteParams, curve: fc.DiscountCurve) -> float:
        if self.kind == "caplet":
            return caplet_price(
                params,
                curve,
                self.expiry,
                self.payment_times[0],
                self.accruals[0],
                self.strike,
            )
        return swaption_price(
            params, curve, self.expiry, self.payment_times, self.accruals, self.strike
        )


@dataclass(frozen=True)
class CalibrationResult:
    params: HullWhiteParams
    rmse: float
    max_abs_error: float
    errors: tuple[float, ...]
    success: bool
    message: str


def calibrate_hull_white(
    instruments: Sequence[CalibrationInstrument],
    curve: fc.DiscountCurve,
    *,
    initial: HullWhiteParams | None = None,
    a_bounds: tuple[float, float] = (1e-4, 1.0),
    sigma_bounds: tuple[float, float] = (1e-6, 0.20),
    fix_a: float | None = None,
    objective: str = "model_sigma_normalised",
) -> CalibrationResult:
    """Least-squares fit of ``(a, sigma)`` to instrument prices.

    ``price`` uses raw price residuals. ``model_sigma_normalised`` scales
    residuals by sensitivity to the Hull-White model parameter ``sigma`` at the
    starting point; the legacy name ``vega_normalised`` is accepted as an alias.
    This is numerical scaling, not a market Vega.

    ``market_vega_normalised`` divides residuals by a quote-convention Vega
    supplied on each calibration instrument, so residuals are approximately in
    implied-volatility units. Scales are frozen at the start of optimisation.
    """
    mode = objective.strip().lower()
    if mode == "vega_normalised":
        mode = "model_sigma_normalised"
    allowed = {"model_sigma_normalised", "market_vega_normalised", "price"}
    if mode not in allowed:
        raise ValueError(
            "objective must be 'price', 'model_sigma_normalised' or "
            "'market_vega_normalised'"
        )
    if not instruments:
        raise ValueError("at least one calibration instrument is required")
    start = initial or HullWhiteParams(0.05, 0.01)
    weights = np.array([i.weight for i in instruments], dtype=float)
    if np.any(weights <= 0.0):
        raise ValueError("weights must be positive")
    scales = np.ones(len(instruments), dtype=float)
    if mode == "model_sigma_normalised":
        bump = 1e-5
        up = HullWhiteParams(start.a, start.sigma + bump)
        down = HullWhiteParams(start.a, max(start.sigma - bump, 1e-9))
        span = up.sigma - down.sigma
        for k, instrument in enumerate(instruments):
            sensitivity = (
                instrument.model_price(up, curve)
                - instrument.model_price(down, curve)
            ) / span
            scales[k] = max(abs(sensitivity), 1e-8)
    elif mode == "market_vega_normalised":
        vegas = [instrument.market_vega for instrument in instruments]
        if any(v is None or not math.isfinite(v) or v <= 0.0 for v in vegas):
            raise ValueError(
                "market_vega_normalised requires a positive finite market_vega "
                "on every calibration instrument"
            )
        scales = np.array([float(v) for v in vegas], dtype=float)
    target = np.array([i.target_price for i in instruments], dtype=float)
    def residuals(x: np.ndarray) -> np.ndarray:
        a = fix_a if fix_a is not None else float(x[0])
        sigma = float(x[-1])
        try:
            trial = HullWhiteParams(a, sigma)
            model = np.array([i.model_price(trial, curve) for i in instruments], dtype=float)
        except (ValueError, OverflowError):
            return np.full(len(instruments), 1e6)
        return weights * (model - target) / scales
    if fix_a is not None:
        x0 = np.array([start.sigma])
        bounds = ([sigma_bounds[0]], [sigma_bounds[1]])
    else:
        x0 = np.array([start.a, start.sigma])
        bounds = ([a_bounds[0], sigma_bounds[0]], [a_bounds[1], sigma_bounds[1]])
    opt = least_squares(residuals, x0, bounds=bounds, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    a = fix_a if fix_a is not None else float(opt.x[0])
    fitted = HullWhiteParams(a, float(opt.x[-1]))
    errors = tuple(float(i.model_price(fitted, curve) - i.target_price) for i in instruments)
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
    return CalibrationResult(fitted, rmse, max(abs(e) for e in errors), errors, bool(opt.success), str(opt.message))


# ---------------------------------------------------------------------------
# Bridges to the rest of the package
# ---------------------------------------------------------------------------
def year_fractions_from_schedule(
    value_date: date, periods: Sequence[fc.CouponPeriod], basis: str = "30/360"
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Payment times in years and accrual factors, for the swaption helpers."""
    times = tuple((p.payment_date - value_date).days / 365.0 for p in periods)
    accruals = tuple(fc.year_fraction(p.start, p.end, basis) for p in periods)
    return times, accruals


def swaption_from_schedule(
    params: HullWhiteParams,
    value_date: date,
    expiry_date: date,
    swap_start: date,
    swap_end: date,
    strike: float,
    curve: fc.DiscountCurve | CurveSet,
    *,
    notional: float = 1.0,
    fixed_frequency: int = 1,
    basis: str = "30/360",
    payer: bool = True,
    cal: fc.TargetCalendar | None = None,
    schedule_spec: fc.ScheduleSpec | None = None,
) -> float:
    """Price a European swaption from calendar dates rather than year fractions."""
    if expiry_date > swap_start:
        raise ValueError("expiry must be on or before the swap start")
    discount = as_curve_set(curve).discount
    cal = cal or fc.TargetCalendar()
    spec = schedule_spec or fc.ScheduleSpec(frequency=fixed_frequency)
    periods = fc.build_schedule(swap_start, swap_end, spec, cal)
    times, accruals = year_fractions_from_schedule(value_date, periods, basis)
    expiry = (expiry_date - value_date).days / 365.0
    return swaption_price(
        params, discount, expiry, times, accruals, strike, notional, payer
    )


def caplet_instruments_from_surface(
    value_date: date,
    effective: date,
    maturity: date,
    strike: float,
    frequency: int,
    curve: fc.DiscountCurve,
    surface,
    *,
    basis: str = "ACT/360",
    cal: fc.TargetCalendar | None = None,
    schedule_spec: fc.ScheduleSpec | None = None,
    model: str = "black",
    shift: float = 0.0,
) -> tuple[CalibrationInstrument, ...]:
    """Build caplet calibration targets from a stripped caplet-vol surface.

    Each live caplet is priced with its own surface volatility and turned into a
    target, so Hull-White is fitted to the same market the stripper consumed.
    """
    cal = cal or fc.TargetCalendar()
    spec = schedule_spec or fc.ScheduleSpec(frequency=frequency)
    periods = [
        p
        for p in fc.build_schedule(effective, maturity, spec, cal)
        if p.fixing_date > value_date
    ]
    if not periods:
        raise ValueError("no live caplets between effective and maturity")

    out: list[CalibrationInstrument] = []
    for p in periods:
        reset = (p.fixing_date - value_date).days / 365.0
        payment = (p.payment_date - value_date).days / 365.0
        accrual = fc.year_fraction(p.start, p.end, basis)
        df_start = curve.df(p.start, value_date)
        df_end = curve.df(p.end, value_date)
        forward = (df_start / df_end - 1.0) / accrual
        vol = surface.vol(reset, strike)
        premium = fc.forward_option_premium(
            forward, strike, reset, vol, 1, model=model, shift=shift
        )
        discount = curve.df(p.payment_date, value_date)
        target = premium * accrual * discount
        market_vega = (
            fc.forward_option_greeks(
                forward, strike, reset, vol, 1, model=model, shift=shift
            )["vega"]
            * accrual
            * discount
        )
        out.append(
            CalibrationInstrument(
                expiry=reset,
                payment_times=(payment,),
                accruals=(accrual,),
                strike=strike,
                target_price=target,
                kind="caplet",
                market_vega=market_vega,
            )
        )
    return tuple(out)
