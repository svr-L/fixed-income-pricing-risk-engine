"""European swaption valuation on a forward swap rate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..conventions import ScheduleSpec, TargetCalendar
from ..curves import CurveSet, as_curve_set
from ..market_data import CURVE_DAYS, CURVE_ZERO_RATES
from ..options import forward_option_greeks, forward_option_premium
from ..term_structure import DiscountCurve
from .swaps import Leg, price_leg


def cash_settled_annuity(rate: float, periods: int, frequency: int) -> float:
    """IRR-style annuity evaluated at a deterministic rate."""
    if periods <= 0:
        raise ValueError("periods must be positive")
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    q = float(frequency)
    if abs(rate) < 1e-14:
        return periods / q
    base = 1.0 + rate / q
    if base <= 0.0:
        raise ValueError("IRR annuity requires 1 + rate/frequency > 0")
    return sum(base ** (-i) for i in range(1, periods + 1)) / q


def discount_to_expiry(curve: DiscountCurve, expiry: date, value_date: date) -> float:
    return curve.df(expiry, value_date)


@dataclass
class SwaptionResult:
    style: str
    forward_rate: float
    strike: float
    expiry_years: float
    annuity: float
    vol: float
    phi: int
    d1: float
    d2: float
    forward_premium: float
    premium: float
    delta: float
    vega: float
    gamma: float
    theta: float
    vanna: float
    volga: float
    vol_model: str = "black"
    vol_shift: float = 0.0
    settlement: str = "physical"
    curve_annuity: float = 0.0
    settlement_method: str = "physical_delivery"


def _normalise_vol_model(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key in {"black", "black76", "lognormal"}:
        return "black"
    if key in {"shifted_black", "shifted_lognormal"}:
        return "shifted_black"
    if key in {"bachelier", "normal"}:
        return "bachelier"
    raise ValueError("vol_model must be black, shifted_black or bachelier")


def price_swaption(
    value_date: date,
    expiry: date,
    swap_start: date,
    swap_end: date,
    strike: float,
    vol: float,
    style: str,
    notional: float = 1_000_000,
    fixed_freq: int = 1,
    fixed_basis: str = "30/360",
    float_freq: int = 2,
    float_basis: str = "ACT/360",
    shift_bp: float = 0.0,
    curve: DiscountCurve | CurveSet | None = None,
    cal: TargetCalendar | None = None,
    fixed_schedule_spec: ScheduleSpec | None = None,
    float_schedule_spec: ScheduleSpec | None = None,
    vol_model: str = "black",
    vol_shift: float = 0.0,
    index: str | None = None,
    settlement: str = "physical",
) -> SwaptionResult:
    side = style.strip().lower()
    if side not in {"payer", "receiver"}:
        raise ValueError("style must be 'Payer' or 'Receiver'")
    if value_date >= expiry:
        raise ValueError("expiry must be after value_date")
    if expiry > swap_start:
        raise ValueError("expiry cannot be after underlying swap_start")
    if swap_start >= swap_end:
        raise ValueError("swap_start must be before swap_end")
    if notional <= 0.0:
        raise ValueError("notional must be positive")
    if vol < 0.0:
        raise ValueError("volatility must be non-negative")
    model = _normalise_vol_model(vol_model)
    settle = settlement.strip().lower()
    if settle not in {"physical", "cash"}:
        raise ValueError("settlement must be 'physical' or 'cash'")

    calendar_obj = cal or TargetCalendar()
    raw_curve = curve or DiscountCurve(CURVE_DAYS, CURVE_ZERO_RATES, value_date)
    curves = as_curve_set(raw_curve)
    if shift_bp:
        curves = curves.bumped(shift_bp)

    fixed_leg = Leg(
        notional=notional,
        effective=swap_start,
        maturity=swap_end,
        frequency=fixed_freq,
        basis=fixed_basis,
        is_fixed=True,
        coupon=strike,
        schedule_spec=fixed_schedule_spec,
    )
    floating_leg = Leg(
        notional=notional,
        effective=swap_start,
        maturity=swap_end,
        frequency=float_freq,
        basis=float_basis,
        is_fixed=False,
        schedule_spec=float_schedule_spec,
        index=index,
    )
    fixed_result = price_leg(fixed_leg, value_date, curves, calendar_obj)
    float_result = price_leg(floating_leg, value_date, curves, calendar_obj)
    curve_annuity = fixed_result.annuity
    if curve_annuity <= 0.0:
        raise ValueError("underlying fixed-leg annuity must be positive")
    forward = float_result.pv / curve_annuity

    annuity = curve_annuity
    settlement_method = "physical_delivery"
    if settle == "cash":
        annuity = (
            cash_settled_annuity(forward, len(fixed_result.flows), fixed_freq)
            * curves.discount.df(expiry, value_date)
            * notional
        )
        if annuity <= 0.0:
            raise ValueError("cash-settled annuity must be positive")
        settlement_method = "frozen_forward_irr_annuity"

    T = (expiry - value_date).days / 365.0
    phi = 1 if side == "payer" else -1
    unit_value = forward_option_premium(
        forward,
        strike,
        T,
        vol,
        phi,
        model=model,
        shift=vol_shift,
    )
    greeks = forward_option_greeks(
        forward,
        strike,
        T,
        vol,
        phi,
        model=model,
        shift=vol_shift,
    )

    return SwaptionResult(
        style="Payer" if phi == 1 else "Receiver",
        forward_rate=forward,
        strike=strike,
        expiry_years=T,
        annuity=annuity,
        vol=vol,
        phi=phi,
        d1=greeks["d1"],
        d2=greeks["d2"],
        forward_premium=unit_value,
        premium=unit_value * annuity,
        delta=greeks["delta"] * annuity,
        vega=greeks["vega"] * annuity,
        gamma=greeks["gamma"] * annuity,
        theta=greeks["theta"] * annuity,
        vanna=greeks["vanna"] * annuity,
        volga=greeks["volga"] * annuity,
        vol_model=model,
        vol_shift=vol_shift,
        settlement=settle,
        curve_annuity=curve_annuity,
        settlement_method=settlement_method,
    )
