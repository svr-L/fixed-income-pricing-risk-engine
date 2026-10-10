"""Cap/floor valuation as strips of caplets/floorlets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from ..conventions import ScheduleSpec, TargetCalendar, remaining_periods, year_fraction
from ..curves import CurveSet, as_curve_set
from ..market_data import (
    CURVE_DAYS,
    CURVE_ZERO_RATES,
    VOL_CUBE,
    VOL_CUBE_EXPIRIES,
    VOL_CUBE_STRIKES,
)
from ..options import forward_option_greeks, forward_option_premium
from ..rfr import RfrConvention, backward_looking_variance_ratio, compounded_rate
from ..term_structure import DiscountCurve
from ..volatility import VolSurface2D


def default_vol_surface() -> VolSurface2D:
    """Synthetic cap-vol surface bundled for examples and tests."""
    return VolSurface2D(
        VOL_CUBE_EXPIRIES,
        VOL_CUBE_STRIKES,
        [[float(x) / 100.0 for x in row] for row in VOL_CUBE],
    )


def vol_cube_bilinear(maturity_years: float, strike: float) -> float:
    return default_vol_surface().vol(float(maturity_years), float(strike))


@dataclass
class Caplet:
    index: int
    start: date
    end: date
    payment_date: date
    af: float
    ttm: float
    forward: float
    strike: float
    vol: float
    df: float
    intrinsic: float
    premium: float
    delta: float
    gamma: float
    vega: float
    theta: float
    option_theta: float
    discount_theta: float
    vanna: float
    volga: float


@dataclass
class CapResult:
    style: str
    npv: float
    vol_used: float
    caplets: list[Caplet] = field(default_factory=list)
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    option_theta: float = 0.0
    discount_theta: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    vol_mode: str = "flat-cap-vol"
    vol_model: str = "black"
    shift: float = 0.0


def _vol_model_key(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "black": "black",
        "black76": "black",
        "lognormal": "black",
        "shifted_black": "shifted_black",
        "shifted_lognormal": "shifted_black",
        "bachelier": "bachelier",
        "normal": "bachelier",
    }
    if key not in aliases:
        raise ValueError("vol_model must be black, shifted_black or bachelier")
    return aliases[key]


def price_cap(
    value_date: date,
    effective: date,
    maturity: date,
    notional: float,
    strike: float,
    frequency: int,
    first_fixing: float | None = None,
    style: str = "CAP",
    basis: str = "ACT/360",
    flat_vol: float | None = None,
    vol_surface: VolSurface2D | None = None,
    caplet_vol_surface: VolSurface2D | None = None,
    curve: DiscountCurve | CurveSet | None = None,
    cal: TargetCalendar | None = None,
    schedule_spec: ScheduleSpec | None = None,
    vol_model: str = "black",
    shift: float = 0.0,
    index: str | None = None,
    rfr: RfrConvention | None = None,
    past_fixings: dict[date, float] | None = None,
    observed_compounding: tuple[date, float] | None = None,
    amortisation: Mapping[date, float] | None = None,
) -> CapResult:
    """Price a cap or floor under flat-cap or stripped caplet volatilities."""
    side = style.strip().upper()
    if side not in {"CAP", "FLOOR"}:
        raise ValueError("style must be 'CAP' or 'FLOOR'")
    if notional <= 0.0:
        raise ValueError("notional must be positive")
    if effective >= maturity:
        raise ValueError("effective must be before maturity")
    if frequency <= 0 or 12 % frequency:
        raise ValueError("frequency must be a positive divisor of 12")
    if flat_vol is not None and flat_vol < 0.0:
        raise ValueError("flat_vol must be non-negative")
    if flat_vol is not None and caplet_vol_surface is not None:
        raise ValueError("flat_vol and caplet_vol_surface are mutually exclusive")
    model = _vol_model_key(vol_model)
    if model != "black" and flat_vol is None and vol_surface is None and caplet_vol_surface is None:
        raise ValueError("non-Black models require an explicit volatility input/surface")

    calendar_obj = cal or TargetCalendar()
    raw_curve = curve or DiscountCurve(CURVE_DAYS, CURVE_ZERO_RATES, value_date)
    curves = as_curve_set(raw_curve)
    discount_curve = curves.discount
    projection_curve = curves.project(index)
    spec = schedule_spec or ScheduleSpec(frequency=frequency)
    if spec.frequency != frequency:
        raise ValueError("frequency and schedule_spec.frequency must match")

    phi = 1 if side == "CAP" else -1
    cap_surface = vol_surface or default_vol_surface()
    maturity_t = max((maturity - value_date).days / 365.0, 0.0)
    representative_vol = float(flat_vol) if flat_vol is not None else cap_surface.vol(maturity_t, strike)

    periods = remaining_periods(effective, maturity, frequency, value_date, calendar_obj, spec)
    current_periods = [p for p in periods if p.start <= value_date < p.end]
    if rfr is None:
        if current_periods and first_fixing is None:
            raise ValueError("current cap/floor period requires first_fixing")
        if not current_periods and first_fixing is not None:
            raise ValueError("first_fixing supplied but no current cap/floor period exists")
        if past_fixings or observed_compounding:
            raise ValueError("past_fixings and observed_compounding require an rfr convention")
    elif first_fixing is not None:
        raise ValueError(
            "first_fixing does not apply to a compounded RFR cap; use past_fixings/observed_compounding"
        )

    amort = dict(amortisation or {})
    valid_starts = {p.start for p in periods}
    unknown = sorted(set(amort).difference(valid_starts))
    if unknown:
        raise ValueError(
            "amortisation entries do not match any remaining accrual start: "
            + ", ".join(d.isoformat() for d in unknown)
        )

    totals = {name: 0.0 for name in (
        "delta", "gamma", "vega", "theta", "option_theta", "discount_theta", "vanna", "volga"
    )}
    caplets: list[Caplet] = []
    npv = 0.0

    for period in periods:
        accrual = year_fraction(period.start, period.end, basis)
        discount = discount_curve.df(period.payment_date, value_date)
        period_notional = float(amort.get(period.start, notional))
        is_current = period.start <= value_date < period.end

        if rfr is None:
            option_time = max((period.fixing_date - value_date).days / 365.0, 0.0)
        else:
            option_time = backward_looking_variance_ratio(value_date, period)

        if flat_vol is not None:
            vol = float(flat_vol)
        elif caplet_vol_surface is not None and option_time > 0.0:
            vol = caplet_vol_surface.vol(option_time, strike)
        else:
            vol = representative_vol

        if rfr is not None:
            forward = compounded_rate(
                period,
                rfr,
                projection_curve,
                value_date,
                calendar_obj,
                past_fixings=past_fixings,
                observed_compounding=observed_compounding,
            )
        elif is_current:
            forward = float(first_fixing)
        else:
            p0 = projection_curve.df(period.start, value_date)
            p1 = projection_curve.df(period.end, value_date)
            forward = (p0 / p1 - 1.0) / accrual

        intrinsic = max(phi * (forward - strike), 0.0) * accrual * period_notional
        unit_value = forward_option_premium(
            forward,
            strike,
            option_time,
            vol,
            phi,
            model=model,
            shift=shift,
        )
        premium = unit_value * accrual * period_notional * discount
        local = forward_option_greeks(
            forward,
            strike,
            option_time,
            vol,
            phi,
            model=model,
            shift=shift,
        )
        scale = accrual * period_notional * discount
        delta = local["delta"] * scale
        gamma = local["gamma"] * scale
        vega = local["vega"] * scale
        vanna = local["vanna"] * scale
        volga = local["volga"] * scale
        option_theta = local["theta"] * scale
        tenor_days = (period.payment_date - value_date).days
        discount_theta = premium * discount_curve.instantaneous_forward(tenor_days)
        theta = option_theta + discount_theta

        values = {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "option_theta": option_theta,
            "discount_theta": discount_theta,
            "vanna": vanna,
            "volga": volga,
        }
        for key, value in values.items():
            totals[key] += value
        npv += premium
        caplets.append(
            Caplet(
                index=period.index,
                start=period.start,
                end=period.end,
                payment_date=period.payment_date,
                af=accrual,
                ttm=option_time,
                forward=forward,
                strike=strike,
                vol=vol,
                df=discount,
                intrinsic=intrinsic,
                premium=premium,
                delta=delta,
                gamma=gamma,
                vega=vega,
                theta=theta,
                option_theta=option_theta,
                discount_theta=discount_theta,
                vanna=vanna,
                volga=volga,
            )
        )

    if flat_vol is not None:
        vol_mode = "flat-input"
    elif caplet_vol_surface is not None:
        vol_mode = "caplet-vol-surface"
    else:
        vol_mode = "flat-cap-vol"

    return CapResult(
        style=side,
        npv=npv,
        vol_used=representative_vol,
        caplets=caplets,
        vol_mode=vol_mode,
        vol_model=model,
        shift=shift,
        **totals,
    )
