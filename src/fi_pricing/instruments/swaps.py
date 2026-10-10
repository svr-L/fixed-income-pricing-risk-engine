"""Vanilla fixed/floating interest-rate swap cashflows and valuation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from ..conventions import ScheduleSpec, TargetCalendar, remaining_periods, year_fraction
from ..curves import CurveSet, as_curve_set
from ..market_data import CURVE_DAYS, CURVE_ZERO_RATES
from ..rfr import RfrConvention, accrued_compounded_rate, compounded_rate
from ..term_structure import DiscountCurve


@dataclass
class Leg:
    notional: float
    effective: date
    maturity: date
    frequency: int
    basis: str
    is_fixed: bool
    coupon: float = 0.0
    spread: float = 0.0
    current_fixing: float | None = None
    past_fixings: dict[date, float] | None = None
    amortisation: Mapping[date, float] | None = None
    schedule_spec: ScheduleSpec | None = None
    index: str | None = None
    rfr: RfrConvention | None = None
    observed_compounding: tuple[date, float] | None = None

    def __post_init__(self) -> None:
        if self.notional <= 0.0:
            raise ValueError("notional must be positive")
        if self.effective >= self.maturity:
            raise ValueError("effective must be before maturity")
        if self.frequency <= 0 or 12 % self.frequency:
            raise ValueError("frequency must be a positive divisor of 12")
        if self.amortisation is not None:
            if not self.amortisation:
                raise ValueError("amortisation schedule must not be empty")
            if any(value < 0.0 for value in self.amortisation.values()):
                raise ValueError("amortising notionals must be non-negative")
        if self.is_fixed:
            if self.current_fixing is not None:
                raise ValueError("current_fixing only applies to floating legs")
            if self.past_fixings:
                raise ValueError("past_fixings only applies to floating legs")
            if self.rfr is not None:
                raise ValueError("rfr conventions only apply to floating legs")
            if self.observed_compounding is not None:
                raise ValueError("observed_compounding only applies to floating legs")
        elif self.rfr is not None and self.current_fixing is not None:
            raise ValueError(
                "compounded RFR legs use past_fixings/observed_compounding, not current_fixing"
            )

    @property
    def is_compounded_rfr(self) -> bool:
        return not self.is_fixed and self.rfr is not None


@dataclass
class CashFlow:
    index: int
    start: date
    end: date
    payment_date: date
    fixing_date: date
    days: int
    af: float
    rate: float
    notional: float
    payment: float
    df: float
    pv: float


@dataclass
class LegResult:
    pv: float
    annuity: float
    accrued: float
    clean: float
    flows: list[CashFlow] = field(default_factory=list)


@dataclass
class IRSResult:
    fixed: LegResult
    floating: LegResult
    par_coupon: float
    receiver_pv: float
    payer_pv: float
    dv01: float
    curve_convexity: float


def _leg_schedule_spec(leg: Leg) -> ScheduleSpec:
    spec = leg.schedule_spec or ScheduleSpec(frequency=leg.frequency)
    if spec.frequency != leg.frequency:
        raise ValueError("leg.frequency and schedule_spec.frequency must match")
    return spec


def _period_notional(leg: Leg, start: date) -> float:
    if leg.amortisation is None:
        return float(leg.notional)
    return float(leg.amortisation.get(start, leg.notional))


def _validate_fixing_inputs(leg: Leg, periods, value_date: date) -> tuple[list, list, dict[date, float]]:
    current = [p for p in periods if p.start <= value_date < p.end]
    already_fixed_unpaid = [
        p for p in periods if p.fixing_date <= value_date and not (p.start <= value_date < p.end)
    ]
    fixings = dict(leg.past_fixings or {})

    if leg.is_compounded_rfr:
        if not current and leg.observed_compounding is not None:
            raise ValueError("observed_compounding supplied but no current accrual period exists")
        return current, already_fixed_unpaid, fixings

    if leg.is_fixed:
        if fixings:
            raise ValueError("past_fixings only applies to floating legs")
        return current, already_fixed_unpaid, fixings

    if current and leg.current_fixing is None:
        raise ValueError("current floating accrual period requires current_fixing")
    if not current and leg.current_fixing is not None:
        raise ValueError("current_fixing supplied but no current floating period exists")

    expected_dates = {p.fixing_date for p in already_fixed_unpaid}
    missing = sorted(expected_dates.difference(fixings))
    if missing:
        dates = ", ".join(d.isoformat() for d in missing)
        raise ValueError(f"already-fixed but unpaid floating periods require past_fixings for {dates}")
    extra = sorted(set(fixings).difference(expected_dates))
    if extra:
        dates = ", ".join(d.isoformat() for d in extra)
        raise ValueError(
            "past_fixings supplied for dates with no corresponding unpaid period: " + dates
        )
    return current, already_fixed_unpaid, fixings


def price_leg(
    leg: Leg,
    value_date: date,
    curve: DiscountCurve | CurveSet,
    cal: TargetCalendar,
) -> LegResult:
    curves = as_curve_set(curve)
    discount_curve = curves.discount
    projection_curve = curves.project(leg.index)
    spec = _leg_schedule_spec(leg)
    periods = remaining_periods(
        leg.effective,
        leg.maturity,
        leg.frequency,
        value_date,
        cal,
        spec,
    )
    current, _, historical_fixings = _validate_fixing_inputs(leg, periods, value_date)
    current_ids = {p.index for p in current}

    amortisation = dict(leg.amortisation or {})
    valid_starts = {p.start for p in periods}
    unmatched = sorted(set(amortisation).difference(valid_starts))
    if unmatched:
        raise ValueError(
            "amortisation entries do not match any remaining accrual start: "
            + ", ".join(d.isoformat() for d in unmatched)
        )

    flows: list[CashFlow] = []
    pv = 0.0
    annuity = 0.0
    accrued = 0.0

    for period in periods:
        accrual = year_fraction(period.start, period.end, leg.basis)
        df = discount_curve.df(period.payment_date, value_date)
        notional = _period_notional(leg, period.start)
        is_current = period.index in current_ids

        if leg.is_fixed:
            coupon_rate = float(leg.coupon)
            cash_rate = coupon_rate
        elif leg.is_compounded_rfr:
            coupon_rate = compounded_rate(
                period,
                leg.rfr,
                projection_curve,
                value_date,
                cal,
                past_fixings=historical_fixings,
                observed_compounding=leg.observed_compounding if is_current else None,
            )
            cash_rate = coupon_rate + leg.spread
        elif is_current:
            coupon_rate = float(leg.current_fixing)
            cash_rate = coupon_rate + leg.spread
        elif period.fixing_date in historical_fixings:
            coupon_rate = float(historical_fixings[period.fixing_date])
            cash_rate = coupon_rate + leg.spread
        else:
            p0 = projection_curve.df(period.start, value_date)
            p1 = projection_curve.df(period.end, value_date)
            coupon_rate = (p0 / p1 - 1.0) / accrual
            cash_rate = coupon_rate + leg.spread

        payment = cash_rate * accrual * notional
        flow_pv = payment * df
        pv += flow_pv
        annuity += accrual * df * notional
        flows.append(
            CashFlow(
                index=period.index,
                start=period.start,
                end=period.end,
                payment_date=period.payment_date,
                fixing_date=period.fixing_date,
                days=(period.end - period.start).days,
                af=accrual,
                rate=coupon_rate,
                notional=notional,
                payment=payment,
                df=df,
                pv=flow_pv,
            )
        )

        if is_current:
            elapsed = year_fraction(period.start, value_date, leg.basis)
            if leg.is_fixed:
                accrued_rate = leg.coupon
            elif leg.is_compounded_rfr:
                accrued_rate = accrued_compounded_rate(
                    period,
                    leg.rfr,
                    projection_curve,
                    value_date,
                    cal,
                    past_fixings=historical_fixings,
                    observed_compounding=leg.observed_compounding,
                ) + leg.spread
            else:
                accrued_rate = coupon_rate + leg.spread
            accrued = accrued_rate * notional * elapsed

    return LegResult(pv=pv, annuity=annuity, accrued=accrued, clean=pv - accrued, flows=flows)


def price_irs(
    trade_date: date,
    value_date: date,
    fixed: Leg,
    floating: Leg,
    curve: DiscountCurve | CurveSet | None = None,
    cal: TargetCalendar | None = None,
) -> IRSResult:
    if trade_date > value_date:
        raise ValueError("trade_date cannot be after value_date")
    if not fixed.is_fixed or floating.is_fixed:
        raise ValueError("fixed leg must be fixed and floating leg must be floating")
    if fixed.notional != floating.notional:
        raise ValueError("vanilla IRS legs must have the same notional")

    calendar_obj = cal or TargetCalendar()
    raw_curve = curve or DiscountCurve(CURVE_DAYS, CURVE_ZERO_RATES, value_date)
    curves = as_curve_set(raw_curve)

    fixed_result = price_leg(fixed, value_date, curves, calendar_obj)
    float_result = price_leg(floating, value_date, curves, calendar_obj)
    if fixed_result.annuity <= 0.0:
        raise ValueError("fixed-leg annuity must be positive")

    par_rate = float_result.pv / fixed_result.annuity
    receiver = fixed_result.pv - float_result.pv
    payer = -receiver

    def payer_value(curve_set: CurveSet) -> float:
        fixed_pv = price_leg(fixed, value_date, curve_set, calendar_obj).pv
        float_pv = price_leg(floating, value_date, curve_set, calendar_obj).pv
        return float_pv - fixed_pv

    up = payer_value(curves.bumped(1.0))
    down = payer_value(curves.bumped(-1.0))
    dv01 = 0.5 * (up - down)
    convexity = up - 2.0 * payer + down
    return IRSResult(
        fixed=fixed_result,
        floating=float_result,
        par_coupon=par_rate,
        receiver_pv=receiver,
        payer_pv=payer,
        dv01=dv01,
        curve_convexity=convexity,
    )
