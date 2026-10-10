"""Compounded-in-arrears RFR coupons (ESTR, SOFR, SONIA).

An RFR coupon is not a single forward-looking fixing. It is the daily
compounding of an overnight rate over an observation period:

    rate = (prod_i (1 + r_i * d_i / basis_days) - 1) * basis_days / D

where ``d_i`` is the number of calendar days each overnight rate applies for and
``D`` is the total. The observation period is usually shifted relative to the
accrual period so the payment amount is known a few days before it settles.

Two shift mechanics are supported:

``lookback``
    Rates are observed ``k`` business days before each accrual date, but each
    observed rate is weighted by the *accrual* period's day counts.
``observation_shift``
    The whole observation window is moved back ``k`` business days, and the
    weights come from the *shifted* window.

Under a deterministic curve the expected compounded rate is computed by
compounding the curve's overnight forwards. With no shift and no historical
fixings this telescopes exactly to ``(P(start)/P(end) - 1) / af``, which is the
consistency test in the suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .core import CouponPeriod, DiscountCurve, TargetCalendar

SHIFT_METHODS = ("lookback", "observation_shift")


@dataclass(frozen=True)
class RfrConvention:
    """Compounding conventions for an overnight-index coupon."""

    index: str = "ESTR"
    lag_days: int = 0
    shift_method: str = "observation_shift"
    basis_days: int = 360
    lockout_days: int = 0

    def __post_init__(self) -> None:
        if self.lag_days < 0:
            raise ValueError("lag_days must be non-negative")
        if self.lockout_days < 0:
            raise ValueError("lockout_days must be non-negative")
        if self.shift_method not in SHIFT_METHODS:
            raise ValueError(f"shift_method must be one of {SHIFT_METHODS}")
        if self.basis_days not in {360, 365}:
            raise ValueError("basis_days must be 360 or 365")


@dataclass(frozen=True)
class ObservationDay:
    """One overnight sub-period inside a coupon.

    ``start``/``end`` delimit the sub-period the rate is *weighted* over, while
    ``rate_start``/``rate_end`` delimit the overnight period the rate itself
    applies to. Under ``observation_shift`` the two coincide; under ``lookback``
    they do not, which is exactly what distinguishes the two conventions.
    """

    fixing_date: date
    start: date
    end: date
    days: int
    rate_start: date
    rate_end: date


def observation_schedule(
    period: CouponPeriod,
    convention: RfrConvention,
    cal: TargetCalendar,
) -> list[ObservationDay]:
    """Business-day observation schedule for one compounded coupon.

    ``fixing_date`` is the date the overnight rate is published for; ``start``
    and ``end`` delimit the sub-period the rate is weighted over.
    """
    if convention.lag_days and convention.shift_method == "observation_shift":
        obs_start = cal.workday(period.start, -convention.lag_days)
        obs_end = cal.workday(period.end, -convention.lag_days)
    else:
        obs_start, obs_end = period.start, period.end

    # If the window opens on a non-business day, the applicable overnight rate is
    # the one published on the PRECEDING business day. Starting from the next
    # business day instead would silently drop the opening weekend from the
    # accrual, so step backwards and clamp the weighting sub-period below.
    first = obs_start if cal.is_business_day(obs_start) else cal.workday(obs_start, -1)
    business_days: list[date] = []
    d = first
    while d < obs_end:
        business_days.append(d)
        d = cal.workday(d, 1)
    if not business_days:
        raise ValueError("coupon period contains no observation business days")

    # Lockout: the final `lockout_days` observations repeat the last live rate.
    live = len(business_days) - convention.lockout_days
    lock_fixing = business_days[max(live - 1, 0)] if convention.lockout_days else None

    out: list[ObservationDay] = []
    for i, day in enumerate(business_days):
        nxt = business_days[i + 1] if i + 1 < len(business_days) else obs_end
        # Clamp to the window so a non-business-day start or end neither drops
        # nor double-counts calendar days.
        sub_start = max(day, obs_start)
        sub_end = min(nxt, obs_end)
        if sub_end <= sub_start:
            continue
        if convention.shift_method == "lookback" and convention.lag_days:
            fixing = cal.workday(day, -convention.lag_days)
        else:
            fixing = day
        if lock_fixing is not None and i >= live:
            fixing = lock_fixing
        # The rate published on `fixing` covers `fixing` to the next business
        # day. Under observation shift that is the weighting sub-period itself,
        # so the compounding telescopes exactly; under lookback and under
        # lockout the rate window is a different one, and it does not.
        if fixing == day and not (lock_fixing is not None and i >= live):
            rate_start, rate_end = sub_start, sub_end
        else:
            rate_start, rate_end = fixing, cal.workday(fixing, 1)
        out.append(
            ObservationDay(
                fixing, sub_start, sub_end, (sub_end - sub_start).days, rate_start, rate_end
            )
        )
    if not out:
        raise ValueError("coupon period contains no observation days")
    return out


def _overnight_forward(
    curve: DiscountCurve,
    start: date,
    end: date,
    value_date: date,
    basis_days: int,
) -> float:
    """Simple overnight forward implied by the projection curve."""
    days = (end - start).days
    if days <= 0:
        raise ValueError("overnight sub-period must have positive length")
    df_start = curve.df(start, value_date)
    df_end = curve.df(end, value_date)
    return (df_start / df_end - 1.0) * basis_days / days


def compounded_rate(
    period: CouponPeriod,
    convention: RfrConvention,
    curve: DiscountCurve,
    value_date: date,
    cal: TargetCalendar,
    *,
    past_fixings: dict[date, float] | None = None,
    observed_compounding: tuple[date, float] | None = None,
) -> float:
    """Compounded overnight rate for one coupon period.

    Observations dated strictly before ``value_date`` are history and must be
    supplied, either as individual overnight rates in ``past_fixings`` or as an
    already-accumulated ``(through_date, compound_factor)`` pair, which is what
    booking systems normally store for a coupon in progress. Later observations
    are projected off ``curve``.
    """
    fixings = dict(past_fixings or {})
    schedule = observation_schedule(period, convention, cal)
    basis = float(convention.basis_days)

    factor = 1.0
    total_days = 0
    observed_through: date | None = None
    if observed_compounding is not None:
        observed_through, factor = observed_compounding[0], float(observed_compounding[1])
        if factor <= 0.0:
            raise ValueError("observed compound factor must be positive")

    missing: list[date] = []
    for obs in schedule:
        total_days += obs.days
        if observed_through is not None and obs.start < observed_through:
            # Already inside the supplied accumulated factor.
            continue
        # An overnight rate for date T is published on T+1, so on the valuation
        # date the rate dated that same day is not yet known and is still
        # projected off the curve. Only strictly earlier dates are history.
        if obs.fixing_date < value_date:
            if obs.fixing_date in fixings:
                rate = float(fixings[obs.fixing_date])
            else:
                missing.append(obs.fixing_date)
                continue
        else:
            rate = _overnight_forward(
                curve, obs.rate_start, obs.rate_end, value_date, convention.basis_days
            )
        factor *= 1.0 + rate * obs.days / basis

    if missing:
        raise ValueError(
            "compounded RFR coupon requires past overnight fixings for "
            + ", ".join(sorted({d.isoformat() for d in missing}))
            + " (or supply observed_compounding)"
        )
    if total_days <= 0:
        raise ValueError("compounded coupon has zero observation days")
    return (factor - 1.0) * basis / total_days


def accrued_compounded_rate(
    period: CouponPeriod,
    convention: RfrConvention,
    curve: DiscountCurve,
    value_date: date,
    cal: TargetCalendar,
    *,
    past_fixings: dict[date, float] | None = None,
    observed_compounding: tuple[date, float] | None = None,
) -> float:
    """Compounded rate over the elapsed part of a coupon, for accrued interest."""
    if value_date <= period.start:
        return 0.0
    stub = CouponPeriod(
        index=period.index,
        unadjusted_start=period.unadjusted_start,
        unadjusted_end=min(value_date, period.unadjusted_end),
        start=period.start,
        end=min(value_date, period.end),
        payment_date=period.payment_date,
        fixing_date=period.fixing_date,
        is_stub=True,
    )
    if stub.end <= stub.start + timedelta(days=0):
        return 0.0
    return compounded_rate(
        stub,
        convention,
        curve,
        value_date,
        cal,
        past_fixings=past_fixings,
        observed_compounding=observed_compounding,
    )


def backward_looking_variance_ratio(
    value_date: date, period: CouponPeriod, *, basis_days: float = 365.0
) -> float:
    """Effective option time for an option on a backward-looking compounded rate.

    A caplet on a term rate expires at the reset: the rate is known in full at
    the start of the period. A caplet on a compounded overnight rate is only
    resolved at the *end* of the period, but the rate stops being random
    gradually as observations accumulate, so it is not simply a longer-dated
    option either.

    Under a Gaussian one-factor volatility structure the accumulated variance of
    the compounded rate carries the decay factor ``(T - u)/(T - S)`` over the
    accrual window (Lyashenko-Mercurio). Integrating the squared factor gives an
    effective time to expiry

        t <= S :  (S - t) + (T - S)/3
        S < t < T :  (T - t)^3 / (3 (T - S)^2)

    which is what this function returns, in years. Multiplying it by the squared
    volatility gives the total variance to use in a Black or Bachelier formula.

    Under this specific one-factor variance approximation, an otherwise
    identical backward-looking option retains the extra ``(T - S)/3`` of
    effective variance beyond the term-rate fixing date. This is a modelling
    approximation, not a universal ordering of market prices across products
    quoted with different volatility conventions.
    """
    start = (period.start - value_date).days / basis_days
    end = (period.end - value_date).days / basis_days
    length = end - start
    if length <= 0.0:
        raise ValueError("coupon period must have positive length")
    if end <= 0.0:
        return 0.0
    if start >= 0.0:
        return start + length / 3.0
    # Already accruing: only the unobserved tail is still random.
    return end**3 / (3.0 * length * length)
