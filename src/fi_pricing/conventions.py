"""Calendars, day-counts and coupon schedules.

This module is written from public market conventions.  It deliberately keeps
contract conventions explicit instead of inferring them from instrument type.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta


def easter_sunday(year: int) -> date:
    """Return Gregorian Easter Sunday (Meeus/Jones/Butcher algorithm)."""
    if year < 1583:
        raise ValueError("Gregorian Easter requires year >= 1583")
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    n = h + ell - 7 * m + 114
    return date(year, n // 31, n % 31 + 1)


def target_holidays(year: int) -> set[date]:
    """TARGET closing days used by the examples in this package."""
    easter = easter_sunday(year)
    return {
        date(year, 1, 1),
        easter - timedelta(days=2),
        easter + timedelta(days=1),
        date(year, 5, 1),
        date(year, 12, 25),
        date(year, 12, 26),
    }


class TargetCalendar:
    """Weekend plus TARGET holidays with standard date adjustments."""

    def __init__(self) -> None:
        self._holiday_cache: dict[int, set[date]] = {}

    def holidays(self, year: int) -> set[date]:
        if year not in self._holiday_cache:
            self._holiday_cache[year] = target_holidays(year)
        return self._holiday_cache[year]

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays(d.year)

    def roll_following(self, d: date) -> date:
        out = d
        while not self.is_business_day(out):
            out += timedelta(days=1)
        return out

    def roll_preceding(self, d: date) -> date:
        out = d
        while not self.is_business_day(out):
            out -= timedelta(days=1)
        return out

    def roll_modified_following(self, d: date) -> date:
        following = self.roll_following(d)
        if following.month == d.month:
            return following
        return self.roll_preceding(d)

    def adjust(self, d: date, convention: str) -> date:
        key = convention.strip().lower().replace("-", "_").replace(" ", "_")
        if key in {"none", "unadjusted"}:
            return d
        if key == "following":
            return self.roll_following(d)
        if key in {"modified_following", "mod_following", "mf"}:
            return self.roll_modified_following(d)
        if key == "preceding":
            return self.roll_preceding(d)
        raise ValueError(f"unsupported business-day convention: {convention!r}")

    def workday(self, start: date, n: int) -> date:
        if n == 0:
            return start
        step = 1 if n > 0 else -1
        left = abs(n)
        out = start
        while left:
            out += timedelta(days=step)
            if self.is_business_day(out):
                left -= 1
        return out


def _last_day_of_feb(d: date) -> bool:
    return d.month == 2 and d.day == calendar.monthrange(d.year, 2)[1]


def year_fraction(d1: date, d2: date, basis: str) -> float:
    """Return an accrual factor for commonly used fixed-income conventions."""
    if d2 < d1:
        raise ValueError("d2 must not precede d1")
    key = basis.upper().replace(" ", "")
    if key in {"ACT/360", "A/360"}:
        return (d2 - d1).days / 360.0
    if key in {"ACT/365", "ACT/365F", "A/365"}:
        return (d2 - d1).days / 365.0
    if key in {"30/360", "30U/360", "BOND"}:
        d1d = d1.day
        d2d = d2.day
        if _last_day_of_feb(d1):
            d1d = 30
        if _last_day_of_feb(d2) and _last_day_of_feb(d1):
            d2d = 30
        if d1d == 31:
            d1d = 30
        if d2d == 31 and d1d >= 30:
            d2d = 30
        numerator = 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + d2d - d1d
        return numerator / 360.0
    if key in {"30E/360", "EUROBOND"}:
        d1d = min(d1.day, 30)
        d2d = min(d2.day, 30)
        numerator = 360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + d2d - d1d
        return numerator / 360.0
    if key in {"ACT/ACT", "ACT/ACTISDA", "A/A"}:
        if d1 == d2:
            return 0.0
        total = 0.0
        cursor = d1
        while cursor < d2:
            boundary = min(d2, date(cursor.year + 1, 1, 1))
            denominator = 366.0 if calendar.isleap(cursor.year) else 365.0
            total += (boundary - cursor).days / denominator
            cursor = boundary
        return total
    raise ValueError(f"unsupported day-count basis: {basis!r}")


def _month_end(d: date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def add_months(d: date, months: int, end_of_month: bool = False) -> date:
    """Shift a date by whole calendar months."""
    absolute_month = d.year * 12 + (d.month - 1) + months
    year, month0 = divmod(absolute_month, 12)
    month = month0 + 1
    last = calendar.monthrange(year, month)[1]
    if end_of_month and _month_end(d):
        day = last
    else:
        day = min(d.day, last)
    return date(year, month, day)


@dataclass(frozen=True)
class ScheduleSpec:
    frequency: int
    business_day_convention: str = "modified_following"
    stub: str = "short_front"
    end_of_month: bool = False
    payment_lag_days: int = 0
    fixing_lag_days: int = 0

    def __post_init__(self) -> None:
        if self.frequency <= 0 or 12 % self.frequency:
            raise ValueError("frequency must be a positive divisor of 12")
        valid_stubs = {"none", "short_front", "long_front", "short_back", "long_back"}
        if self.stub not in valid_stubs:
            raise ValueError(f"unsupported stub convention: {self.stub!r}")
        if self.payment_lag_days < 0 or self.fixing_lag_days < 0:
            raise ValueError("payment and fixing lags must be non-negative")


@dataclass(frozen=True)
class CouponPeriod:
    index: int
    unadjusted_start: date
    unadjusted_end: date
    start: date
    end: date
    payment_date: date
    fixing_date: date
    is_stub: bool = False


def _front_stub_boundaries(effective: date, maturity: date, spec: ScheduleSpec) -> list[date]:
    step = 12 // spec.frequency
    interior: list[date] = []
    cursor = maturity
    prior = maturity
    while True:
        prior = add_months(cursor, -step, spec.end_of_month)
        if prior <= effective:
            break
        interior.append(prior)
        cursor = prior
    interior.sort()
    if prior == effective:
        return [effective, *interior, maturity]
    if spec.stub == "none":
        raise ValueError("dates are not regular and stub='none'")
    if spec.stub == "short_front":
        return [effective, *interior, maturity]
    if spec.stub == "long_front":
        return [effective, *interior[1:], maturity] if interior else [effective, maturity]
    raise ValueError("front-stub generation requires a front-stub convention")


def _back_stub_boundaries(effective: date, maturity: date, spec: ScheduleSpec) -> list[date]:
    step = 12 // spec.frequency
    interior: list[date] = []
    cursor = effective
    after = effective
    while True:
        after = add_months(cursor, step, spec.end_of_month)
        if after >= maturity:
            break
        interior.append(after)
        cursor = after
    if after == maturity:
        return [effective, *interior, maturity]
    if spec.stub == "none":
        raise ValueError("dates are not regular and stub='none'")
    if spec.stub == "short_back":
        return [effective, *interior, maturity]
    if spec.stub == "long_back":
        return [effective, *interior[:-1], maturity] if interior else [effective, maturity]
    raise ValueError("back-stub generation requires a back-stub convention")


def build_schedule(
    effective: date,
    maturity: date,
    spec: ScheduleSpec,
    cal: TargetCalendar | None = None,
) -> list[CouponPeriod]:
    """Construct a coupon schedule with explicit stubs, rolls and lags."""
    if effective >= maturity:
        raise ValueError("effective must be before maturity")
    calendar_obj = cal or TargetCalendar()
    if spec.stub in {"none", "short_front", "long_front"}:
        boundaries = _front_stub_boundaries(effective, maturity, spec)
    else:
        boundaries = _back_stub_boundaries(effective, maturity, spec)

    step = 12 // spec.frequency
    result: list[CouponPeriod] = []
    for idx, (u0, u1) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True), start=1):
        start = calendar_obj.adjust(u0, spec.business_day_convention)
        end = calendar_obj.adjust(u1, spec.business_day_convention)
        payment = calendar_obj.workday(end, spec.payment_lag_days) if spec.payment_lag_days else end
        fixing = calendar_obj.workday(start, -spec.fixing_lag_days) if spec.fixing_lag_days else start
        regular_end = add_months(u0, step, spec.end_of_month)
        result.append(
            CouponPeriod(
                index=idx,
                unadjusted_start=u0,
                unadjusted_end=u1,
                start=start,
                end=end,
                payment_date=payment,
                fixing_date=fixing,
                is_stub=(regular_end != u1),
            )
        )
    return result


def generate_schedule(
    effective: date,
    maturity: date,
    freq: int,
    cal: TargetCalendar,
    *,
    business_day_convention: str = "modified_following",
    stub: str = "short_front",
    end_of_month: bool = False,
) -> list[tuple[date, date]]:
    spec = ScheduleSpec(
        frequency=freq,
        business_day_convention=business_day_convention,
        stub=stub,
        end_of_month=end_of_month,
    )
    return [(p.start, p.end) for p in build_schedule(effective, maturity, spec, cal)]


def remaining_periods(
    effective: date,
    maturity: date,
    freq: int,
    value_date: date,
    cal: TargetCalendar,
    spec: ScheduleSpec | None = None,
) -> list[CouponPeriod]:
    schedule_spec = spec or ScheduleSpec(frequency=freq)
    return [
        p
        for p in build_schedule(effective, maturity, schedule_spec, cal)
        if p.payment_date > value_date
    ]
