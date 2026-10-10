from datetime import date

import pytest

from fi_pricing.core import (
    DiscountCurve,
    ScheduleSpec,
    black76_forward_premium,
    build_schedule,
    easter_sunday,
    year_fraction,
)


def test_front_stub_never_starts_before_effective() -> None:
    effective = date(2024, 3, 20)
    maturity = date(2029, 1, 15)
    schedule = build_schedule(effective, maturity, ScheduleSpec(2, stub="short_front"))
    assert schedule[0].unadjusted_start == effective
    assert all(p.unadjusted_start >= effective for p in schedule)
    assert all(p.unadjusted_end <= maturity for p in schedule)
    assert schedule[0].is_stub


def test_long_front_and_back_stubs_are_explicit() -> None:
    effective = date(2024, 3, 20)
    maturity = date(2029, 1, 15)
    short_front = build_schedule(effective, maturity, ScheduleSpec(2, stub="short_front"))
    long_front = build_schedule(effective, maturity, ScheduleSpec(2, stub="long_front"))
    assert long_front[0].unadjusted_end > short_front[0].unadjusted_end

    short_back = build_schedule(effective, maturity, ScheduleSpec(2, stub="short_back"))
    long_back = build_schedule(effective, maturity, ScheduleSpec(2, stub="long_back"))
    assert long_back[-1].unadjusted_start < short_back[-1].unadjusted_start


def test_30e_360_is_distinct_from_us_30_360() -> None:
    d1 = date(2024, 1, 15)
    d2 = date(2024, 7, 31)
    assert year_fraction(d1, d2, "30/360") == pytest.approx(196 / 360)
    assert year_fraction(d1, d2, "30E/360") == pytest.approx(195 / 360)


def test_gregorian_easter_works_beyond_old_algorithm_range() -> None:
    assert easter_sunday(2200) == date(2200, 4, 6)


def test_roll_forward_accepts_zero_tenor_node() -> None:
    curve = DiscountCurve([0, 365, 730], [0.02, 0.025, 0.03], date(2025, 1, 2))
    rolled = curve.roll_forward(30)
    assert rolled.df_days(0) == pytest.approx(1.0)
    assert rolled.df_days(365) == pytest.approx(curve.df_days(395) / curve.df_days(30))


def test_black_rejects_negative_rate_domain_instead_of_silent_intrinsic() -> None:
    with pytest.raises(ValueError, match="Bachelier or shifted Black"):
        black76_forward_premium(-0.005, 0.01, 1.0, 0.20, 1)
