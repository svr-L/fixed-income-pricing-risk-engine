from datetime import date

import pytest

from fi_pricing.bootstrap import DepositQuote, FRAQuote, SwapQuote, bootstrap_single_curve
from fi_pricing.core import (
    DiscountCurve,
    ScheduleSpec,
    TargetCalendar,
    add_months,
    build_schedule,
    year_fraction,
)


def test_log_linear_df_interpolation_is_linear_in_log_discount_factor() -> None:
    curve = DiscountCurve.from_discount_factors(
        [0, 365, 730],
        [1.0, 0.97, 0.92],
        date(2025, 1, 2),
        interpolation="log_linear_df",
    )
    expected = (0.97 * 0.92) ** 0.5
    assert curve.df_days(547.5) == pytest.approx(expected, rel=1e-12)


def test_single_curve_bootstrap_recovers_flat_curve_from_deposit_and_swaps() -> None:
    value_date = date(2025, 1, 2)
    cal = TargetCalendar()
    true_curve = DiscountCurve([0, 3650], [0.03, 0.03], value_date, interpolation="log_linear_df")
    spec = ScheduleSpec(2, business_day_convention="unadjusted", stub="none")

    dep_mat = add_months(value_date, 6)
    alpha = year_fraction(value_date, dep_mat, "ACT/360")
    dep_rate = (1.0 / true_curve.df(dep_mat, value_date) - 1.0) / alpha
    deposit = DepositQuote(dep_mat, dep_rate)

    swaps = []
    for years in range(1, 6):
        maturity = add_months(value_date, 12 * years)
        schedule = build_schedule(value_date, maturity, spec, cal)
        annuity = sum(
            year_fraction(p.start, p.end, "30/360") * true_curve.df(p.payment_date, value_date)
            for p in schedule
        )
        par = (1.0 - true_curve.df(schedule[-1].end, value_date)) / annuity
        swaps.append(SwapQuote(maturity, par, 2, "30/360", spec))

    result = bootstrap_single_curve(
        value_date,
        deposits=[deposit],
        swaps=swaps,
        cal=cal,
        interpolation="log_linear_df",
    )
    for node_date, df in zip(result.node_dates[1:], result.discount_factors[1:], strict=True):
        assert df == pytest.approx(true_curve.df(node_date, value_date), rel=2e-10)
    assert max(abs(x) for x in result.repricing_errors) < 1e-11


def test_fra_bootstrap_extends_discount_curve_directly() -> None:
    value_date = date(2025, 1, 2)
    true_curve = DiscountCurve([0, 3650], [0.025, 0.025], value_date, interpolation="log_linear_df")
    six_months = add_months(value_date, 6)
    one_year = add_months(value_date, 12)
    dep_alpha = year_fraction(value_date, six_months, "ACT/360")
    dep_rate = (1.0 / true_curve.df(six_months, value_date) - 1.0) / dep_alpha
    fra_alpha = year_fraction(six_months, one_year, "ACT/360")
    fra_rate = (true_curve.df(six_months, value_date) / true_curve.df(one_year, value_date) - 1.0) / fra_alpha
    result = bootstrap_single_curve(
        value_date,
        deposits=[DepositQuote(six_months, dep_rate)],
        fras=[FRAQuote(six_months, one_year, fra_rate)],
    )
    assert result.curve.df(one_year, value_date) == pytest.approx(
        true_curve.df(one_year, value_date), rel=1e-12
    )
