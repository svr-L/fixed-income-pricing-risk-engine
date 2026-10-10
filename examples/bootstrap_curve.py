from datetime import date

from fi_pricing.bootstrap import DepositQuote, SwapQuote, bootstrap_single_curve
from fi_pricing.core import (
    DiscountCurve,
    ScheduleSpec,
    TargetCalendar,
    add_months,
    build_schedule,
    year_fraction,
)

value_date = date(2025, 1, 2)
cal = TargetCalendar()
true_curve = DiscountCurve([0, 3650], [0.03, 0.03], value_date, interpolation="log_linear_df")
spec = ScheduleSpec(2, business_day_convention="unadjusted", stub="none")

dep_mat = add_months(value_date, 6)
alpha = year_fraction(value_date, dep_mat, "ACT/360")
dep_rate = (1.0 / true_curve.df(dep_mat, value_date) - 1.0) / alpha

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
    deposits=[DepositQuote(dep_mat, dep_rate)],
    swaps=swaps,
    cal=cal,
)

print("BOOTSTRAPPED DISCOUNT CURVE")
for d, df in zip(result.node_dates, result.discount_factors, strict=True):
    print(f"{d.isoformat()}  DF={df:.10f}")
print(f"max repricing error = {max(abs(x) for x in result.repricing_errors):.3e}")
