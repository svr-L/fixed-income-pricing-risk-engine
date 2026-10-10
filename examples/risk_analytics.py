from datetime import date

from fi_pricing.cap_floor import price_cap
from fi_pricing.core import DiscountCurve
from fi_pricing.irs import Leg, price_irs
from fi_pricing.market_data import CURVE_DAYS, CURVE_ZERO_RATES
from fi_pricing.risk import (
    bucketed_surface_vega,
    carry_roll_down,
    curve_factor_risk,
    explain_curve_pnl,
    key_rate_risk,
    short_rate_parameter_greeks,
)
from fi_pricing.short_rate import ShortRateParams
from fi_pricing.volatility import VolSurface2D

value_date = date(2025, 1, 2)
curve = DiscountCurve(CURVE_DAYS, CURVE_ZERO_RATES, value_date)
fixed = Leg(1_000_000, date(2024, 7, 2), date(2030, 7, 2), 1, "30/360", True, 0.0325)
floating = Leg(
    1_000_000,
    date(2024, 7, 2),
    date(2030, 7, 2),
    2,
    "ACT/360",
    False,
    current_fixing=0.0290,
)


def payer_pv(c):
    return price_irs(date(2024, 12, 30), value_date, fixed, floating, curve=c).payer_pv


print("KEY-RATE DV01")
for bucket in key_rate_risk(payer_pv, curve):
    print(f"{bucket.tenor_years:>5.1f}Y  {bucket.dv01:>10.2f}  convexity {bucket.convexity:>10.4f}")

print("\nCURVE FACTOR RISK")
for factor in curve_factor_risk(payer_pv, curve):
    print(f"{factor.name:>10s}  DV01 {factor.dv01:>10.2f}  convexity {factor.convexity:>10.4f}")

def dated_payer_pv(vd, c):
    return price_irs(date(2024, 12, 30), vd, fixed, floating, curve=c).payer_pv


carry_roll = carry_roll_down(dated_payer_pv, value_date, curve, horizon_days=5)
print("\n5-DAY CARRY / ROLL-DOWN")
print(f"Financing carry {carry_roll.financing_carry:>12.4f}")
print(f"Roll-down       {carry_roll.roll_down:>12.4f}")
print(f"Total P&L       {carry_roll.total_pnl:>12.4f}")

# A small non-parallel curve shock and second-order P&L explain.
shock_bp = [1.5 - 0.08 * (d / 365.0) for d in CURVE_DAYS]
explain = explain_curve_pnl(payer_pv, curve, curve.bumped_nodes(shock_bp))
print("\nCURVE P&L EXPLAIN")
print(f"Actual P&L      {explain.actual_pnl:>12.4f}")
print(f"First order     {explain.first_order:>12.4f}")
print(f"Second order    {explain.second_order:>12.4f}")
print(f"Residual        {explain.residual:>12.6f}")

surface = VolSurface2D(
    [0.5, 1.0, 2.0, 5.0, 10.0],
    [0.01, 0.02, 0.03, 0.04, 0.05],
    [
        [0.24, 0.22, 0.20, 0.19, 0.19],
        [0.23, 0.21, 0.195, 0.185, 0.185],
        [0.22, 0.20, 0.19, 0.18, 0.18],
        [0.20, 0.19, 0.18, 0.175, 0.175],
        [0.19, 0.18, 0.175, 0.17, 0.17],
    ],
)


def cap_pv(s):
    return price_cap(
        value_date=value_date,
        effective=date(2024, 10, 2),
        maturity=date(2030, 10, 2),
        notional=1_000_000,
        strike=0.0325,
        frequency=2,
        first_fixing=0.0290,
        caplet_vol_surface=s,
    ).npv


vega = bucketed_surface_vega(cap_pv, surface)
print("\nLARGEST CAPLET-VOL VEGA BUCKETS (1 vol point P&L)")
for bucket in sorted(vega, key=lambda x: abs(x.vega_1vol_point), reverse=True)[:8]:
    print(
        f"expiry {bucket.expiry:>4.1f}Y strike {bucket.strike:>6.2%}: "
        f"{bucket.vega_1vol_point:>10.2f}"
    )

params = ShortRateParams(0.028, 0.45, 0.041, 0.012)
print("\nVASICEK Q MODEL-PARAMETER GREEKS")
for greek in short_rate_parameter_greeks(payer_pv, "vasicek", params, value_date, CURVE_DAYS):
    print(f"{greek.parameter:>5s}: dPV/dp {greek.delta:>12.2f}, d2PV/dp2 {greek.gamma:>12.2f}")
