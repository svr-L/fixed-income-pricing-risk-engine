"""Two-stage multi-curve build, an RFR coupon and the discount/basis risk split."""

from datetime import date

from fi_pricing import core as fc
from fi_pricing.bootstrap import DepositQuote, SwapQuote, bootstrap_multi_curve
from fi_pricing.curves import CurveSet
from fi_pricing.irs import Leg, price_irs, price_leg
from fi_pricing.rfr import RfrConvention, compounded_rate
from fi_pricing.risk import multi_curve_risk

VALUE_DATE = date(2025, 1, 2)
CAL = fc.TargetCalendar()

# ---------------------------------------------------------------------------
# Stage one: OIS discount curve. Stage two: EURIBOR 6M projection curve, priced
# with OIS discounting.
# ---------------------------------------------------------------------------
result = bootstrap_multi_curve(
    VALUE_DATE,
    ois_deposits=[DepositQuote(date(2025, 7, 2), 0.0250)],
    ois_swaps=[
        SwapQuote(date(2027, 1, 4), 0.0255),
        SwapQuote(date(2030, 1, 2), 0.0265),
        SwapQuote(date(2035, 1, 2), 0.0275),
    ],
    projection_quotes={
        "EURIBOR6M": [
            SwapQuote(date(2027, 1, 4), 0.0290),
            SwapQuote(date(2030, 1, 2), 0.0302),
            SwapQuote(date(2035, 1, 2), 0.0315),
        ]
    },
    projection_deposits={"EURIBOR6M": [DepositQuote(date(2025, 7, 2), 0.0285)]},
)
curves = result.curves

print("Two-stage bootstrap")
print(f"  OIS repricing error        : {max(abs(e) for e in result.discount.repricing_errors):.2e}")
proj_err = max(abs(e) for e in result.projection["EURIBOR6M"].repricing_errors)
print(f"  EURIBOR6M repricing error  : {proj_err:.2e}")
for tenor in (365, 1825, 3650):
    print(f"  basis at {tenor:5d}d          : {result.basis_bp('EURIBOR6M', tenor):7.3f} bp")

# ---------------------------------------------------------------------------
# A compounded ESTR coupon, and the invariant it must satisfy.
# ---------------------------------------------------------------------------
period = fc.build_schedule(
    date(2025, 4, 2), date(2026, 4, 2), fc.ScheduleSpec(4), CAL
)[0]
ois = curves.discount
plain = compounded_rate(period, RfrConvention(index="ESTR"), ois, VALUE_DATE, CAL)
telescoped = (ois.df(period.start, VALUE_DATE) / ois.df(period.end, VALUE_DATE) - 1.0)
telescoped *= 360.0 / (period.end - period.start).days

print("\nCompounded ESTR coupon")
print(f"  period                     : {period.start} -> {period.end}")
print(f"  compounded overnight rate  : {plain * 100:.6f}%")
print(f"  telescoped curve ratio     : {telescoped * 100:.6f}%")
print(f"  difference                 : {plain - telescoped:+.2e}")
for convention, label in (
    (RfrConvention(lag_days=5, shift_method="lookback"), "5d lookback"),
    (RfrConvention(lag_days=5, shift_method="observation_shift"), "5d obs shift"),
    (RfrConvention(lockout_days=5), "5d lockout"),
):
    rate = compounded_rate(period, convention, ois, VALUE_DATE, CAL)
    print(f"  {label:<27}: {rate * 100:.6f}%")

# ---------------------------------------------------------------------------
# Risk split: a par EURIBOR swap carries almost no discount-curve DV01.
# ---------------------------------------------------------------------------
fixed = Leg(1e6, date(2025, 4, 2), date(2030, 4, 2), 1, "30/360", True, coupon=0.0302)
floating = Leg(
    1e6, date(2025, 4, 2), date(2030, 4, 2), 2, "ACT/360", False, index="EURIBOR6M"
)
risk = multi_curve_risk(
    lambda c: price_irs(VALUE_DATE, VALUE_DATE, fixed, floating, c, CAL).payer_pv,
    curves,
)
print("\nMulti-curve risk (payer swap)")
print(f"  total DV01                 : {risk.total_dv01:10,.2f}")
print(f"  discount DV01              : {risk.discount_dv01:10,.2f}")
print(f"  basis (projection) DV01    : {risk.basis_dv01:10,.2f}")

# ---------------------------------------------------------------------------
# Single-curve behaviour is preserved exactly.
# ---------------------------------------------------------------------------
bare = price_irs(VALUE_DATE, VALUE_DATE, fixed, floating, ois, CAL).payer_pv
promoted = price_irs(VALUE_DATE, VALUE_DATE, fixed, floating, CurveSet.single(ois), CAL).payer_pv
print("\nBackward compatibility")
print(f"  bare DiscountCurve         : {bare:14,.8f}")
print(f"  promoted CurveSet          : {promoted:14,.8f}")
print(f"  difference                 : {bare - promoted:+.2e}")

rfr_leg = Leg(
    1e6, date(2025, 4, 2), date(2030, 4, 2), 4, "ACT/360", False,
    rfr=RfrConvention(index="ESTR"),
)
print(f"\nCompounded ESTR leg PV       : {price_leg(rfr_leg, VALUE_DATE, ois, CAL).pv:14,.2f}")
