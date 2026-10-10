# Multi-curve pricing and compounded RFR coupons

`CurveSet` separates collateral/OIS discounting from index-specific forward projection.
Passing a bare `DiscountCurve` promotes it to a single-curve set for backward-compatible
vanilla pricing.

## Two-stage bootstrap

`bootstrap_multi_curve` first constructs the discount curve from OIS instruments and then
builds each projection curve from index quotes using OIS discounting. Floating cashflows
are projected from the index curve and discounted on the discount curve; the single-curve
telescoping shortcut is not used when the curves differ.

## Compounded RFR coupons

`RfrConvention` supports lookback, observation shift and lockout. With no lag/lockout,
daily forward compounding telescopes to the discount-factor ratio; dedicated invariance
tests cover business-day, weekend and holiday openings.

Observed overnight rates strictly before the valuation date are history and must be
provided through daily fixings or an observed compounded factor for an in-progress coupon.

## Risk decomposition

`multi_curve_risk` reports total DV01, discount-curve DV01 and projection/basis DV01. This
separates sensitivities that a single aggregate parallel bump cannot identify.
