# Admissibility checks

`fi_pricing.arbitrage_checks` contains reusable diagnostics that return a `CheckResult`
and can raise `ArbitrageError` when strict enforcement is requested.

## Discount factors

Discount factors must be positive. They need not be decreasing: negative rates can make
discount factors rise above one. `check_discount_factors` therefore checks positivity and
screens implied continuously compounded forwards against a configurable plausibility
floor rather than imposing non-negative rates.

## Total variance

`check_total_variance_monotone` verifies non-decreasing `sigma^2 T` across expiry for a
fixed strike grid. It is used for cap/caplet surface diagnostics. For swaption cubes it is
best interpreted as a term-structure diagnostic because different expiries need not share
an identical underlying forward swap.

## Strike convexity

`check_butterfly_convexity` verifies convexity of option prices in strike on the supplied
grid. SABR calibration uses this to distinguish numerical fit quality from smile
admissibility. The diagnostic is restricted to the quoted/interpolated strike range; it
does not certify extrapolation.

## Model-domain guards

Shifted Black rejects an insufficient shift, and the Hagan SABR approximation rejects
non-finite or non-positive implied volatilities rather than passing a broken expansion
downstream to an option formula.
