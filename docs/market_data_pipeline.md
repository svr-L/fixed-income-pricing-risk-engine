# Market-data pipeline: quotes, curves and volatility

Version 0.5 adds the first explicit market-data construction layer upstream of
pricing and risk.

## Curve bootstrap

`bootstrap_single_curve` accepts synthetic deposit, FRA and par-swap quotes and
builds a continuously compounded zero curve. Deposit and FRA discount factors
are updated directly; the terminal discount factor of each par swap is solved
by root finding. Coupon dates between quoted nodes use the selected curve
interpolation.

Two interpolation modes are available:

- `linear_zero`: linear interpolation of continuously compounded zero rates;
- `log_linear_df`: linear interpolation of log discount factors.

The latter gives smoother discount-factor behaviour and is the default of the
bootstrapper. Monotone-convex forward interpolation remains a future extension.

## Volatility conventions

The forward-option layer supports:

- Black-76 / lognormal volatility;
- shifted Black / shifted-lognormal volatility;
- Bachelier / normal volatility.

Bachelier and shifted Black allow the pricing layer to handle negative-rate
environments without silently collapsing to intrinsic value.

`implied_volatility` and model-specific wrappers invert an undiscounted option
premium back to the quoted volatility convention using a bracketed root solver.

## Cap-vol to caplet-vol stripping

A quoted cap volatility is a flat volatility for the full cap strip; it is not a
caplet volatility. `strip_caplet_vol_surface` prices each quoted cap and strips
forward caplet volatilities recursively. Between successive quoted cap
maturities, newly introduced caplets are assumed to share one piecewise-constant
forward volatility. The resulting expiry-by-strike caplet surface can then be
used for market-facing bucketed Vega and caplet-level risk.

The test suite verifies that stripped caplet vols reconstruct the original cap
prices across every synthetic maturity/strike quote.
