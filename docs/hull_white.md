# Hull-White one-factor

## Why this model and not Vasicek or CIR

The package already had constant-parameter Vasicek and CIR models. They are
valid affine pricing models, but their term structure is an output of a small
parameter set and therefore does not in general reproduce an arbitrary observed
initial discount curve exactly. A deterministic shift can fix that limitation;
Hull-White builds the curve fit directly into the model through a time-dependent
drift.

Hull-White is the same Ornstein-Uhlenbeck dynamic with a time-dependent drift:

    dr(t) = (theta(t) - a r(t)) dt + sigma dW(t)

`theta(t)` is chosen so that the model reproduces the initial curve exactly. It
never has to be computed: the affine reconstruction

    P(t,T) = A(t,T) exp(-B(t,T) r(t))
    B(t,T) = (1 - e^{-a(T-t)}) / a
    A(t,T) = [P(0,T)/P(0,t)] exp( B(t,T) f(0,t) - sigma^2/(4a) (1-e^{-2at}) B(t,T)^2 )

absorbs it through the market discount factors and the market instantaneous
forward `f(0,t)`, which `DiscountCurve.instantaneous_forward` already provides.

The test suite asserts the consequence directly: `P^HW(0,T)` equals the market
discount factor to 1e-15 for every maturity and for every `(a, sigma)`.

## Closed forms

**Zero-coupon bond options.** Under the T-forward measure the forward bond price
is lognormal, so a European option on `P(T,S)` is Black with integrated
volatility

    sigma_p = sigma * sqrt((1 - e^{-2aT}) / (2a)) * B(T,S)

**Caplets.** A caplet on the rate set at `T` and paid at `S` is `(1 + K*tau)`
puts on `P(T,S)` struck at `1/(1 + K*tau)`. A floorlet is the corresponding
call. Note the direction: a *cap* is a *put* on the bond.

**Swaptions (Jamshidian).** A payer swaption is a put on the fixed-leg coupon
bond struck at par. In a one-factor model every bond price is monotone in the
single state variable, so there is a unique critical short rate `r*` at which
the coupon bond is worth par, and the coupon-bond option decomposes into a
portfolio of zero-bond options struck at each `P(T, T_i; r*)`. This is exact,
not an approximation.

## Calibration

`calibrate_hull_white` fits `(a, sigma)` by least squares to instrument prices.
`caplet_instruments_from_surface` builds the targets from a stripped caplet-vol
surface, so the model is fitted to the same market the stripper consumed.

Two practical points:

- `a` and `sigma` are weakly jointly identified from a single expiry slice. Mean
  reversion mainly governs how volatility decays *across* expiries, so a target
  set that does not span expiries can trade one parameter against the other.
  `fix_a` exists for that case.
- A two-parameter model cannot fit a smile. Calibrating to a full strike-by-
  expiry strip lands at roughly 1% of the average premium in RMSE, which is the
  honest order of magnitude, not a bug. If you need the smile, this is the point
  at which a two-factor or stochastic-volatility extension earns its keep.


### Calibration objectives

`price` minimises raw price errors. `model_sigma_normalised` divides errors by
sensitivity to the Hull-White model parameter `sigma` at the initial point; this
is useful numerical scaling but is **not** a market Vega. The legacy name
`vega_normalised` remains an alias for backwards compatibility.

`market_vega_normalised` requires each calibration instrument to carry the Vega
of its market quote convention (Black/Bachelier/shifted Black). Price residuals
then read approximately in implied-volatility units, which is the preferred
cross-instrument metric when those Vegas are available.

## What this unlocks, and what it does not

The analytic forms above are all European. The reason to have a curve-consistent
short-rate model at all is the path-dependent products the closed forms cannot
reach - Bermudan swaptions and callable structures - which need a trinomial tree
or Least-Squares Monte Carlo on top of this layer. That is deliberately left for
a later version; what is here is the calibrated model those methods would run on.

## Scope

This is a single-curve model. In the multi-curve setting it is applied to the
discount curve with the projection basis held deterministic. That is the
standard practical assumption, but it is an assumption, and it is the reason
`swaption_from_schedule` takes the discount curve out of a `CurveSet` rather
than pretending to model both.
