# Risk analytics

The risk layer separates market sensitivities, model-parameter sensitivities and local
option Greeks.

## Curve risk

Available measures include parallel DV01/convexity, key-rate DV01 and convexity,
level/slope/curvature factor shocks, arbitrary user-supplied factor loadings and a
second-order curve P&L explain with optional cross-Gamma terms.

For a `CurveSet`, `multi_curve_risk` separates discounting risk from projection/basis
risk. This is distinct from a single parallel bump across every curve.

## Volatility risk

Black/shifted-Black/Bachelier option analytics expose Delta, Gamma, Vega, Theta, Vanna
and Volga. Caps aggregate caplet-level sensitivities and `bucketed_surface_vega` bumps
volatility-surface nodes independently.

## Carry / roll-down

`carry_roll_down` advances the valuation date on the deterministic forward curve implied
by today's discount factors. It reports roll-down, financing carry and total P&L under an
explicit convention; it is not claimed as a unique market-standard decomposition.

## Model risk

`short_rate_parameter_greeks` measures sensitivity to Vasicek/CIR Q-model parameters by
bump-and-reprice without recalibration. These are model-risk sensitivities, not directly
hedgeable market Greeks.

## P&L explain

`explain_curve_pnl` compares full-revaluation P&L with a first/second-order curve explain.
The residual can reveal higher-order nonlinearity, interpolation effects or omitted risk
factors such as volatility/basis changes.
