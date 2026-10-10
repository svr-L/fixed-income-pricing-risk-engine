# SABR and swaption volatility cubes

`SwaptionVolCube` stores expiry × underlying-tenor × strike implied volatilities.
Strike and tenor interpolation are linear in volatility; expiry interpolation is explicit
(`linear_vol`, `linear_total_variance` or the supported forward-vol rule) so that the
chosen surface convention is visible to callers.

`SABRParams` and `sabr_lognormal_vol` implement the shifted-lognormal Hagan approximation.
`calibrate_sabr_smile` fits alpha/rho/nu at fixed beta and shift in implied-volatility
units. `calibrate_sabr_cube` calibrates each expiry/tenor smile and reports aggregate fit
statistics and smile admissibility diagnostics.

A close calibration fit is not the same as an admissible smile. The Hagan expansion can
produce butterfly-convexity violations inside the quoted strike range, so calibration
results report a separate diagnostic and can optionally enforce it strictly.

The cube total-variance check is a term-structure diagnostic. Because swaptions at
different expiries generally reference different forward swaps, it should not be read as
a complete theorem of calendar no-arbitrage for the entire swaption cube.
