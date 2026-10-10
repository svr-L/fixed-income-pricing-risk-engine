# Validation

The package uses synthetic fixtures and model identities rather than external private
spreadsheets or cached third-party outputs.

Core checks include par-swap zero NPV, floating-leg telescoping, cap-floor parity,
key-rate/parallel DV01 reconciliation, curve-bootstrap quote repricing, RFR compounding
identities, option implied-volatility round trips, short-rate synthetic recovery,
Hull-White initial-curve fit and SABR synthetic calibration/admissibility tests.

These checks serve two purposes: regression protection and model diagnostics. A numerical
fit is not treated as sufficient evidence of economic admissibility; the volatility and
SABR layers therefore expose separate static-arbitrage diagnostics.
