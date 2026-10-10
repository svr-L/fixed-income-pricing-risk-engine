# Vasicek and CIR under Q and P

`fi_pricing.models.short_rate` implements two standard affine short-rate models.

Under the risk-neutral measure, analytic zero-bond formulas are used to calibrate model
parameters cross-sectionally to a synthetic zero curve. Under the physical measure,
Vasicek parameters are estimated from the exact Gaussian transition and CIR parameters
from the exact non-central-chi-square transition density.

The Q calibration and P estimation are deliberately separate: they answer different
questions and no market-price-of-risk mapping is assumed by default.

The test suite uses only model-generated curves and histories and verifies parameter/curve
recovery, likelihood finiteness and CIR admissibility diagnostics.
