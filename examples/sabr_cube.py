import numpy as np

from fi_pricing.sabr import (
    SABRParams,
    SwaptionVolCube,
    calibrate_sabr_cube,
    sabr_lognormal_vol,
)

expiries = (1.0, 5.0)
tenors = (2.0, 10.0)
strikes = (-0.005, 0.005, 0.015, 0.025, 0.035)
forwards = np.array([[0.010, 0.015], [0.012, 0.018]])
grid = []
for i, expiry in enumerate(expiries):
    slab = []
    for j, _tenor in enumerate(tenors):
        params = SABRParams(
            0.020 + 0.002 * i + 0.001 * j,
            0.5,
            -0.2,
            0.40,
            0.02,
        )
        slab.append(
            [
                sabr_lognormal_vol(
                    float(forwards[i, j]), strike, expiry, params
                )
                for strike in strikes
            ]
        )
    grid.append(slab)

cube = SwaptionVolCube(expiries, tenors, strikes, grid)
fit = calibrate_sabr_cube(cube, forwards, beta=0.5, shift=0.02)
print("Cube SABR RMSE:", fit.rmse_vol)
