from fi_pricing.core import forward_option_premium
from fi_pricing.volatility import (
    implied_vol_bachelier,
    implied_vol_black76,
    implied_vol_shifted_black,
)

black = forward_option_premium(0.035, 0.03, 2.0, 0.22, 1, model="black")
normal = forward_option_premium(-0.005, 0.0, 1.5, 0.012, 1, model="bachelier")
shifted = forward_option_premium(
    -0.005,
    0.0,
    1.5,
    0.30,
    1,
    model="shifted_black",
    shift=0.02,
)

print("IMPLIED-VOL ROUND TRIPS")
print(f"Black-76      : {implied_vol_black76(black, 0.035, 0.03, 2.0, 1):.6f}")
print(f"Bachelier     : {implied_vol_bachelier(normal, -0.005, 0.0, 1.5, 1):.6f}")
print(
    "Shifted Black : "
    f"{implied_vol_shifted_black(shifted, -0.005, 0.0, 1.5, 1, 0.02):.6f}"
)
