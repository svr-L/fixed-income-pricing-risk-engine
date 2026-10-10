"""Compatibility facade for cap/floor instruments."""
from .instruments.caps import CapResult, Caplet, default_vol_surface, price_cap, vol_cube_bilinear

__all__ = ["CapResult", "Caplet", "default_vol_surface", "price_cap", "vol_cube_bilinear"]
