"""Compatibility facade for swap instruments."""
from .instruments.swaps import CashFlow, IRSResult, Leg, LegResult, price_irs, price_leg

__all__ = ["CashFlow", "IRSResult", "Leg", "LegResult", "price_irs", "price_leg"]
