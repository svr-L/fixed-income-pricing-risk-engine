"""Compatibility facade for European swaptions."""
from .instruments.swaptions import (
    SwaptionResult,
    cash_settled_annuity,
    discount_to_expiry,
    price_swaption,
)

__all__ = ["SwaptionResult", "cash_settled_annuity", "discount_to_expiry", "price_swaption"]
