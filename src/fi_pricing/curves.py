"""Multi-curve container separating discounting from forward projection.

Since the 2008 basis blow-out, the curve used to discount a cashflow and the
curve used to project the forward rate that generates it are different objects.
:class:`CurveSet` makes that separation explicit while keeping every existing
single-curve call site working: pass a bare :class:`~fi_pricing.core.DiscountCurve`
anywhere a ``CurveSet`` is accepted and it is promoted to a set whose discount
and projection curves coincide.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from .core import DiscountCurve

CurveLike = "DiscountCurve | CurveSet"


@dataclass(frozen=True)
class CurveSet:
    """A discount curve plus zero or more index projection curves.

    ``projection`` is keyed by index name (for example ``"EURIBOR6M"``). A leg
    with ``index=None``, or with an index that has no dedicated curve, projects
    off the discount curve, which reproduces single-curve behaviour exactly.
    """

    discount: DiscountCurve
    projection: Mapping[str, DiscountCurve] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.discount, DiscountCurve):
            raise TypeError("discount must be a DiscountCurve")
        for name, curve in self.projection.items():
            if not isinstance(curve, DiscountCurve):
                raise TypeError(f"projection curve {name!r} must be a DiscountCurve")
        object.__setattr__(self, "projection", dict(self.projection))

    # -- construction --------------------------------------------------------
    @classmethod
    def single(cls, curve: DiscountCurve) -> CurveSet:
        """Single-curve set: projection and discounting share one curve."""
        return cls(curve, {})

    @property
    def value_date(self) -> date | None:
        return self.discount.value_date

    @property
    def is_single_curve(self) -> bool:
        return not self.projection

    def project(self, index: str | None) -> DiscountCurve:
        """Curve used to project forwards for ``index``."""
        if index is None:
            return self.discount
        return self.projection.get(index, self.discount)

    def has_projection(self, index: str | None) -> bool:
        return index is not None and index in self.projection

    def with_projection(self, index: str, curve: DiscountCurve) -> CurveSet:
        return CurveSet(self.discount, {**self.projection, index: curve})

    def with_discount(self, curve: DiscountCurve) -> CurveSet:
        return CurveSet(curve, self.projection)

    # -- risk helpers --------------------------------------------------------
    def bumped(self, shift_bp: float) -> CurveSet:
        """Bump every curve in the set: the classic parallel rate move."""
        return CurveSet(
            self.discount.bumped(shift_bp),
            {name: curve.bumped(shift_bp) for name, curve in self.projection.items()},
        )

    def bumped_discount(self, shift_bp: float) -> CurveSet:
        """Bump discounting only, holding projection fixed."""
        return CurveSet(self.discount.bumped(shift_bp), self.projection)

    def bumped_projection(self, shift_bp: float, index: str | None = None) -> CurveSet:
        """Bump projection only, holding discounting fixed.

        With ``index=None`` every projection curve moves; this is the basis
        sensitivity of the position. In a single-curve set there is nothing to
        bump, so the set is returned unchanged.
        """
        if index is None:
            return CurveSet(
                self.discount,
                {name: curve.bumped(shift_bp) for name, curve in self.projection.items()},
            )
        if index not in self.projection:
            raise KeyError(f"no projection curve for index {index!r}")
        return CurveSet(
            self.discount,
            {**self.projection, index: self.projection[index].bumped(shift_bp)},
        )

    def bumped_nodes(self, shifts_bp) -> CurveSet:
        """Node-wise bump applied to every curve in the set."""
        return CurveSet(
            self.discount.bumped_nodes(shifts_bp),
            {name: curve.bumped_nodes(shifts_bp) for name, curve in self.projection.items()},
        )

    def roll_forward(self, horizon_days: int) -> CurveSet:
        return CurveSet(
            self.discount.roll_forward(horizon_days),
            {
                name: curve.roll_forward(horizon_days)
                for name, curve in self.projection.items()
            },
        )


def as_curve_set(curve: DiscountCurve | CurveSet) -> CurveSet:
    """Promote a bare discount curve to a single-curve :class:`CurveSet`."""
    if isinstance(curve, CurveSet):
        return curve
    if isinstance(curve, DiscountCurve):
        return CurveSet.single(curve)
    raise TypeError("expected a DiscountCurve or a CurveSet")
