from datetime import date

from fi_pricing.irs import Leg, price_irs


def main() -> None:
    value_date = date(2025, 1, 2)
    fixed = Leg(
        notional=1_000_000, effective=date(2024, 7, 2), maturity=date(2030, 7, 2),
        frequency=1, basis="30/360", is_fixed=True, coupon=0.0325,
    )
    floating = Leg(
        notional=1_000_000, effective=date(2024, 7, 2), maturity=date(2030, 7, 2),
        frequency=2, basis="ACT/360", is_fixed=False, current_fixing=0.0290,
    )
    res = price_irs(date(2024, 12, 30), value_date, fixed, floating)
    print("IRS synthetic example")
    print(f"Fixed-leg PV       : {res.fixed.pv:,.2f}")
    print(f"Floating-leg PV    : {res.floating.pv:,.2f}")
    print(f"Par swap rate      : {res.par_coupon * 100:.6f}%")
    print(f"Receiver swap NPV  : {res.receiver_pv:,.2f}")
    print(f"Payer DV01         : {res.dv01:,.2f}")
    print(f"Curve convexity    : {res.curve_convexity:,.4f} per bp^2")


if __name__ == "__main__":
    main()
