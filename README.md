# Fixed-Income Pricing & Quantitative Risk Engine

Independent Python implementation of interest-rate pricing, market-data construction,
volatility modelling and quantitative risk analytics. The package is designed as a
transparent research/engineering engine rather than a production trading system.

## Scope

The library currently includes:

- explicit TARGET calendar, day-count, stub, fixing and payment conventions;
- single- and multi-curve OIS/projection infrastructure;
- bootstrap from deposits, FRAs, OIS swaps and par swaps;
- compounded-RFR coupons with lookback, observation shift and lockout;
- IRS, caps/floors and European swaptions, including amortising notionals;
- Black-76, shifted-Black and Bachelier pricing/Greeks and implied-vol inversion;
- cap-vol to caplet-vol stripping and price-preserving quote-convention conversion;
- parallel/key-rate/factor DV01, curve convexity, bucketed Vega, carry/roll-down and
  second-order P&L explain;
- Vasicek/CIR risk-neutral calibration and physical-measure parameter estimation;
- Hull-White 1F curve-consistent option pricing and calibration;
- shifted SABR smile/cube calibration with admissibility diagnostics;
- synthetic negative-rate and multi-curve validation fixtures.

## Architecture

```text
src/fi_pricing/
  conventions.py       calendars, day counts, schedules
  term_structure.py    discount curves and deterministic rolls
  options.py           Black / shifted Black / Bachelier building blocks
  instruments/         swaps, caps/floors, swaptions
  bootstrap.py         single- and multi-curve construction
  curves.py            discount/projection CurveSet
  rfr.py               compounded overnight-rate mechanics
  volatility.py        IV inversion, surfaces, stripping/conversion
  risk.py              curve/vol/model risk and P&L explain
  models/short_rate.py Vasicek/CIR P/Q layer
  hull_white.py        Hull-White 1F
  sabr.py              SABR and swaption-vol cube
```

Legacy import paths such as `fi_pricing.core`, `fi_pricing.irs`,
`fi_pricing.cap_floor`, `fi_pricing.swaption` and `fi_pricing.short_rate` are thin
public facades over the focused implementation modules.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

Examples:

```bash
python examples/price_irs.py
python examples/price_cap.py
python examples/price_swaption.py
python examples/multi_curve.py
python examples/hull_white.py
python examples/sabr_cube.py
python examples/risk_analytics.py
```

## Validation philosophy

The public repository uses only deterministic synthetic fixtures. Validation is based on
closed-form identities, no-arbitrage/admissibility checks, round trips and invariants,
including:

- floating-leg telescoping in the single-curve setting;
- par-swap NPV approximately zero;
- cap-floor parity;
- key-rate DV01 reconciliation to parallel DV01;
- multi-curve quote repricing;
- RFR compounding telescoping when conventions permit it;
- implied-volatility price/vol round trips;
- Hull-White exact fit to the initial discount curve;
- synthetic Vasicek/CIR calibration and exact-transition estimation tests;
- synthetic SABR smile/cube recovery plus strike-admissibility diagnostics.

The current clean public suite contains **157 tests**.

## Provenance and data

This repository is an independent implementation based on public mathematical definitions
and market-convention specifications. It contains no third-party spreadsheet/codebase,
course file, proprietary market-data table, trade snapshot or cached third-party output.
All bundled market data and trades are generated synthetic examples.

See [`PROVENANCE.md`](PROVENANCE.md) and [`REFERENCES.md`](REFERENCES.md).

## Limitations

This is not a production valuation library. Important limitations include simplified
collateral/funding conventions, finite-difference risk, approximation choices for
backward-looking RFR option variance and frozen-forward IRR-annuity cash-settled
swaptions, and SABR/Hull-White modelling choices that require model-risk judgement.

## License

MIT. See `LICENSE`.
