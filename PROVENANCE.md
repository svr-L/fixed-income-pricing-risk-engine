# Provenance

The public v1.0 codebase is an independent implementation of standard fixed-income
mathematics and market conventions.

## What is included

- source code written from public mathematical/model definitions and market conventions;
- deterministic synthetic curves, volatility surfaces and trade examples;
- regression, property and invariance tests against those synthetic fixtures and analytic
  identities.

## What is not included

- third-party spreadsheets, VBA or other source code;
- screenshots, worksheet/cell references or translated comments from third-party code;
- third-party trade examples, historical market-data tables or cached numerical outputs;
- private development history predating this clean public codebase.

The public repository should be initialized with a fresh Git history. Earlier private
prototypes are not part of this release.

## Validation

The clean implementation is validated through public mathematical identities and synthetic
round trips rather than by treating a private legacy implementation as the pricing oracle.
Where numerical regression values are retained, the underlying inputs are synthetic and the
same functionality is also covered by independent invariance/closed-form tests.
