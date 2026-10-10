# Public-release checklist

Before the first public push:

- create a **new** GitHub repository; do not import private prototype history;
- verify that no spreadsheet, `.mat`, raw market snapshot, screenshot or private fixture is present;
- run `grep`/repository search for private source names, worksheet names and legacy provenance strings;
- run `pytest` and `ruff check .` through CI;
- keep bundled data synthetic and deterministic;
- keep public validation based on formulas, invariants and synthetic round trips;
- publish only the clean v1.x history.

Suggested first commands:

```bash
git init
git add .
git commit -m "Initial public release"
```
