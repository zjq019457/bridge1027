# BridgePack Portable Reproduction V1

This directory is an independent, portable reproduction environment for the 2026-09-28 fresh BridgePack G1 run.

## Golden references

- Historical G1: macro 39.385; 1,150 raw predictions exist; historical packed/input/output token IDs were not saved.
- Portable Golden V1: macro 39.3517; fresh run IDs are saved; it agreed with historical decoded predictions for 1,141/1,150 rows.
- Portable target: 1,150/1,150 packed IDs, final input IDs, generated output IDs, and decoded predictions exact against Portable Golden V1.

Portable Golden token IDs must not be described as historical G1 token IDs. No method tuning was performed.

See `reports/PORTABLE_REPRO_DECISION.json` and `reports/PORTABLE_SAME_HOST_REPRO_REPORT.md`.
