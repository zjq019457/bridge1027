# Portable source scope

This is a runtime source snapshot for the 2026-09-28 fresh BridgePack G1 reproduction.
It contains the BridgePack implementation, G1 runner, generation helper, LongBench scorer, and prompt/config files.
It excludes `.git`, model weights, HF cache, result trees, temporary caches, and logs.

The snapshot preserves the historical G1 generation behavior, including its historical prompt binding, because the Portable Golden V1 target is the fresh run rather than a protocol rewrite.
