"""Paired, task-stratified bootstrap between two run dirs (same protocol as sig_rv5.py).
usage: sig_pair.py RUN_A RUN_B [RUN_A RUN_B ...]   (prints A minus B)
"""
import sys
from sig_rv5 import load, paired  # noqa: E402

args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    print(f"{a} minus {b}:", paired(load(f"{a}/raw_predictions.jsonl"), load(f"{b}/raw_predictions.jsonl")))
