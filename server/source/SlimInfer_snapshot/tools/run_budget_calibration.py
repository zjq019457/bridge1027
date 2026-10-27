#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


runpy.run_path(str(Path(__file__).with_name("run_ab_budget_calibration_fix.py")), run_name="__main__")
