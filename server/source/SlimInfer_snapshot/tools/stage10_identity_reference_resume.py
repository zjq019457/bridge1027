#!/usr/bin/env python3
"""Resume Stage 10 with canonical unified input identity metadata.

Historical DRM prediction files predate the unified record schema: they retain
the correct prediction at each sample index but do not contain canonical prompt,
tokenization, generation, or runtime identity fields.  Stage 2 CC predictions
were produced under the same frozen task protocol and contain those fields for
all 3,950 samples.  This wrapper changes only the CPU-side validation reference;
DRM prediction and selection equivalence remain checked against the frozen DRM
artifacts by run_qwen_ablation_suite.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools import run_qwen_ablation_suite as suite


IDENTITY_FIX_ID = "stage10-unified-identity-reference-v1"


def unified_identity_reference(
    pipeline_dir: Path,
    family: str,
    task: str,
) -> Dict[int, Dict[str, Any]]:
    del family
    path = suite._canonical_prediction_path(
        pipeline_dir, "cc_sliminfer_v1", task
    )
    return {
        int(row["sample_index"]): row
        for row in suite._read_predictions(path)
    }


def install() -> None:
    suite._identity_reference = unified_identity_reference


def main() -> int:
    install()
    return suite.main()


if __name__ == "__main__":
    raise SystemExit(main())
