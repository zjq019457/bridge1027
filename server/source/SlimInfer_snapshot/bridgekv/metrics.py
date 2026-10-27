import json
from pathlib import Path
from typing import Any, Dict


def write_bridgekv_summaries(out_dir: Path, summary: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "bridge_graph_summary.json", summary.get("graph_summary", {}))
    write_json(out_dir / "bridge_score_summary.json", summary.get("score_summary", {}))
    write_json(out_dir / "selected_blocks_summary.json", summary.get("selection_summary", {}))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
