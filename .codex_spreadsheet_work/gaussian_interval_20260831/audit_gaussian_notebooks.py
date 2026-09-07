from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("/Users/yimingzang/Documents/Project/benchmark2")
NOTEBOOK_DIR = ROOT / "benchmark" / "examples" / "gaussian" / "notebooks"
CONSUMER_MARKERS = (
    "load_cached_metric_frames",
    "load_comparison_data",
    "run_summary_dimension_comparison",
    "run_mean_loess_comparison",
)
BANNED = re.compile(
    r"central\s+95%|q2\.5|q97\.5|signed_error_coverage\s*=\s*0\.95|"
    r"quantile\s*\(\s*0\.025|quantile\s*\(\s*0\.975",
    re.IGNORECASE,
)


def code_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


consumers = []
hardcoded_old_intervals = []
for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
    code = code_text(path)
    markers = [marker for marker in CONSUMER_MARKERS if marker in code]
    if markers:
        consumers.append({"notebook": path.name, "loaders": markers})
    matches = sorted(set(match.group(0) for match in BANNED.finditer(code)))
    if matches:
        hardcoded_old_intervals.append({"notebook": path.name, "matches": matches})

result = {
    "plotting_notebooks_using_shared_loaders": len(consumers),
    "consumers": consumers,
    "hardcoded_old_interval_matches": hardcoded_old_intervals,
}
(ROOT / ".codex_spreadsheet_work" / "gaussian_interval_20260831" / "notebook_audit.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result))
