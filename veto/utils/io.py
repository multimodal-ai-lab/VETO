import csv
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence


def save_dicts_to_csv(
    rows: List[Dict[str, Any]],
    filepath: Path,
    *,
    fieldnames: Sequence[str],
) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must be non-empty")
    cols = list(fieldnames)
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})


def load_csv_dicts(filepath: Path) -> List[Dict[str, Any]]:
    filepath = Path(filepath)
    if not filepath.is_file():
        return []
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_key(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
    if not vals:
        raise ValueError(f"No values for {key!r}")
    return float(statistics.mean(vals))


def format_duration(seconds: float) -> str:
    """Human-readable duration for logging (e.g. ``135.2s`` or ``2m 15s``)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"

