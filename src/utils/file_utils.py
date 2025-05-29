"""
File and path utility functions.
"""
import pathlib
import json
from typing import Optional, List
from src.models.schemas import TrainingPair


# ======================================================================
# Paths & constants
# ======================================================================
LOCAL_MODELS_DIR = pathlib.Path(r"breakfix-kb-model")
BASE_MODEL_DIR = LOCAL_MODELS_DIR / "all-mpnet-base-v2"
RUNS_DIR = BASE_MODEL_DIR / "fine-tuned-runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LEGACY_DIR = BASE_MODEL_DIR / "fine-tuned"


def latest_run_dir() -> Optional[pathlib.Path]:
    """Pick latest fine-tuned directory (if any)."""
    candidates = sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()],
                        key=lambda p: p.name,
                        reverse=True)
    return candidates[0] if candidates else (LEGACY_DIR if LEGACY_DIR.exists() else None)


def latest_pairs_file() -> Optional[pathlib.Path]:
    """Get the latest pairs.json file."""
    files = sorted(RUNS_DIR.glob("*/pairs.json"), key=lambda p: p.parent.name, reverse=True)
    return files[0] if files else None


def load_pairs_from_disk() -> List[TrainingPair]:
    """Load training pairs from disk."""
    path = latest_pairs_file()
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [TrainingPair(**item) for item in raw]


def _new_output_dir() -> pathlib.Path:
    """Create a new timestamped output directory."""
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out = RUNS_DIR / f"fine-tuned-{ts}"
    out.mkdir(parents=True, exist_ok=False)   # fail fast on duplicate timestamp
    return out 