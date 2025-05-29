"""
SentenceTransformer model management.
"""
import os
import json
import logging
import threading
from typing import List
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

from src.models.schemas import TrainingPair
from src.utils.file_utils import latest_run_dir, BASE_MODEL_DIR, _new_output_dir

# Set environment variables
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

logger = logging.getLogger(__name__)

# Thread safety for model access
model_lock = threading.Lock()

# Global model instance
model = None

# Initialize model at module load
try:
    load_path = latest_run_dir() or BASE_MODEL_DIR
    with model_lock:
        model = SentenceTransformer(str(load_path))
        _ = model.encode("health-check")
    logger.info(f"Model loaded from: {load_path}")
except Exception as e:
    logger.exception("Failed to load model")
    raise


def fine_tune(job_id: str, pairs: List[TrainingPair], jobs: dict):
    """Fine-tune the model with new training pairs."""
    global model
    try:
        jobs[job_id]["status"] = "running"
        with model_lock:
            examples = [InputExample(texts=[p.input, p.target], label=1.0) for p in pairs]
            loader = DataLoader(
                examples,
                shuffle=True,
                batch_size=8,
                collate_fn=model.smart_batching_collate
            )
            loss_fn = losses.CosineSimilarityLoss(model)
            model.fit([(loader, loss_fn)], epochs=1,
                      optimizer_params={"lr": 1e-5},
                      show_progress_bar=False)
            out_dir = _new_output_dir()
            model.save(str(out_dir))
            model = SentenceTransformer(str(out_dir))  # hot‑reload
            with open(out_dir / "pairs.json", "w", encoding="utf-8") as f:
                json.dump([p.dict() for p in pairs], f, ensure_ascii=False, indent=2)
        jobs[job_id] = {"status": "finished", "msg": f"saved to {out_dir}"}
    except Exception as e:
        logger.exception("Training failed")
        jobs[job_id] = {"status": "failed", "msg": str(e)} 