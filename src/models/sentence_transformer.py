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
from src.utils.file_utils import latest_run_dir, BASE_MODEL_DIR, _new_output_dir, get_model_path

# Set environment variables for cleaner output
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# Don't force offline mode - let it download if needed
# os.environ["HF_HUB_OFFLINE"] = "1"

logger = logging.getLogger(__name__)

# Thread safety for model access
model_lock = threading.Lock()

# Global model instance
model = None

# Initialize model at module load
try:
    model_path = get_model_path()
    with model_lock:
        model = SentenceTransformer(model_path)
        # Test the model with a simple encode to ensure it works
        _ = model.encode("health-check")
    logger.info(f"Model loaded from: {model_path}")
except Exception as e:
    logger.error(f"Failed to load model from {get_model_path()}: {e}")
    logger.info("This is normal on first run - the model will be downloaded when the service starts")
    # Don't raise the exception - let the service handle model loading later
    model = None


class SentenceTransformerModel:
    """Wrapper for SentenceTransformer model with lazy loading"""
    
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
    
    def get_model(self) -> SentenceTransformer:
        """Get model instance with lazy loading"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    model_path = get_model_path()
                    self._model = SentenceTransformer(model_path)
                    logger.info(f"Lazy loaded model from: {model_path}")
        return self._model
    
    def encode(self, texts, **kwargs):
        """Encode texts using the model"""
        return self.get_model().encode(texts, **kwargs)


# Global model wrapper instance
model_wrapper = SentenceTransformerModel()

def get_model() -> SentenceTransformer:
    """Get the global model instance"""
    return model_wrapper.get_model()


def fine_tune(job_id: str, pairs: List[TrainingPair], jobs: dict):
    """Fine-tune the model with new training pairs."""
    global model
    try:
        jobs[job_id]["status"] = "running"
        current_model = get_model()  # Use the wrapper to get model
        
        with model_lock:
            examples = [InputExample(texts=[p.input, p.target], label=1.0) for p in pairs]
            loader = DataLoader(
                examples,
                shuffle=True,
                batch_size=8,
                collate_fn=current_model.smart_batching_collate
            )
            loss_fn = losses.CosineSimilarityLoss(current_model)
            current_model.fit([(loader, loss_fn)], epochs=1,
                      optimizer_params={"lr": 1e-5},
                      show_progress_bar=False)
            out_dir = _new_output_dir()
            current_model.save(str(out_dir))
            
            # Update the global model wrapper
            model_wrapper._model = SentenceTransformer(str(out_dir))  # hot‑reload
            model = model_wrapper._model  # Update legacy global for compatibility
            
            with open(out_dir / "pairs.json", "w", encoding="utf-8") as f:
                json.dump([p.dict() for p in pairs], f, ensure_ascii=False, indent=2)
        jobs[job_id] = {"status": "finished", "msg": f"saved to {out_dir}"}
    except Exception as e:
        logger.exception("Training failed")
        jobs[job_id] = {"status": "failed", "msg": str(e)} 