"""
Knowledge Base Model Management for sentence transformers.
"""

import os
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

from .schemas import TrainingPair, KnowledgeBaseItem, ModelInfo

logger = logging.getLogger(__name__)


class KBModelManager:
    """
    Manages the Knowledge Base SentenceTransformer model including
    loading, training, and inference operations.
    """

    def __init__(
        self,
        base_model_name: str = "all-mpnet-base-v2",
        models_dir: str = "models",
        device: Optional[str] = None,
    ):
        """
        Initialize the KB Model Manager.

        Args:
            base_model_name: Name of the base SentenceTransformer model
            models_dir: Directory to store models
            device: Device to run the model on (cuda/cpu)
        """
        self.base_model_name = base_model_name
        self.models_dir = Path(models_dir)
        self.trained_models_dir = self.models_dir / "trained"
        self.artifacts_dir = self.models_dir / "artifacts"
        
        # Create directories
        self.trained_models_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Thread safety
        self._model_lock = threading.Lock()
        self._model: Optional[SentenceTransformer] = None
        self._current_model_path: Optional[Path] = None
        
        # Disable HuggingFace progress bars for cleaner logs
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        
        # Load model at initialization
        self._load_model()

    def _load_model(self) -> None:
        """Load the latest available model or base model."""
        try:
            # Try to load the latest fine-tuned model
            latest_model_path = self._get_latest_model_path()
            
            if latest_model_path and latest_model_path.exists():
                model_path = latest_model_path
                logger.info(f"Loading fine-tuned model from: {model_path}")
            else:
                model_path = self.base_model_name
                logger.info(f"Loading base model: {model_path}")
            
            with self._model_lock:
                self._model = SentenceTransformer(str(model_path), device=self.device)
                self._current_model_path = model_path if isinstance(model_path, Path) else None
                
                # Health check
                _ = self._model.encode("health-check", convert_to_tensor=False)
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.exception("Failed to load model")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def _get_latest_model_path(self) -> Optional[Path]:
        """Get the path to the latest fine-tuned model."""
        if not self.trained_models_dir.exists():
            return None
            
        # Look for timestamped model directories
        model_dirs = [
            d for d in self.trained_models_dir.iterdir()
            if d.is_dir() and d.name.startswith("fine-tuned-")
        ]
        
        if not model_dirs:
            return None
            
        # Sort by directory name (timestamp) and return the latest
        latest_dir = sorted(model_dirs, key=lambda x: x.name, reverse=True)[0]
        return latest_dir

    def _get_latest_training_pairs(self) -> List[TrainingPair]:
        """Load the latest training pairs from disk."""
        pairs_files = list(self.trained_models_dir.glob("*/pairs.json"))
        
        if not pairs_files:
            return []
            
        # Get the latest pairs file
        latest_pairs_file = sorted(pairs_files, key=lambda x: x.parent.name, reverse=True)[0]
        
        try:
            with open(latest_pairs_file, "r", encoding="utf-8") as f:
                raw_pairs = json.load(f)
            return [TrainingPair(**pair) for pair in raw_pairs]
        except Exception as e:
            logger.warning(f"Failed to load training pairs from {latest_pairs_file}: {e}")
            return []

    def encode(self, texts: List[str], convert_to_tensor: bool = True) -> torch.Tensor:
        """
        Encode texts using the current model.

        Args:
            texts: List of texts to encode
            convert_to_tensor: Whether to return as tensor

        Returns:
            Encoded embeddings
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
            
        with self._model_lock:
            return self._model.encode(texts, convert_to_tensor=convert_to_tensor)

    def find_similar(
        self,
        query: str,
        corpus_texts: List[str],
        top_k: int = 1
    ) -> List[Tuple[int, float]]:
        """
        Find similar texts in the corpus for the given query.

        Args:
            query: Query text
            corpus_texts: List of corpus texts to search
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        if not corpus_texts:
            return []
            
        # Encode query and corpus
        query_embedding = self.encode([query], convert_to_tensor=True)
        corpus_embeddings = self.encode(corpus_texts, convert_to_tensor=True)
        
        # Calculate similarities
        similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        # Get top-k results
        top_results = torch.topk(similarities, k=min(top_k, len(corpus_texts)))
        
        return [
            (int(idx), float(score))
            for idx, score in zip(top_results.indices, top_results.values)
        ]

    def fine_tune(
        self,
        training_pairs: List[TrainingPair],
        epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        save_model: bool = True
    ) -> Optional[Path]:
        """
        Fine-tune the model with training pairs.

        Args:
            training_pairs: List of training pairs
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
            save_model: Whether to save the fine-tuned model

        Returns:
            Path to saved model if save_model=True, else None
        """
        if not training_pairs:
            raise ValueError("No training pairs provided")
            
        logger.info(f"Starting fine-tuning with {len(training_pairs)} pairs")
        
        try:
            with self._model_lock:
                # Prepare training data
                examples = [
                    InputExample(texts=[pair.input, pair.target], label=1.0)
                    for pair in training_pairs
                ]
                
                # Create data loader
                train_dataloader = DataLoader(
                    examples,
                    shuffle=True,
                    batch_size=batch_size,
                    collate_fn=self._model.smart_batching_collate
                )
                
                # Define loss function
                train_loss = losses.CosineSimilarityLoss(self._model)
                
                # Fine-tune the model
                self._model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=epochs,
                    optimizer_params={"lr": learning_rate},
                    show_progress_bar=False
                )
                
                # Save the model if requested
                if save_model:
                    output_path = self._save_fine_tuned_model(training_pairs)
                    
                    # Reload the model from the saved path
                    self._model = SentenceTransformer(str(output_path), device=self.device)
                    self._current_model_path = output_path
                    
                    logger.info(f"Fine-tuning completed and model saved to: {output_path}")
                    return output_path
                else:
                    logger.info("Fine-tuning completed (model not saved)")
                    return None
                    
        except Exception as e:
            logger.exception("Fine-tuning failed")
            raise RuntimeError(f"Fine-tuning failed: {str(e)}")

    def _save_fine_tuned_model(self, training_pairs: List[TrainingPair]) -> Path:
        """Save the fine-tuned model with metadata."""
        # Create timestamped directory
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        output_dir = self.trained_models_dir / f"fine-tuned-{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
        
        # Save the model
        self._model.save(str(output_dir))
        
        # Save training pairs
        pairs_file = output_dir / "pairs.json"
        with open(pairs_file, "w", encoding="utf-8") as f:
            json.dump([pair.dict() for pair in training_pairs], f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "base_model": self.base_model_name,
            "training_pairs_count": len(training_pairs),
            "device": self.device,
            "created_at": datetime.utcnow().isoformat()
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return output_dir

    def get_model_info(self) -> ModelInfo:
        """Get information about the current model."""
        model_path = str(self._current_model_path) if self._current_model_path else self.base_model_name
        
        # Try to get model size
        size_mb = None
        if self._current_model_path and self._current_model_path.exists():
            try:
                size_bytes = sum(
                    f.stat().st_size
                    for f in self._current_model_path.rglob("*")
                    if f.is_file()
                )
                size_mb = size_bytes / (1024 * 1024)
            except Exception:
                pass
        
        return ModelInfo(
            model_name=self.base_model_name,
            model_path=model_path,
            version="1.0.0",  # You might want to make this dynamic
            created_at=datetime.utcnow(),
            size_mb=size_mb,
            performance_metrics={}
        )

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def get_learned_pairs(self) -> List[TrainingPair]:
        """Get all learned training pairs."""
        return self._get_latest_training_pairs() 