"""
Training service for managing background fine-tuning jobs.
"""

import uuid
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from ..models.schemas import TrainingPair, TrainingJobStatus
from ..models.kb_model import KBModelManager

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Service for managing background training jobs.
    Handles job queuing, execution, and status tracking.
    """

    def __init__(self, model_manager: KBModelManager, max_workers: int = 2):
        """
        Initialize the Training Service.

        Args:
            model_manager: KB Model Manager instance
            max_workers: Maximum number of concurrent training jobs
        """
        self.model_manager = model_manager
        self.max_workers = max_workers
        
        # Job tracking
        self._jobs: Dict[str, TrainingJobStatus] = {}
        self._jobs_lock = threading.Lock()
        
        # Thread pool for background training
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"Training service initialized with {max_workers} workers")

    def submit_training_job(
        self,
        training_pairs: List[TrainingPair],
        epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 1e-5
    ) -> str:
        """
        Submit a new training job.

        Args:
            training_pairs: List of training pairs
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate

        Returns:
            Job ID for tracking the job
        """
        if not training_pairs:
            raise ValueError("No training pairs provided")

        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job status
        job_status = TrainingJobStatus(
            job_id=job_id,
            status="queued",
            message="Job queued for processing",
            training_pairs_count=len(training_pairs)
        )
        
        # Store job status
        with self._jobs_lock:
            self._jobs[job_id] = job_status
        
        # Submit job to thread pool
        future = self._executor.submit(
            self._execute_training_job,
            job_id,
            training_pairs,
            epochs,
            batch_size,
            learning_rate
        )
        
        # Add callback to handle completion
        future.add_done_callback(lambda f: self._handle_job_completion(job_id, f))
        
        logger.info(f"Training job {job_id} submitted with {len(training_pairs)} pairs")
        return job_id

    def get_job_status(self, job_id: str) -> Optional[TrainingJobStatus]:
        """
        Get the status of a training job.

        Args:
            job_id: Job ID

        Returns:
            Job status or None if job not found
        """
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def get_all_jobs(self) -> Dict[str, TrainingJobStatus]:
        """Get all training jobs."""
        with self._jobs_lock:
            return self._jobs.copy()

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a training job (if it's still queued).

        Args:
            job_id: Job ID

        Returns:
            True if job was cancelled, False otherwise
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job and job.status == "queued":
                job.status = "cancelled"
                job.message = "Job cancelled by user"
                job.completed_at = datetime.utcnow()
                logger.info(f"Training job {job_id} cancelled")
                return True
        return False

    def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """
        Clean up completed jobs older than specified age.

        Args:
            max_age_hours: Maximum age in hours for completed jobs

        Returns:
            Number of jobs cleaned up
        """
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0
        
        with self._jobs_lock:
            jobs_to_remove = []
            for job_id, job in self._jobs.items():
                if (job.status in ["finished", "failed", "cancelled"] and
                    job.completed_at and
                    job.completed_at.timestamp() < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self._jobs[job_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old training jobs")
        
        return cleaned_count

    def _execute_training_job(
        self,
        job_id: str,
        training_pairs: List[TrainingPair],
        epochs: int,
        batch_size: int,
        learning_rate: float
    ) -> None:
        """
        Execute a training job.

        Args:
            job_id: Job ID
            training_pairs: Training pairs
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        try:
            # Update job status to running
            self._update_job_status(job_id, "running", "Training in progress...")
            
            logger.info(f"Starting training job {job_id}")
            
            # Perform fine-tuning
            model_path = self.model_manager.fine_tune(
                training_pairs=training_pairs,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_model=True
            )
            
            # Update job status to finished
            message = f"Training completed successfully. Model saved to: {model_path}"
            self._update_job_status(job_id, "finished", message)
            
            logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            # Update job status to failed
            error_message = f"Training failed: {str(e)}"
            self._update_job_status(job_id, "failed", error_message)
            
            logger.exception(f"Training job {job_id} failed")

    def _update_job_status(self, job_id: str, status: str, message: str) -> None:
        """Update job status."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                job.message = message
                if status in ["finished", "failed", "cancelled"]:
                    job.completed_at = datetime.utcnow()

    def _handle_job_completion(self, job_id: str, future) -> None:
        """Handle job completion callback."""
        try:
            # Get any exception that occurred
            exception = future.exception()
            if exception:
                logger.error(f"Training job {job_id} failed with exception: {exception}")
        except Exception as e:
            logger.error(f"Error handling job completion for {job_id}: {e}")

    def get_training_stats(self) -> Dict[str, int]:
        """Get training statistics."""
        with self._jobs_lock:
            stats = {
                "total_jobs": len(self._jobs),
                "queued": 0,
                "running": 0,
                "finished": 0,
                "failed": 0,
                "cancelled": 0
            }
            
            for job in self._jobs.values():
                if job.status in stats:
                    stats[job.status] += 1
            
            return stats

    def shutdown(self) -> None:
        """Shutdown the training service."""
        logger.info("Shutting down training service...")
        self._executor.shutdown(wait=True)
        logger.info("Training service shutdown complete") 