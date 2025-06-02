"""
Simplified Knowledge Base Service with ChromaDB Scale Support
Consolidates complex /src implementation into maintainable, production-ready code
"""

import os
import json
import uuid
import logging
import threading
import pathlib
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

# Core dependencies
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import chromadb
from chromadb.config import Settings
import numpy as np

# ======================================================================
# Configuration
# ======================================================================
class Config:
    """Simple configuration using environment variables"""
    MODEL_NAME = os.getenv("MODEL_NAME", "all-mpnet-base-v2")
    MODELS_DIR = pathlib.Path(os.getenv("MODELS_DIR", "models"))
    CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chroma_db")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    HOT_CACHE_SIZE = int(os.getenv("HOT_CACHE_SIZE", "100"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ======================================================================
# Logging setup
# ======================================================================
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ======================================================================
# Unified Data Models
# ======================================================================
class Query(BaseModel):
    text: str

class KnowledgeItem(BaseModel):
    """Unified model for both training and knowledge storage"""
    problem: str
    solution: str
    category: str = "general"
    source: str = "manual"  # "manual", "trained", "imported"
    tags: List[str] = []

class TrainingData(BaseModel):
    data: List[KnowledgeItem]

class Document(BaseModel):
    """For bulk document ingestion"""
    title: str
    content: str
    category: str = "general"
    source: str = "manual"

# ======================================================================
# Scalable Knowledge Base with Hybrid Storage
# ======================================================================
class ScalableKnowledgeBase:
    """Hybrid storage: in-memory hot cache + ChromaDB for scale"""
    
    def __init__(self):
        self.model_lock = threading.Lock()
        self.model = None
        self.hot_cache: List[KnowledgeItem] = []
        self.hot_cache_embeddings = None
        self.chroma_client = None
        self.collection = None
        
        # Job tracking for training
        self.training_jobs: Dict[str, Dict] = {}
        
        # Setup directories
        Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.runs_dir = Config.MODELS_DIR / "fine-tuned-runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize(self):
        """Initialize model and ChromaDB"""
        try:
            # Load model (latest fine-tuned or base)
            model_path = self._get_latest_model_path()
            with self.model_lock:
                self.model = SentenceTransformer(str(model_path))
                # Test model
                _ = self.model.encode("health check")
            logger.info(f"Model loaded from: {model_path}")
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=Config.CHROMADB_PATH)
            self.collection = self.chroma_client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized")
            
            # Load existing training data into hot cache
            self._load_hot_cache()
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def _get_latest_model_path(self) -> pathlib.Path:
        """Get latest fine-tuned model or base model"""
        candidates = sorted([p for p in self.runs_dir.iterdir() if p.is_dir()],
                          key=lambda p: p.name, reverse=True)
        return candidates[0] if candidates else Config.MODEL_NAME
    
    def _load_hot_cache(self):
        """Load recent training data into hot cache"""
        try:
            pairs_files = sorted(self.runs_dir.glob("*/pairs.json"), 
                               key=lambda p: p.parent.name, reverse=True)
            
            if pairs_files:
                with open(pairs_files[0], "r", encoding="utf-8") as f:
                    raw_pairs = json.load(f)
                
                # Convert old format to new unified format
                for pair in raw_pairs[:Config.HOT_CACHE_SIZE]:
                    if "input" in pair and "target" in pair:
                        # Old TrainingPair format
                        item = KnowledgeItem(
                            problem=pair["input"],
                            solution=pair["target"],
                            source="trained"
                        )
                    else:
                        # New format
                        item = KnowledgeItem(**pair)
                    
                    self.hot_cache.append(item)
                
                # Pre-compute embeddings for hot cache
                if self.hot_cache:
                    with self.model_lock:
                        problems = [item.problem for item in self.hot_cache]
                        self.hot_cache_embeddings = self.model.encode(problems)
                    
                    logger.info(f"Hot cache loaded with {len(self.hot_cache)} items")
                    
        except Exception as e:
            logger.warning(f"Could not load hot cache: {e}")
    
    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search with hot cache fallback to ChromaDB"""
        try:
            # Try hot cache first (ultra-fast)
            if self.hot_cache_embeddings is not None:
                with self.model_lock:
                    query_emb = self.model.encode(query)
                
                scores = self._cosine_similarity(query_emb, self.hot_cache_embeddings)
                best_idx = int(scores.argmax())
                best_score = float(scores[best_idx])
                
                if best_score > 0.7:  # High confidence hit
                    best_item = self.hot_cache[best_idx]
                    return {
                        "problem": query,
                        "solution": best_item.solution,
                        "similarity_score": best_score,
                        "source": "hot_cache",
                        "category": best_item.category
                    }
            
            # Fallback to ChromaDB for comprehensive search
            return self._search_chromadb(query, top_k)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(500, f"Search error: {str(e)}")
    
    def _search_chromadb(self, query: str, top_k: int) -> Dict[str, Any]:
        """Search ChromaDB for comprehensive results"""
        try:
            with self.model_lock:
                query_emb = self.model.encode(query)
            
            results = self.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'][0]:
                return {
                    "problem": query,
                    "solution": "No relevant solutions found. Try rephrasing your query.",
                    "similarity_score": 0.0,
                    "source": "no_results",
                    "category": "system"
                }
            
            # Return best result
            best_doc = results['documents'][0][0]
            best_metadata = results['metadatas'][0][0]
            best_distance = results['distances'][0][0]
            
            return {
                "problem": query,
                "solution": best_doc,
                "similarity_score": 1 - best_distance,
                "source": "chromadb",
                "category": best_metadata.get("category", "general"),
                "title": best_metadata.get("title", "Unknown")
            }
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            # Fallback to empty result
            return {
                "problem": query,
                "solution": "Search temporarily unavailable. Please try again.",
                "similarity_score": 0.0,
                "source": "error",
                "category": "system"
            }
    
    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Add large documents with chunking"""
        try:
            all_chunks = []
            all_metadata = []
            all_ids = []
            
            for doc in documents:
                chunks = self._chunk_document(doc.content)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc.title}_{i}_{uuid.uuid4().hex[:8]}"
                    all_chunks.append(chunk)
                    all_metadata.append({
                        "title": doc.title,
                        "category": doc.category,
                        "source": doc.source,
                        "chunk_index": i
                    })
                    all_ids.append(chunk_id)
            
            # Batch encode and store
            with self.model_lock:
                embeddings = self.model.encode(all_chunks)
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=all_chunks,
                metadatas=all_metadata,
                ids=all_ids
            )
            
            logger.info(f"Added {len(documents)} documents ({len(all_chunks)} chunks)")
            
            return {
                "documents_added": len(documents),
                "chunks_created": len(all_chunks),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Document addition failed: {e}")
            raise HTTPException(500, f"Failed to add documents: {str(e)}")
    
    def train_model(self, job_id: str, training_data: List[KnowledgeItem]):
        """Background training with job tracking"""
        try:
            self.training_jobs[job_id]["status"] = "running"
            
            with self.model_lock:
                # Convert to training examples
                examples = [
                    InputExample(texts=[item.problem, item.solution], label=1.0)
                    for item in training_data
                ]
                
                # Create data loader
                loader = DataLoader(
                    examples,
                    shuffle=True,
                    batch_size=8,
                    collate_fn=self.model.smart_batching_collate
                )
                
                # Train model
                loss_fn = losses.CosineSimilarityLoss(self.model)
                self.model.fit(
                    [(loader, loss_fn)],
                    epochs=1,
                    optimizer_params={"lr": 1e-5},
                    show_progress_bar=False
                )
                
                # Save model
                timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                output_dir = self.runs_dir / f"fine-tuned-{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=False)
                
                self.model.save(str(output_dir))
                
                # Save training data
                with open(output_dir / "pairs.json", "w", encoding="utf-8") as f:
                    json.dump([item.dict() for item in training_data], f, 
                            ensure_ascii=False, indent=2)
                
                # Hot reload model
                self.model = SentenceTransformer(str(output_dir))
                
                # Update hot cache
                self.hot_cache.extend(training_data[-Config.HOT_CACHE_SIZE:])
                self.hot_cache = self.hot_cache[-Config.HOT_CACHE_SIZE:]  # Keep recent
                
                # Rebuild hot cache embeddings
                if self.hot_cache:
                    problems = [item.problem for item in self.hot_cache]
                    self.hot_cache_embeddings = self.model.encode(problems)
            
            self.training_jobs[job_id] = {
                "status": "completed",
                "message": f"Model saved to {output_dir}",
                "completion_time": datetime.now().isoformat()
            }
            
            logger.info(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            self.training_jobs[job_id] = {
                "status": "failed",
                "message": str(e),
                "completion_time": datetime.now().isoformat()
            }
    
    def _chunk_document(self, content: str) -> List[str]:
        """Simple sentence-based chunking"""
        sentences = content.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < Config.CHUNK_SIZE:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _cosine_similarity(self, a, b):
        """Fast cosine similarity calculation"""
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))

# ======================================================================
# Global instance
# ======================================================================
kb = ScalableKnowledgeBase()

# ======================================================================
# FastAPI Application
# ======================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    logger.info("Starting Simplified Knowledge Base Service...")
    try:
        kb.initialize()
        logger.info("✅ Service initialized successfully")
        yield
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise
    finally:
        logger.info("🛑 Service shutting down...")

app = FastAPI(
    title="Simplified Knowledge Base Service",
    description="Production-ready knowledge base with ChromaDB scale support",
    version="3.0.0",
    lifespan=lifespan
)

# ======================================================================
# Core API Endpoints
# ======================================================================
@app.post("/troubleshoot")
def troubleshoot(query: Query):
    """Search for solutions to problems"""
    try:
        result = kb.search(query.text)
        return result
    except Exception as e:
        logger.error(f"Troubleshoot failed: {e}")
        raise HTTPException(500, f"Search failed: {str(e)}")

@app.post("/train")
def start_training(training_data: TrainingData, background_tasks: BackgroundTasks):
    """Start model fine-tuning"""
    try:
        if not training_data.data:
            raise HTTPException(400, "No training data provided")
        
        job_id = str(uuid.uuid4())
        kb.training_jobs[job_id] = {
            "status": "queued",
            "start_time": datetime.now().isoformat(),
            "training_items_count": len(training_data.data)
        }
        
        background_tasks.add_task(kb.train_model, job_id, training_data.data)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": f"Training started with {len(training_data.data)} items"
        }
        
    except Exception as e:
        logger.error(f"Training start failed: {e}")
        raise HTTPException(500, f"Failed to start training: {str(e)}")

@app.get("/train/{job_id}")
def get_training_status(job_id: str):
    """Check training job status"""
    if job_id not in kb.training_jobs:
        raise HTTPException(404, "Training job not found")
    
    return kb.training_jobs[job_id]

@app.post("/documents/bulk")
def add_documents(documents: List[Document]):
    """Add large document collections"""
    try:
        result = kb.add_documents(documents)
        return result
    except Exception as e:
        logger.error(f"Document addition failed: {e}")
        raise HTTPException(500, f"Failed to add documents: {str(e)}")

# ======================================================================
# Health endpoint (minimal)
# ======================================================================
@app.get("/")
def root():
    """Basic service info"""
    return {
        "service": "Simplified Knowledge Base",
        "version": "3.0.0",
        "status": "healthy",
        "hot_cache_size": len(kb.hot_cache),
        "endpoints": ["/troubleshoot", "/train", "/documents/bulk"]
    }

# ======================================================================
# Development server
# ======================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False) 