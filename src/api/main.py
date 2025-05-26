"""
Main FastAPI application for KB Nova Pipeline Models.
"""

import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..models.schemas import (
    Query, TrainingData, TrainingJobResponse, TrainingJobStatus,
    TroubleshootResponse, HealthCheck, KnowledgeBaseItem,
    ChatRequest, ChatResponse, ChatSession, ChromaStats
)
from ..models.kb_model import KBModelManager
from ..data.knowledge_base import KnowledgeBaseService
from ..data.chroma_service import ChromaService
from ..training.trainer import TrainingService
from ..inference.troubleshoot import TroubleshootService
from ..api.chat import ChatService
from ..utils.config import get_settings
from ..utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global variables for services
model_manager: KBModelManager = None
kb_service: KnowledgeBaseService = None
chroma_service: ChromaService = None
training_service: TrainingService = None
troubleshoot_service: TroubleshootService = None
chat_service: ChatService = None
app_start_time: float = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global model_manager, kb_service, chroma_service, training_service, troubleshoot_service, chat_service, app_start_time
    
    logger.info("Starting KB Nova Pipeline Models API...")
    app_start_time = time.time()
    
    try:
        # Load configuration
        settings = get_settings()
        
        # Initialize services
        logger.info("Initializing services...")
        
        # Initialize model manager
        model_manager = KBModelManager(
            base_model_name=settings.model.architecture,
            models_dir=settings.paths.models
        )
        
        # Initialize knowledge base service
        kb_service = KnowledgeBaseService(data_dir="data")
        
        # Initialize Chroma service
        chroma_service = ChromaService(
            model_manager=model_manager,
            persist_directory="data/chroma",
            collection_name="kb_documents"
        )
        
        # Initialize existing knowledge base items in Chroma
        kb_items = kb_service.get_all_items()
        if kb_items:
            chroma_service.add_knowledge_base_items(kb_items)
            logger.info(f"Loaded {len(kb_items)} knowledge base items into Chroma")
        
        # Initialize training service
        training_service = TrainingService(
            model_manager=model_manager,
            max_workers=2
        )
        
        # Initialize troubleshoot service
        troubleshoot_service = TroubleshootService(
            model_manager=model_manager,
            kb_service=kb_service,
            chroma_service=chroma_service,
            confidence_threshold=0.5
        )
        
        # Initialize chat service
        chat_service = ChatService(
            chroma_service=chroma_service,
            troubleshoot_service=troubleshoot_service,
            max_context_messages=10,
            context_similarity_threshold=0.7
        )
        
        logger.info("All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.exception("Failed to initialize services")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down services...")
        if training_service:
            training_service.shutdown()
        logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="KB Nova Pipeline Models",
    description="AI/ML Pipeline for Knowledge Base Processing and Model Training",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get services
def get_model_manager() -> KBModelManager:
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return model_manager


def get_kb_service() -> KnowledgeBaseService:
    if kb_service is None:
        raise HTTPException(status_code=503, detail="Knowledge base service not initialized")
    return kb_service


def get_training_service() -> TrainingService:
    if training_service is None:
        raise HTTPException(status_code=503, detail="Training service not initialized")
    return training_service


def get_troubleshoot_service() -> TroubleshootService:
    if troubleshoot_service is None:
        raise HTTPException(status_code=503, detail="Troubleshoot service not initialized")
    return troubleshoot_service


def get_chroma_service() -> ChromaService:
    if chroma_service is None:
        raise HTTPException(status_code=503, detail="Chroma service not initialized")
    return chroma_service


def get_chat_service() -> ChatService:
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    return chat_service


# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check(
    model_mgr: KBModelManager = Depends(get_model_manager)
):
    """Health check endpoint."""
    try:
        uptime = time.time() - app_start_time if app_start_time else 0
        model_info = model_mgr.get_model_info()
        
        return HealthCheck(
            status="healthy",
            model_loaded=model_mgr.is_model_loaded(),
            model_info=model_info,
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.exception("Health check failed")
        return HealthCheck(
            status="unhealthy",
            model_loaded=False,
            uptime_seconds=0
        )


# Training endpoints
@app.post("/train", response_model=TrainingJobResponse)
async def submit_training_job(
    payload: TrainingData,
    trainer: TrainingService = Depends(get_training_service)
):
    """Submit a training job."""
    try:
        if not payload.data:
            raise HTTPException(status_code=400, detail="No training data provided")
        
        job_id = trainer.submit_training_job(payload.data)
        
        return TrainingJobResponse(
            job_id=job_id,
            note=f"Training job submitted successfully",
            pairs_count=len(payload.data)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to submit training job")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/train/{job_id}", response_model=TrainingJobStatus)
async def get_training_job_status(
    job_id: str,
    trainer: TrainingService = Depends(get_training_service)
):
    """Get training job status."""
    job_status = trainer.get_job_status(job_id)
    if job_status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_status


@app.get("/train", response_model=Dict[str, TrainingJobStatus])
async def get_all_training_jobs(
    trainer: TrainingService = Depends(get_training_service)
):
    """Get all training jobs."""
    return trainer.get_all_jobs()


@app.delete("/train/{job_id}")
async def cancel_training_job(
    job_id: str,
    trainer: TrainingService = Depends(get_training_service)
):
    """Cancel a training job."""
    success = trainer.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
    return {"message": "Job cancelled successfully"}


# Troubleshooting endpoints
@app.post("/troubleshoot", response_model=TroubleshootResponse)
async def troubleshoot_query(
    query: Query,
    ts_service: TroubleshootService = Depends(get_troubleshoot_service)
):
    """Process a troubleshooting query."""
    try:
        return ts_service.troubleshoot(query)
    except Exception as e:
        logger.exception("Troubleshooting failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/troubleshoot/similar", response_model=List[TroubleshootResponse])
async def get_similar_solutions(
    query: Query,
    top_k: int = 5,
    ts_service: TroubleshootService = Depends(get_troubleshoot_service)
):
    """Get multiple similar solutions for a query."""
    try:
        if top_k < 1 or top_k > 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
        
        return ts_service.get_similar_solutions(query, top_k=top_k)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get similar solutions")
        raise HTTPException(status_code=500, detail="Internal server error")


# Knowledge base endpoints
@app.get("/kb", response_model=List[KnowledgeBaseItem])
async def get_knowledge_base(
    kb_svc: KnowledgeBaseService = Depends(get_kb_service)
):
    """Get all knowledge base items."""
    return kb_svc.get_all_items()


@app.post("/kb")
async def add_knowledge_base_item(
    item: KnowledgeBaseItem,
    kb_svc: KnowledgeBaseService = Depends(get_kb_service)
):
    """Add a new knowledge base item."""
    try:
        kb_svc.add_item(item)
        return {"message": "Knowledge base item added successfully"}
    except Exception as e:
        logger.exception("Failed to add knowledge base item")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/kb/categories")
async def get_kb_categories(
    kb_svc: KnowledgeBaseService = Depends(get_kb_service)
):
    """Get all knowledge base categories."""
    return {"categories": kb_svc.get_categories()}


@app.get("/kb/tags")
async def get_kb_tags(
    kb_svc: KnowledgeBaseService = Depends(get_kb_service)
):
    """Get all knowledge base tags."""
    return {"tags": kb_svc.get_tags()}


@app.get("/kb/stats")
async def get_kb_stats(
    kb_svc: KnowledgeBaseService = Depends(get_kb_service)
):
    """Get knowledge base statistics."""
    return kb_svc.get_stats()


# Statistics and monitoring endpoints
@app.get("/stats")
async def get_system_stats(
    trainer: TrainingService = Depends(get_training_service),
    ts_service: TroubleshootService = Depends(get_troubleshoot_service),
    kb_svc: KnowledgeBaseService = Depends(get_kb_service),
    chat_svc: ChatService = Depends(get_chat_service),
    chroma_svc: ChromaService = Depends(get_chroma_service)
):
    """Get system statistics."""
    try:
        return {
            "training_stats": trainer.get_training_stats(),
            "corpus_stats": ts_service.get_corpus_stats(),
            "kb_stats": kb_svc.get_stats(),
            "chat_stats": chat_svc.get_chat_stats(),
            "chroma_stats": chroma_svc.get_collection_stats(),
            "uptime_seconds": time.time() - app_start_time if app_start_time else 0
        }
    except Exception as e:
        logger.exception("Failed to get system stats")
        raise HTTPException(status_code=500, detail="Internal server error")


# Model information endpoint
@app.get("/model/info")
async def get_model_info(
    model_mgr: KBModelManager = Depends(get_model_manager)
):
    """Get model information."""
    try:
        return model_mgr.get_model_info()
    except Exception as e:
        logger.exception("Failed to get model info")
        raise HTTPException(status_code=500, detail="Internal server error")


# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """Process a chat message."""
    try:
        response = chat_svc.chat(
            message=request.message,
            session_id=request.session_id,
            use_context=request.use_context,
            include_chat_history=request.include_chat_history
        )
        return ChatResponse(**response)
    except Exception as e:
        logger.exception("Chat processing failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/chat/session")
async def create_chat_session(
    chat_svc: ChatService = Depends(get_chat_service)
):
    """Create a new chat session."""
    try:
        session_id = chat_svc.create_session()
        return {"session_id": session_id, "message": "Chat session created successfully"}
    except Exception as e:
        logger.exception("Failed to create chat session")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/chat/session/{session_id}/history")
async def get_chat_history(
    session_id: str,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """Get chat history for a session."""
    try:
        history = chat_svc.get_session_history(session_id)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        logger.exception("Failed to get chat history")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/chat/session/{session_id}")
async def clear_chat_session(
    session_id: str,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """Clear a chat session."""
    try:
        success = chat_svc.clear_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Chat session cleared successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to clear chat session")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/chat/sessions", response_model=List[ChatSession])
async def get_all_chat_sessions(
    chat_svc: ChatService = Depends(get_chat_service)
):
    """Get all active chat sessions."""
    try:
        sessions = chat_svc.get_all_sessions()
        return [ChatSession(**session) for session in sessions]
    except Exception as e:
        logger.exception("Failed to get chat sessions")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/chat/search")
async def search_conversations(
    query: str,
    session_id: Optional[str] = None,
    limit: int = 10,
    chat_svc: ChatService = Depends(get_chat_service)
):
    """Search across chat conversations."""
    try:
        if limit < 1 or limit > 50:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 50")
        
        results = chat_svc.search_conversations(
            query=query,
            session_id=session_id,
            limit=limit
        )
        return {"query": query, "results": results}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to search conversations")
        raise HTTPException(status_code=500, detail="Internal server error")


# Chroma endpoints
@app.get("/chroma/stats", response_model=ChromaStats)
async def get_chroma_stats(
    chroma_svc: ChromaService = Depends(get_chroma_service)
):
    """Get Chroma database statistics."""
    try:
        stats = chroma_svc.get_collection_stats()
        return ChromaStats(**stats)
    except Exception as e:
        logger.exception("Failed to get Chroma stats")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/chroma/sync-kb")
async def sync_knowledge_base_to_chroma(
    chroma_svc: ChromaService = Depends(get_chroma_service),
    kb_svc: KnowledgeBaseService = Depends(get_kb_service)
):
    """Sync knowledge base items to Chroma."""
    try:
        # Clear existing KB items and re-add them
        kb_items = kb_svc.get_all_items()
        chroma_svc.add_knowledge_base_items(kb_items)
        return {"message": f"Synced {len(kb_items)} knowledge base items to Chroma"}
    except Exception as e:
        logger.exception("Failed to sync knowledge base to Chroma")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/chroma/clear")
async def clear_chroma_collection(
    chroma_svc: ChromaService = Depends(get_chroma_service)
):
    """Clear all documents from Chroma collection."""
    try:
        chroma_svc.clear_collection()
        return {"message": "Chroma collection cleared successfully"}
    except Exception as e:
        logger.exception("Failed to clear Chroma collection")
        raise HTTPException(status_code=500, detail="Internal server error")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "KB Nova Pipeline Models API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 