"""
Pydantic models for KB Nova Pipeline data structures.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Query(BaseModel):
    """Query model for troubleshooting requests."""
    text: str = Field(..., description="The query text to search for solutions")


class TrainingPair(BaseModel):
    """Training pair model for fine-tuning the model."""
    input: str = Field(..., description="Input text for training")
    target: str = Field(..., description="Target/expected output for training")


class TrainingData(BaseModel):
    """Training data collection model."""
    data: List[TrainingPair] = Field(..., description="List of training pairs")


class KnowledgeBaseItem(BaseModel):
    """Knowledge base item model."""
    description: str = Field(..., description="Problem description")
    resolution: str = Field(..., description="Solution or resolution")
    category: Optional[str] = Field(None, description="Category of the issue")
    tags: Optional[List[str]] = Field(default_factory=list, description="Tags for categorization")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class TroubleshootResponse(BaseModel):
    """Response model for troubleshooting queries."""
    query: str = Field(..., description="Original query text")
    response: str = Field(..., description="Recommended solution")
    similarity_score: float = Field(..., description="Similarity score between query and solution")
    confidence: Optional[str] = Field(None, description="Confidence level (high/medium/low)")
    source: Optional[str] = Field(None, description="Source of the solution (kb/learned)")


class TrainingJobStatus(BaseModel):
    """Training job status model."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (queued/running/finished/failed)")
    message: str = Field(default="", description="Status message or error details")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Job creation time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    training_pairs_count: Optional[int] = Field(None, description="Number of training pairs processed")


class TrainingJobResponse(BaseModel):
    """Response model for training job submission."""
    job_id: str = Field(..., description="Unique job identifier")
    note: str = Field(..., description="Confirmation message")
    pairs_count: int = Field(..., description="Number of training pairs accepted")


class ModelInfo(BaseModel):
    """Model information model."""
    model_name: str = Field(..., description="Name of the model")
    model_path: str = Field(..., description="Path to the model")
    version: str = Field(..., description="Model version")
    created_at: datetime = Field(..., description="Model creation timestamp")
    size_mb: Optional[float] = Field(None, description="Model size in MB")
    performance_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Performance metrics")


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_info: Optional[ModelInfo] = Field(None, description="Information about the loaded model")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")


class ChatMessage(BaseModel):
    """Chat message model."""
    content: str = Field(..., description="Message content")
    role: str = Field(..., description="Message role (user/assistant)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    session_id: Optional[str] = Field(None, description="Session ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Optional session ID")
    use_context: bool = Field(True, description="Whether to use knowledge base context")
    include_chat_history: bool = Field(True, description="Whether to include chat history in context")


class ChatResponse(BaseModel):
    """Chat response model."""
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="Original user message")
    response: str = Field(..., description="Assistant response")
    confidence: str = Field(..., description="Response confidence level")
    source: str = Field(..., description="Response source")
    context_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Context sources used")
    timestamp: str = Field(..., description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChatSession(BaseModel):
    """Chat session model."""
    session_id: str = Field(..., description="Session ID")
    created_at: str = Field(..., description="Session creation timestamp")
    updated_at: str = Field(..., description="Session last update timestamp")
    message_count: int = Field(..., description="Number of messages in session")


class ChromaStats(BaseModel):
    """Chroma database statistics model."""
    total_documents: int = Field(..., description="Total number of documents")
    knowledge_base_items: int = Field(..., description="Number of knowledge base items")
    chat_messages: int = Field(..., description="Number of chat messages")
    collection_name: str = Field(..., description="Collection name")
    persist_directory: str = Field(..., description="Persistence directory") 