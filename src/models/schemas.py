"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List


class Query(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Query text (1-1000 characters)")


class TrainingPair(BaseModel):
    input: str
    target: str


class TrainingData(BaseModel):
    data: List[TrainingPair]


class KnowledgeBaseItem(BaseModel):
    description: str
    resolution: str 