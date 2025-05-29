"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel
from typing import List


class Query(BaseModel):
    text: str


class TrainingPair(BaseModel):
    input: str
    target: str


class TrainingData(BaseModel):
    data: List[TrainingPair]


class KnowledgeBaseItem(BaseModel):
    description: str
    resolution: str 