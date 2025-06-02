"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class Query(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Query text (1-1000 characters)")


class TrainingPair(BaseModel):
    input: str
    target: str


class TrainingData(BaseModel):
    data: List[TrainingPair]


class KnowledgeBaseItem(BaseModel):
    # Primary fields (required)
    issue: str = Field(..., description="The problem or issue description")
    resolution: str = Field(..., description="The solution or resolution")
    
    # Optional fields for enhanced categorization
    category: str = Field(default="general", description="Issue category")
    tags: List[str] = Field(default_factory=list, description="Tags for better search")
    
    # Legacy field for backward compatibility
    description: Optional[str] = Field(default=None, description="Legacy description field")
    
    def __init__(self, **data):
        # Handle backward compatibility: if description is provided but not issue, use description as issue
        if 'description' in data and 'issue' not in data:
            data['issue'] = data['description']
        super().__init__(**data) 