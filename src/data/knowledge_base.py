"""
Knowledge base data and management.
"""
from typing import List
from src.models.schemas import KnowledgeBaseItem, TrainingPair
from src.utils.file_utils import load_pairs_from_disk

# Simple in-memory KB (replace with DB later)
knowledge_base: List[KnowledgeBaseItem] = [
    KnowledgeBaseItem(description="npm ERR! code ERESOLVE", resolution="Delete node_modules & package-lock.json, then run npm install."),
    KnowledgeBaseItem(description="Script not running after package install", resolution="Check package.json scripts and dependencies."),
    KnowledgeBaseItem(description="npm install hangs", resolution="Clear npm cache (npm cache clean --force) or check network."),
    KnowledgeBaseItem(description="Update npm version", resolution="Run npm install -g npm@latest."),
]

# Build learned_pairs from disk
defined_pairs: List[TrainingPair] = load_pairs_from_disk()
learned_pairs = defined_pairs 