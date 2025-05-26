"""
Knowledge Base service for managing static knowledge base items.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..models.schemas import KnowledgeBaseItem

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """
    Service for managing knowledge base items.
    Handles loading, saving, and querying static knowledge base data.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the Knowledge Base Service.

        Args:
            data_dir: Directory to store knowledge base data
        """
        self.data_dir = Path(data_dir)
        self.kb_file = self.data_dir / "processed" / "knowledge_base.json"
        
        # Ensure directories exist
        self.kb_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load knowledge base
        self._knowledge_base: List[KnowledgeBaseItem] = []
        self._load_knowledge_base()

    def _load_knowledge_base(self) -> None:
        """Load knowledge base from file or initialize with default data."""
        if self.kb_file.exists():
            try:
                with open(self.kb_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._knowledge_base = [KnowledgeBaseItem(**item) for item in data]
                logger.info(f"Loaded {len(self._knowledge_base)} knowledge base items")
            except Exception as e:
                logger.warning(f"Failed to load knowledge base from {self.kb_file}: {e}")
                self._initialize_default_kb()
        else:
            self._initialize_default_kb()

    def _initialize_default_kb(self) -> None:
        """Initialize with default knowledge base items."""
        default_items = [
            KnowledgeBaseItem(
                description="npm ERR! code ERESOLVE",
                resolution="Delete node_modules & package-lock.json, then run npm install.",
                category="npm",
                tags=["npm", "dependency", "error"]
            ),
            KnowledgeBaseItem(
                description="Script not running after package install",
                resolution="Check package.json scripts and dependencies.",
                category="npm",
                tags=["npm", "scripts", "package.json"]
            ),
            KnowledgeBaseItem(
                description="npm install hangs",
                resolution="Clear npm cache (npm cache clean --force) or check network.",
                category="npm",
                tags=["npm", "cache", "network"]
            ),
            KnowledgeBaseItem(
                description="Update npm version",
                resolution="Run npm install -g npm@latest.",
                category="npm",
                tags=["npm", "update", "version"]
            ),
            KnowledgeBaseItem(
                description="Python import error module not found",
                resolution="Check if module is installed (pip list) and verify PYTHONPATH.",
                category="python",
                tags=["python", "import", "module", "path"]
            ),
            KnowledgeBaseItem(
                description="Docker container won't start",
                resolution="Check docker logs <container_id> for error details and verify port availability.",
                category="docker",
                tags=["docker", "container", "startup", "logs"]
            ),
            KnowledgeBaseItem(
                description="Git merge conflict",
                resolution="Use git status to see conflicts, edit files to resolve, then git add and git commit.",
                category="git",
                tags=["git", "merge", "conflict", "resolution"]
            ),
            KnowledgeBaseItem(
                description="Database connection timeout",
                resolution="Check database server status, network connectivity, and connection string parameters.",
                category="database",
                tags=["database", "connection", "timeout", "network"]
            )
        ]
        
        self._knowledge_base = default_items
        self._save_knowledge_base()
        logger.info(f"Initialized knowledge base with {len(default_items)} default items")

    def _save_knowledge_base(self) -> None:
        """Save knowledge base to file."""
        try:
            data = [item.dict() for item in self._knowledge_base]
            with open(self.kb_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            logger.debug(f"Saved knowledge base to {self.kb_file}")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def get_all_items(self) -> List[KnowledgeBaseItem]:
        """Get all knowledge base items."""
        return self._knowledge_base.copy()

    def get_descriptions(self) -> List[str]:
        """Get all problem descriptions."""
        return [item.description for item in self._knowledge_base]

    def get_resolutions(self) -> List[str]:
        """Get all resolutions."""
        return [item.resolution for item in self._knowledge_base]

    def get_items_by_category(self, category: str) -> List[KnowledgeBaseItem]:
        """Get knowledge base items by category."""
        return [
            item for item in self._knowledge_base
            if item.category and item.category.lower() == category.lower()
        ]

    def get_items_by_tag(self, tag: str) -> List[KnowledgeBaseItem]:
        """Get knowledge base items by tag."""
        return [
            item for item in self._knowledge_base
            if item.tags and tag.lower() in [t.lower() for t in item.tags]
        ]

    def add_item(self, item: KnowledgeBaseItem) -> None:
        """Add a new knowledge base item."""
        # Set timestamps
        item.created_at = datetime.utcnow()
        item.updated_at = datetime.utcnow()
        
        self._knowledge_base.append(item)
        self._save_knowledge_base()
        logger.info(f"Added new knowledge base item: {item.description[:50]}...")

    def update_item(self, index: int, updated_item: KnowledgeBaseItem) -> bool:
        """
        Update a knowledge base item by index.

        Args:
            index: Index of the item to update
            updated_item: Updated item data

        Returns:
            True if successful, False if index is invalid
        """
        if 0 <= index < len(self._knowledge_base):
            # Preserve created_at, update updated_at
            updated_item.created_at = self._knowledge_base[index].created_at
            updated_item.updated_at = datetime.utcnow()
            
            self._knowledge_base[index] = updated_item
            self._save_knowledge_base()
            logger.info(f"Updated knowledge base item at index {index}")
            return True
        return False

    def remove_item(self, index: int) -> bool:
        """
        Remove a knowledge base item by index.

        Args:
            index: Index of the item to remove

        Returns:
            True if successful, False if index is invalid
        """
        if 0 <= index < len(self._knowledge_base):
            removed_item = self._knowledge_base.pop(index)
            self._save_knowledge_base()
            logger.info(f"Removed knowledge base item: {removed_item.description[:50]}...")
            return True
        return False

    def search_items(self, query: str) -> List[KnowledgeBaseItem]:
        """
        Search knowledge base items by text in description or resolution.

        Args:
            query: Search query

        Returns:
            List of matching items
        """
        query_lower = query.lower()
        matching_items = []
        
        for item in self._knowledge_base:
            if (query_lower in item.description.lower() or
                query_lower in item.resolution.lower() or
                (item.category and query_lower in item.category.lower()) or
                (item.tags and any(query_lower in tag.lower() for tag in item.tags))):
                matching_items.append(item)
        
        return matching_items

    def get_categories(self) -> List[str]:
        """Get all unique categories."""
        categories = set()
        for item in self._knowledge_base:
            if item.category:
                categories.add(item.category)
        return sorted(list(categories))

    def get_tags(self) -> List[str]:
        """Get all unique tags."""
        tags = set()
        for item in self._knowledge_base:
            if item.tags:
                tags.update(item.tags)
        return sorted(list(tags))

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        categories = self.get_categories()
        tags = self.get_tags()
        
        category_counts = {}
        for category in categories:
            category_counts[category] = len(self.get_items_by_category(category))
        
        return {
            "total_items": len(self._knowledge_base),
            "categories": categories,
            "category_counts": category_counts,
            "total_categories": len(categories),
            "total_tags": len(tags),
            "tags": tags
        }

    def reload(self) -> None:
        """Reload knowledge base from file."""
        self._load_knowledge_base() 