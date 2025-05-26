"""
Chroma DB service for vector embeddings and document storage.
"""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..models.schemas import KnowledgeBaseItem, Query, TroubleshootResponse
from ..models.kb_model import KBModelManager

logger = logging.getLogger(__name__)


class ChromaService:
    """
    Service for managing Chroma vector database operations.
    Handles document storage, embeddings, and vector search.
    """

    def __init__(
        self,
        model_manager: KBModelManager,
        persist_directory: str = "data/chroma",
        collection_name: str = "kb_documents"
    ):
        """
        Initialize the Chroma Service.

        Args:
            model_manager: KB Model Manager for embeddings
            persist_directory: Directory to persist Chroma data
            collection_name: Name of the Chroma collection
        """
        self.model_manager = model_manager
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        self._initialize_client()
        
        # Get or create collection
        self._initialize_collection()
        
        logger.info(f"Chroma service initialized with collection: {collection_name}")

    def _initialize_client(self) -> None:
        """Initialize Chroma client with persistence."""
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Chroma client initialized with persistence at: {self.persist_directory}")
        except Exception as e:
            logger.exception("Failed to initialize Chroma client")
            raise RuntimeError(f"Chroma client initialization failed: {str(e)}")

    def _initialize_collection(self) -> None:
        """Initialize or get existing collection."""
        try:
            # Create custom embedding function using our model
            embedding_function = ChromaEmbeddingFunction(self.model_manager)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={"description": "KB Nova Pipeline knowledge base documents"}
            )
            
            logger.info(f"Collection '{self.collection_name}' ready with {self.collection.count()} documents")
            
        except Exception as e:
            logger.exception("Failed to initialize collection")
            raise RuntimeError(f"Collection initialization failed: {str(e)}")

    def add_knowledge_base_items(self, items: List[KnowledgeBaseItem]) -> None:
        """
        Add knowledge base items to Chroma collection.

        Args:
            items: List of knowledge base items to add
        """
        if not items:
            return
        
        try:
            documents = []
            metadatas = []
            ids = []
            
            for item in items:
                # Create document text combining description and resolution
                document = f"Problem: {item.description}\nSolution: {item.resolution}"
                documents.append(document)
                
                # Create metadata
                metadata = {
                    "type": "knowledge_base",
                    "category": item.category or "general",
                    "tags": ",".join(item.tags) if item.tags else "",
                    "created_at": item.created_at.isoformat() if item.created_at else datetime.utcnow().isoformat(),
                    "description": item.description,
                    "resolution": item.resolution
                }
                metadatas.append(metadata)
                
                # Generate unique ID
                ids.append(f"kb_{uuid.uuid4().hex}")
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(items)} knowledge base items to Chroma")
            
        except Exception as e:
            logger.exception("Failed to add knowledge base items")
            raise RuntimeError(f"Failed to add items to Chroma: {str(e)}")

    def add_chat_message(
        self,
        user_message: str,
        assistant_response: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a chat conversation to Chroma for future reference.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            session_id: Optional session ID for grouping conversations
            metadata: Additional metadata

        Returns:
            Document ID
        """
        try:
            # Create document text
            document = f"User: {user_message}\nAssistant: {assistant_response}"
            
            # Create metadata
            chat_metadata = {
                "type": "chat",
                "session_id": session_id or "default",
                "timestamp": datetime.utcnow().isoformat(),
                "user_message": user_message,
                "assistant_response": assistant_response
            }
            
            if metadata:
                chat_metadata.update(metadata)
            
            # Generate unique ID
            doc_id = f"chat_{uuid.uuid4().hex}"
            
            # Add to collection
            self.collection.add(
                documents=[document],
                metadatas=[chat_metadata],
                ids=[doc_id]
            )
            
            logger.debug(f"Added chat message to Chroma: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.exception("Failed to add chat message")
            raise RuntimeError(f"Failed to add chat message: {str(e)}")

    def search_similar_documents(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include_distances: bool = True
    ) -> Dict[str, Any]:
        """
        Search for similar documents in the collection.

        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filter conditions
            include_distances: Whether to include similarity distances

        Returns:
            Search results with documents, metadata, and distances
        """
        try:
            include_list = ["documents", "metadatas"]
            if include_distances:
                include_list.append("distances")
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=include_list
            )
            
            logger.debug(f"Found {len(results['documents'][0])} similar documents for query")
            return results
            
        except Exception as e:
            logger.exception("Failed to search documents")
            raise RuntimeError(f"Document search failed: {str(e)}")

    def search_knowledge_base(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base items specifically.

        Args:
            query: Search query
            n_results: Number of results to return
            category: Optional category filter

        Returns:
            List of matching knowledge base items with scores
        """
        where_filter = {"type": "knowledge_base"}
        if category:
            where_filter["category"] = category
        
        results = self.search_similar_documents(
            query=query,
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results.get("distances", [None] * len(results["documents"][0]))[0]
            )):
                formatted_results.append({
                    "description": metadata.get("description", ""),
                    "resolution": metadata.get("resolution", ""),
                    "category": metadata.get("category", ""),
                    "tags": metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                    "similarity_score": 1 - distance if distance is not None else 0.0,
                    "document": doc,
                    "metadata": metadata
                })
        
        return formatted_results

    def search_chat_history(
        self,
        query: str,
        session_id: Optional[str] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search chat history for relevant conversations.

        Args:
            query: Search query
            session_id: Optional session ID filter
            n_results: Number of results to return

        Returns:
            List of relevant chat conversations
        """
        where_filter = {"type": "chat"}
        if session_id:
            where_filter["session_id"] = session_id
        
        results = self.search_similar_documents(
            query=query,
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results.get("distances", [None] * len(results["documents"][0]))[0]
            ):
                formatted_results.append({
                    "user_message": metadata.get("user_message", ""),
                    "assistant_response": metadata.get("assistant_response", ""),
                    "session_id": metadata.get("session_id", ""),
                    "timestamp": metadata.get("timestamp", ""),
                    "similarity_score": 1 - distance if distance is not None else 0.0,
                    "document": doc,
                    "metadata": metadata
                })
        
        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Chroma collection."""
        try:
            total_count = self.collection.count()
            
            # Get counts by type
            kb_results = self.collection.get(where={"type": "knowledge_base"}, limit=1)
            chat_results = self.collection.get(where={"type": "chat"}, limit=1)
            
            # Count by querying with limit
            kb_count = len(self.collection.get(where={"type": "knowledge_base"})["ids"])
            chat_count = len(self.collection.get(where={"type": "chat"})["ids"])
            
            return {
                "total_documents": total_count,
                "knowledge_base_items": kb_count,
                "chat_messages": chat_count,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.exception("Failed to get collection stats")
            return {"error": str(e)}

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self._initialize_collection()
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.exception("Failed to clear collection")
            raise RuntimeError(f"Failed to clear collection: {str(e)}")

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete specific documents by IDs.

        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from collection")
        except Exception as e:
            logger.exception("Failed to delete documents")
            raise RuntimeError(f"Failed to delete documents: {str(e)}")


class ChromaEmbeddingFunction:
    """Custom embedding function using our KB model."""
    
    def __init__(self, model_manager: KBModelManager):
        self.model_manager = model_manager
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input texts."""
        try:
            # Use our model to generate embeddings
            embeddings = self.model_manager.encode(input, convert_to_tensor=False)
            
            # Convert to list of lists
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            else:
                return embeddings
                
        except Exception as e:
            logger.exception("Failed to generate embeddings")
            raise RuntimeError(f"Embedding generation failed: {str(e)}") 