"""
Chat service for conversational AI with Chroma DB integration.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models.schemas import Query, TroubleshootResponse
from ..data.chroma_service import ChromaService
from ..inference.troubleshoot import TroubleshootService

logger = logging.getLogger(__name__)


class ChatMessage:
    """Chat message model."""
    
    def __init__(
        self,
        content: str,
        role: str = "user",
        timestamp: Optional[datetime] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.role = role  # "user" or "assistant"
        self.timestamp = timestamp or datetime.utcnow()
        self.session_id = session_id
        self.metadata = metadata or {}


class ChatSession:
    """Chat session model."""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class ChatService:
    """
    Service for managing conversational AI with context awareness.
    Uses Chroma DB for storing and retrieving chat history and context.
    """

    def __init__(
        self,
        chroma_service: ChromaService,
        troubleshoot_service: TroubleshootService,
        max_context_messages: int = 10,
        context_similarity_threshold: float = 0.7
    ):
        """
        Initialize the Chat Service.

        Args:
            chroma_service: Chroma service for vector operations
            troubleshoot_service: Troubleshoot service for generating responses
            max_context_messages: Maximum number of context messages to consider
            context_similarity_threshold: Minimum similarity for context retrieval
        """
        self.chroma_service = chroma_service
        self.troubleshoot_service = troubleshoot_service
        self.max_context_messages = max_context_messages
        self.context_similarity_threshold = context_similarity_threshold
        
        # In-memory session storage (in production, use Redis or database)
        self.sessions: Dict[str, ChatSession] = {}
        
        logger.info("Chat service initialized")

    def create_session(self) -> str:
        """
        Create a new chat session.

        Returns:
            Session ID
        """
        session = ChatSession()
        self.sessions[session.session_id] = session
        logger.info(f"Created new chat session: {session.session_id}")
        return session.session_id

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get a chat session by ID.

        Args:
            session_id: Session ID

        Returns:
            Chat session or None if not found
        """
        return self.sessions.get(session_id)

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        use_context: bool = True,
        include_chat_history: bool = True
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate a response.

        Args:
            message: User message
            session_id: Optional session ID
            use_context: Whether to use context from knowledge base
            include_chat_history: Whether to include chat history in context

        Returns:
            Chat response with metadata
        """
        try:
            # Get or create session
            if session_id:
                session = self.get_session(session_id)
                if not session:
                    session = ChatSession(session_id)
                    self.sessions[session_id] = session
            else:
                session = ChatSession()
                self.sessions[session.session_id] = session
                session_id = session.session_id

            # Add user message to session
            user_message = ChatMessage(
                content=message,
                role="user",
                session_id=session_id
            )
            session.messages.append(user_message)
            session.updated_at = datetime.utcnow()

            # Generate response with context
            response_data = self._generate_contextual_response(
                message=message,
                session=session,
                use_context=use_context,
                include_chat_history=include_chat_history
            )

            # Add assistant response to session
            assistant_message = ChatMessage(
                content=response_data["response"],
                role="assistant",
                session_id=session_id,
                metadata=response_data.get("metadata", {})
            )
            session.messages.append(assistant_message)

            # Store conversation in Chroma for future context
            self.chroma_service.add_chat_message(
                user_message=message,
                assistant_response=response_data["response"],
                session_id=session_id,
                metadata={
                    "confidence": response_data.get("confidence", "unknown"),
                    "source": response_data.get("source", "unknown"),
                    "context_used": use_context,
                    "chat_history_used": include_chat_history
                }
            )

            return {
                "session_id": session_id,
                "message": message,
                "response": response_data["response"],
                "confidence": response_data.get("confidence", "unknown"),
                "source": response_data.get("source", "unknown"),
                "context_sources": response_data.get("context_sources", []),
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": response_data.get("metadata", {})
            }

        except Exception as e:
            logger.exception("Chat processing failed")
            return {
                "session_id": session_id or "unknown",
                "message": message,
                "response": f"I apologize, but I encountered an error processing your message: {str(e)}",
                "confidence": "low",
                "source": "error",
                "context_sources": [],
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {"error": str(e)}
            }

    def _generate_contextual_response(
        self,
        message: str,
        session: ChatSession,
        use_context: bool,
        include_chat_history: bool
    ) -> Dict[str, Any]:
        """
        Generate a contextual response using knowledge base and chat history.

        Args:
            message: User message
            session: Chat session
            use_context: Whether to use knowledge base context
            include_chat_history: Whether to include chat history

        Returns:
            Response data with metadata
        """
        context_sources = []
        enhanced_query = message

        # Get relevant context from knowledge base
        if use_context:
            kb_context = self.chroma_service.search_knowledge_base(
                query=message,
                n_results=3
            )
            
            if kb_context:
                context_sources.extend([
                    {
                        "type": "knowledge_base",
                        "description": item["description"],
                        "resolution": item["resolution"],
                        "similarity_score": item["similarity_score"]
                    }
                    for item in kb_context
                    if item["similarity_score"] >= self.context_similarity_threshold
                ])

        # Get relevant chat history
        if include_chat_history and len(session.messages) > 1:
            chat_context = self.chroma_service.search_chat_history(
                query=message,
                session_id=session.session_id,
                n_results=3
            )
            
            if chat_context:
                context_sources.extend([
                    {
                        "type": "chat_history",
                        "user_message": item["user_message"],
                        "assistant_response": item["assistant_response"],
                        "similarity_score": item["similarity_score"]
                    }
                    for item in chat_context
                    if item["similarity_score"] >= self.context_similarity_threshold
                ])

        # Enhance query with context if available
        if context_sources:
            context_text = self._build_context_text(context_sources)
            enhanced_query = f"Context: {context_text}\n\nUser Question: {message}"

        # Generate response using troubleshoot service
        query = Query(text=enhanced_query)
        troubleshoot_response = self.troubleshoot_service.troubleshoot(query)

        return {
            "response": troubleshoot_response.response,
            "confidence": troubleshoot_response.confidence,
            "source": troubleshoot_response.source,
            "context_sources": context_sources,
            "metadata": {
                "original_query": message,
                "enhanced_query": enhanced_query,
                "similarity_score": troubleshoot_response.similarity_score,
                "context_count": len(context_sources)
            }
        }

    def _build_context_text(self, context_sources: List[Dict[str, Any]]) -> str:
        """
        Build context text from context sources.

        Args:
            context_sources: List of context sources

        Returns:
            Formatted context text
        """
        context_parts = []
        
        for source in context_sources:
            if source["type"] == "knowledge_base":
                context_parts.append(
                    f"KB: {source['description']} -> {source['resolution']}"
                )
            elif source["type"] == "chat_history":
                context_parts.append(
                    f"Previous: Q: {source['user_message']} A: {source['assistant_response']}"
                )
        
        return " | ".join(context_parts)

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get chat history for a session.

        Args:
            session_id: Session ID

        Returns:
            List of chat messages
        """
        session = self.get_session(session_id)
        if not session:
            return []

        return [
            {
                "content": msg.content,
                "role": msg.role,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in session.messages
        ]

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a chat session.

        Args:
            session_id: Session ID

        Returns:
            True if session was cleared, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared chat session: {session_id}")
            return True
        return False

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get information about all active sessions."""
        return [
            {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "message_count": len(session.messages)
            }
            for session in self.sessions.values()
        ]

    def search_conversations(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search across all conversations.

        Args:
            query: Search query
            session_id: Optional session ID filter
            limit: Maximum number of results

        Returns:
            List of matching conversations
        """
        return self.chroma_service.search_chat_history(
            query=query,
            session_id=session_id,
            n_results=limit
        )

    def get_chat_stats(self) -> Dict[str, Any]:
        """Get chat service statistics."""
        total_messages = sum(len(session.messages) for session in self.sessions.values())
        
        return {
            "active_sessions": len(self.sessions),
            "total_messages": total_messages,
            "chroma_stats": self.chroma_service.get_collection_stats(),
            "max_context_messages": self.max_context_messages,
            "context_similarity_threshold": self.context_similarity_threshold
        } 