"""
Inference service for troubleshooting queries using the KB model.
"""

import logging
from typing import List, Optional, Tuple

from ..models.schemas import Query, TroubleshootResponse, TrainingPair, KnowledgeBaseItem
from ..models.kb_model import KBModelManager
from ..data.knowledge_base import KnowledgeBaseService

logger = logging.getLogger(__name__)


class TroubleshootService:
    """
    Service for handling troubleshooting queries.
    Combines learned knowledge from training pairs with static knowledge base.
    """

    def __init__(
        self,
        model_manager: KBModelManager,
        kb_service: KnowledgeBaseService,
        chroma_service=None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the Troubleshoot Service.

        Args:
            model_manager: KB Model Manager instance
            kb_service: Knowledge Base Service instance
            chroma_service: Optional Chroma Service for enhanced search
            confidence_threshold: Minimum confidence threshold for responses
        """
        self.model_manager = model_manager
        self.kb_service = kb_service
        self.chroma_service = chroma_service
        self.confidence_threshold = confidence_threshold
        
        logger.info("Troubleshoot service initialized")

    def troubleshoot(self, query: Query) -> TroubleshootResponse:
        """
        Process a troubleshooting query and return the best matching solution.

        Args:
            query: Query object containing the problem description

        Returns:
            TroubleshootResponse with the recommended solution
        """
        try:
            # Get learned pairs and knowledge base items
            learned_pairs = self.model_manager.get_learned_pairs()
            kb_items = self.kb_service.get_all_items()
            
            # Build corpus for search
            corpus_inputs, corpus_answers, sources = self._build_search_corpus(
                learned_pairs, kb_items
            )
            
            if not corpus_inputs:
                return TroubleshootResponse(
                    query=query.text,
                    response="No knowledge base available. Please add training data or knowledge base items.",
                    similarity_score=0.0,
                    confidence="low",
                    source="system"
                )
            
            # Find the best matching solution
            best_match = self._find_best_match(query.text, corpus_inputs)
            
            if best_match is None:
                return TroubleshootResponse(
                    query=query.text,
                    response="No suitable solution found. Please try rephrasing your query or contact support.",
                    similarity_score=0.0,
                    confidence="low",
                    source="system"
                )
            
            best_idx, similarity_score = best_match
            
            # Get the response and source
            response = corpus_answers[best_idx]
            source = sources[best_idx]
            
            # Determine confidence level
            confidence = self._calculate_confidence(similarity_score)
            
            logger.info(f"Query processed: similarity={similarity_score:.3f}, confidence={confidence}")
            
            return TroubleshootResponse(
                query=query.text,
                response=response,
                similarity_score=similarity_score,
                confidence=confidence,
                source=source
            )
            
        except Exception as e:
            logger.exception("Troubleshooting failed")
            return TroubleshootResponse(
                query=query.text,
                response=f"An error occurred while processing your query: {str(e)}",
                similarity_score=0.0,
                confidence="low",
                source="error"
            )

    def get_similar_solutions(
        self,
        query: Query,
        top_k: int = 5
    ) -> List[TroubleshootResponse]:
        """
        Get multiple similar solutions for a query.

        Args:
            query: Query object
            top_k: Number of top solutions to return

        Returns:
            List of TroubleshootResponse objects
        """
        try:
            # Get learned pairs and knowledge base items
            learned_pairs = self.model_manager.get_learned_pairs()
            kb_items = self.kb_service.get_all_items()
            
            # Build corpus for search
            corpus_inputs, corpus_answers, sources = self._build_search_corpus(
                learned_pairs, kb_items
            )
            
            if not corpus_inputs:
                return []
            
            # Find top-k matches
            matches = self.model_manager.find_similar(
                query.text, corpus_inputs, top_k=min(top_k, len(corpus_inputs))
            )
            
            responses = []
            for idx, similarity_score in matches:
                confidence = self._calculate_confidence(similarity_score)
                
                response = TroubleshootResponse(
                    query=query.text,
                    response=corpus_answers[idx],
                    similarity_score=similarity_score,
                    confidence=confidence,
                    source=sources[idx]
                )
                responses.append(response)
            
            return responses
            
        except Exception as e:
            logger.exception("Failed to get similar solutions")
            return []

    def _build_search_corpus(
        self,
        learned_pairs: List[TrainingPair],
        kb_items: List[KnowledgeBaseItem]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Build search corpus from learned pairs and knowledge base items.

        Args:
            learned_pairs: List of training pairs
            kb_items: List of knowledge base items

        Returns:
            Tuple of (inputs, answers, sources)
        """
        corpus_inputs = []
        corpus_answers = []
        sources = []
        
        # Add learned pairs
        for pair in learned_pairs:
            corpus_inputs.append(pair.input)
            corpus_answers.append(pair.target)
            sources.append("learned")
        
        # Add knowledge base items
        for item in kb_items:
            corpus_inputs.append(item.description)
            corpus_answers.append(item.resolution)
            sources.append("knowledge_base")
        
        return corpus_inputs, corpus_answers, sources

    def _find_best_match(
        self,
        query: str,
        corpus_inputs: List[str]
    ) -> Optional[Tuple[int, float]]:
        """
        Find the best matching solution for a query.

        Args:
            query: Query text
            corpus_inputs: List of corpus input texts

        Returns:
            Tuple of (index, similarity_score) or None if no good match
        """
        matches = self.model_manager.find_similar(query, corpus_inputs, top_k=1)
        
        if not matches:
            return None
        
        best_idx, similarity_score = matches[0]
        
        # Check if similarity meets threshold
        if similarity_score < self.confidence_threshold:
            logger.warning(f"Best match similarity {similarity_score:.3f} below threshold {self.confidence_threshold}")
            # Still return the best match but with low confidence
        
        return best_idx, similarity_score

    def _calculate_confidence(self, similarity_score: float) -> str:
        """
        Calculate confidence level based on similarity score.

        Args:
            similarity_score: Similarity score between 0 and 1

        Returns:
            Confidence level string
        """
        if similarity_score >= 0.8:
            return "high"
        elif similarity_score >= 0.6:
            return "medium"
        else:
            return "low"

    def get_corpus_stats(self) -> dict:
        """Get statistics about the search corpus."""
        try:
            learned_pairs = self.model_manager.get_learned_pairs()
            kb_items = self.kb_service.get_all_items()
            
            return {
                "learned_pairs_count": len(learned_pairs),
                "knowledge_base_items_count": len(kb_items),
                "total_corpus_size": len(learned_pairs) + len(kb_items),
                "confidence_threshold": self.confidence_threshold
            }
        except Exception as e:
            logger.exception("Failed to get corpus stats")
            return {
                "error": str(e)
            }

    def update_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold.

        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold updated to {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0") 