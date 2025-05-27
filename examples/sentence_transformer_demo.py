#!/usr/bin/env python3
"""
SentenceTransformer Demo - Your Code in KB Nova Pipeline

This script demonstrates how your exact SentenceTransformer code
integrates with the KB Nova Pipeline Models architecture.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from sentence_transformers import SentenceTransformer
from models.kb_model import KBModelManager
from data.knowledge_base import KnowledgeBaseService
from data.chroma_service import ChromaService
import numpy as np


def demo_your_original_code():
    """Demonstrate your original SentenceTransformer code."""
    print("=== Your Original SentenceTransformer Code ===")
    
    # 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # The sentences to encode
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    
    # 2. Calculate embeddings by calling model.encode()
    embeddings = model.encode(sentences)
    print(f"Embeddings shape: {embeddings.shape}")
    # [3, 384]
    
    # 3. Calculate the embedding similarities
    similarities = model.similarity(embeddings, embeddings)
    print("Similarity matrix:")
    print(similarities)
    # tensor([[1.0000, 0.6660, 0.1046],
    #         [0.6660, 1.0000, 0.1411],
    #         [0.1046, 0.1411, 1.0000]])
    
    return model, embeddings, similarities


def demo_kb_model_manager_integration():
    """Demonstrate how your code integrates with KB Model Manager."""
    print("\n=== KB Model Manager Integration ===")
    
    # Initialize with your preferred model
    kb_manager = KBModelManager(
        base_model_name="all-MiniLM-L6-v2",  # Your model!
        models_dir="../models"
    )
    
    print(f"Model loaded: {kb_manager.is_model_loaded()}")
    print(f"Model info: {kb_manager.get_model_info().model_name}")
    print(f"Device: {kb_manager.device}")
    
    # Your sentences using KB Manager
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    
    # Encode using KB Manager (same as your model.encode())
    embeddings = kb_manager.encode(sentences, convert_to_tensor=False)
    print(f"KB Manager embeddings shape: {embeddings.shape}")
    
    # Find similar sentences (enhanced functionality)
    query = "It's a beautiful sunny day"
    similar_results = kb_manager.find_similar(
        query=query,
        corpus_texts=sentences,
        top_k=2
    )
    
    print(f"\nSimilarity search for: '{query}'")
    for idx, score in similar_results:
        print(f"  [{score:.3f}] {sentences[idx]}")
    
    return kb_manager


def demo_troubleshooting_use_case():
    """Demonstrate your model with troubleshooting scenarios."""
    print("\n=== Troubleshooting Use Case ===")
    
    # Load knowledge base
    kb_service = KnowledgeBaseService(data_dir="../data")
    kb_items = kb_service.get_all_items()
    
    # Initialize KB Manager with your model
    kb_manager = KBModelManager(base_model_name="all-MiniLM-L6-v2")
    
    # Troubleshooting queries (similar to your sentences)
    troubleshooting_queries = [
        "npm install fails with dependency error",
        "Docker container won't start properly",
        "Python import module not found error",
    ]
    
    # Knowledge base problems
    kb_problems = [item.description for item in kb_items[:5]]
    
    print("Troubleshooting Queries:")
    for i, query in enumerate(troubleshooting_queries):
        print(f"  {i+1}. {query}")
    
    print(f"\nKnowledge Base Problems ({len(kb_problems)} items):")
    for i, problem in enumerate(kb_problems):
        print(f"  {i+1}. {problem}")
    
    # Encode all texts (your model.encode() equivalent)
    query_embeddings = kb_manager.encode(troubleshooting_queries, convert_to_tensor=False)
    problem_embeddings = kb_manager.encode(kb_problems, convert_to_tensor=False)
    
    print(f"\nQuery embeddings shape: {query_embeddings.shape}")
    print(f"Problem embeddings shape: {problem_embeddings.shape}")
    
    # Find best matches for each query (enhanced similarity)
    print("\n=== Best Matches ===")
    for i, query in enumerate(troubleshooting_queries):
        similar_results = kb_manager.find_similar(
            query=query,
            corpus_texts=kb_problems,
            top_k=2
        )
        
        print(f"\nQuery: {query}")
        for idx, score in similar_results:
            print(f"  [{score:.3f}] {kb_problems[idx]}")
    
    return query_embeddings, problem_embeddings


def demo_chroma_integration():
    """Demonstrate Chroma DB integration with your model."""
    print("\n=== Chroma DB Integration ===")
    
    try:
        # Initialize KB Manager with your model
        kb_manager = KBModelManager(base_model_name="all-MiniLM-L6-v2")
        
        # Initialize Chroma with your model
        chroma_service = ChromaService(
            model_manager=kb_manager,
            persist_directory="../data/chroma_demo",
            collection_name="minilm_demo"
        )
        
        # Your sentences in Chroma
        documents = [
            "The weather is lovely today.",
            "It's so sunny outside!",
            "He drove to the stadium.",
            "npm install fails with dependency error",
            "Docker container won't start properly",
        ]
        
        # Add documents to Chroma (uses your model for embeddings)
        for i, doc in enumerate(documents):
            chroma_service.add_document(
                document_id=f"doc_{i}",
                content=doc,
                metadata={"type": "demo", "index": i}
            )
        
        print(f"Added {len(documents)} documents to Chroma")
        
        # Search using your model embeddings
        query = "sunny weather"
        results = chroma_service.search_documents(query, n_results=3)
        
        print(f"\nChroma search for: '{query}'")
        for result in results:
            print(f"  [{result['similarity_score']:.3f}] {result['content']}")
        
        # Collection stats
        stats = chroma_service.get_collection_stats()
        print(f"\nChroma collection stats: {stats['total_documents']} documents")
        
        return chroma_service
        
    except Exception as e:
        print(f"Chroma demo failed: {str(e)}")
        return None


def demo_production_api_integration():
    """Show how your model works in the production API."""
    print("\n=== Production API Integration ===")
    
    # This is how your model is used in the FastAPI endpoints
    kb_manager = KBModelManager(base_model_name="all-MiniLM-L6-v2")
    
    # Simulate API request
    user_query = "npm package installation error"
    
    # Load knowledge base
    kb_service = KnowledgeBaseService(data_dir="../data")
    kb_items = kb_service.get_all_items()
    
    # Find similar problems (API logic)
    problem_descriptions = [item.description for item in kb_items]
    similar_results = kb_manager.find_similar(
        query=user_query,
        corpus_texts=problem_descriptions,
        top_k=3
    )
    
    print(f"API Query: {user_query}")
    print("API Response:")
    
    for idx, score in similar_results:
        kb_item = kb_items[idx]
        print(f"  Match [{score:.3f}]:")
        print(f"    Problem: {kb_item.description}")
        print(f"    Solution: {kb_item.resolution}")
        print(f"    Category: {kb_item.category}")
        print()


def main():
    """Run all demonstrations."""
    print("🚀 SentenceTransformer Demo - KB Nova Pipeline Integration")
    print("=" * 60)
    
    # 1. Your original code
    original_model, embeddings, similarities = demo_your_original_code()
    
    # 2. KB Model Manager integration
    kb_manager = demo_kb_model_manager_integration()
    
    # 3. Troubleshooting use case
    query_emb, problem_emb = demo_troubleshooting_use_case()
    
    # 4. Chroma integration
    chroma_service = demo_chroma_integration()
    
    # 5. Production API integration
    demo_production_api_integration()
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("\nKey Points:")
    print("• Your all-MiniLM-L6-v2 model is fully integrated")
    print("• Same encode() and similarity() functionality")
    print("• Enhanced with production features:")
    print("  - Thread-safe operations")
    print("  - Model management and fine-tuning")
    print("  - Chroma DB vector storage")
    print("  - FastAPI endpoints")
    print("  - Knowledge base integration")
    print("• Ready for production deployment!")


if __name__ == "__main__":
    main() 