#!/usr/bin/env python3
"""
Simple SentenceTransformer Demo - Your Code in KB Nova Pipeline

This script demonstrates how your exact SentenceTransformer code
integrates conceptually with the KB Nova Pipeline Models architecture.

Note: This demo shows the integration patterns without requiring
the full sentence-transformers installation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


class MockSentenceTransformer:
    """
    Mock SentenceTransformer that simulates your all-MiniLM-L6-v2 model
    for demonstration purposes.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        self.max_seq_length = 256
        print(f"Mock SentenceTransformer loaded: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def encode(self, sentences: List[str], convert_to_tensor: bool = True) -> np.ndarray:
        """Mock encode method that generates random embeddings for demo."""
        # Generate random embeddings (in real usage, these would be actual embeddings)
        embeddings = np.random.rand(len(sentences), self.embedding_dim)
        
        # Normalize embeddings (like real sentence transformers)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        print(f"Encoded {len(sentences)} sentences to {embeddings.shape}")
        return embeddings
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between embeddings."""
        # Cosine similarity
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        print(f"Calculated similarity matrix: {similarity_matrix.shape}")
        return similarity_matrix


def demo_your_original_code():
    """Demonstrate your original SentenceTransformer code."""
    print("=== Your Original SentenceTransformer Code ===")
    
    # 1. Load a pretrained Sentence Transformer model
    model = MockSentenceTransformer("all-MiniLM-L6-v2")
    
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
    # Shows similarity scores between all sentence pairs
    
    return model, embeddings, similarities


class MockKBModelManager:
    """
    Mock KB Model Manager that shows how your SentenceTransformer
    integrates with the KB Nova Pipeline architecture.
    """
    
    def __init__(self, base_model_name: str = "all-MiniLM-L6-v2"):
        self.base_model_name = base_model_name
        self.model = MockSentenceTransformer(base_model_name)
        self.device = "cpu"  # or "cuda" if available
        print(f"KB Model Manager initialized with {base_model_name}")
    
    def encode(self, texts: List[str], convert_to_tensor: bool = True) -> np.ndarray:
        """Encode texts using the underlying model."""
        return self.model.encode(texts, convert_to_tensor)
    
    def find_similar(self, query: str, corpus_texts: List[str], top_k: int = 1) -> List[Tuple[int, float]]:
        """Find similar texts in corpus for the given query."""
        if not corpus_texts:
            return []
        
        # Encode query and corpus
        query_embedding = self.encode([query])
        corpus_embeddings = self.encode(corpus_texts)
        
        # Calculate similarities
        similarities = self.model.similarity(query_embedding, corpus_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.base_model_name,
            "embedding_dimension": self.model.embedding_dim,
            "max_sequence_length": self.model.max_seq_length,
            "device": self.device
        }
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


def demo_kb_model_manager_integration():
    """Demonstrate how your code integrates with KB Model Manager."""
    print("\n=== KB Model Manager Integration ===")
    
    # Initialize with your preferred model
    kb_manager = MockKBModelManager(base_model_name="all-MiniLM-L6-v2")
    
    print(f"Model loaded: {kb_manager.is_model_loaded()}")
    model_info = kb_manager.get_model_info()
    print(f"Model info: {model_info['model_name']}")
    print(f"Embedding dimension: {model_info['embedding_dimension']}")
    print(f"Device: {model_info['device']}")
    
    # Your sentences using KB Manager
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]
    
    # Encode using KB Manager (same as your model.encode())
    embeddings = kb_manager.encode(sentences)
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
    
    # Mock knowledge base items
    kb_problems = [
        "npm install fails with ERESOLVE dependency conflict",
        "Docker container exits immediately after startup",
        "Python module import error: ModuleNotFoundError",
        "Git merge conflict in package.json file",
        "Database connection timeout after 30 seconds",
    ]
    
    # Initialize KB Manager with your model
    kb_manager = MockKBModelManager(base_model_name="all-MiniLM-L6-v2")
    
    # Troubleshooting queries (similar to your sentences)
    troubleshooting_queries = [
        "npm package installation error",
        "Docker container won't start properly",
        "Python import module not found error",
    ]
    
    print("Troubleshooting Queries:")
    for i, query in enumerate(troubleshooting_queries):
        print(f"  {i+1}. {query}")
    
    print(f"\nKnowledge Base Problems ({len(kb_problems)} items):")
    for i, problem in enumerate(kb_problems):
        print(f"  {i+1}. {problem}")
    
    # Encode all texts (your model.encode() equivalent)
    query_embeddings = kb_manager.encode(troubleshooting_queries)
    problem_embeddings = kb_manager.encode(kb_problems)
    
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


def demo_api_integration():
    """Show how your model works in the production API."""
    print("\n=== Production API Integration ===")
    
    # This is how your model is used in the FastAPI endpoints
    kb_manager = MockKBModelManager(base_model_name="all-MiniLM-L6-v2")
    
    # Mock knowledge base
    kb_items = [
        {
            "id": "kb_001",
            "description": "npm install fails with ERESOLVE dependency conflict",
            "resolution": "Run 'npm install --legacy-peer-deps' to resolve dependency conflicts",
            "category": "npm"
        },
        {
            "id": "kb_002", 
            "description": "Docker container exits immediately after startup",
            "resolution": "Check container logs with 'docker logs <container_id>' and verify entry point",
            "category": "docker"
        },
        {
            "id": "kb_003",
            "description": "Python module import error: ModuleNotFoundError",
            "resolution": "Install missing module with 'pip install <module_name>' or check PYTHONPATH",
            "category": "python"
        }
    ]
    
    # Simulate API request
    user_query = "npm package installation error"
    
    # Find similar problems (API logic)
    problem_descriptions = [item["description"] for item in kb_items]
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
        print(f"    Problem: {kb_item['description']}")
        print(f"    Solution: {kb_item['resolution']}")
        print(f"    Category: {kb_item['category']}")
        print()


def demo_architecture_overview():
    """Show the overall architecture integration."""
    print("\n=== KB Nova Pipeline Architecture Overview ===")
    
    architecture = {
        "Your SentenceTransformer Code": {
            "model": "all-MiniLM-L6-v2",
            "functionality": ["encode()", "similarity()"],
            "embedding_dim": 384,
            "use_case": "Text similarity and search"
        },
        "KB Model Manager": {
            "wraps": "SentenceTransformer",
            "adds": ["Thread safety", "Model management", "Fine-tuning"],
            "location": "src/models/kb_model.py"
        },
        "Knowledge Base Service": {
            "manages": "Static knowledge base data",
            "provides": "Structured KB items",
            "location": "src/data/knowledge_base.py"
        },
        "Chroma Service": {
            "uses": "Your embeddings for vector search",
            "provides": "Persistent vector database",
            "location": "src/data/chroma_service.py"
        },
        "FastAPI Endpoints": {
            "exposes": "REST API for troubleshooting",
            "uses": "All above components",
            "location": "src/api/main.py"
        },
        "Chat Service": {
            "provides": "Conversational AI",
            "uses": "Context-aware search",
            "location": "src/api/chat.py"
        }
    }
    
    print("Component Integration:")
    for component, details in architecture.items():
        print(f"\n📦 {component}:")
        for key, value in details.items():
            if isinstance(value, list):
                print(f"  {key}: {', '.join(value)}")
            else:
                print(f"  {key}: {value}")
    
    print(f"\n🔄 Data Flow:")
    print(f"1. User Query → FastAPI Endpoint")
    print(f"2. Query → Your SentenceTransformer (encode)")
    print(f"3. Query Embedding → Chroma/KB Search")
    print(f"4. Similar Items → Response Generation")
    print(f"5. Structured Response → User")


def main():
    """Run all demonstrations."""
    print("🚀 SentenceTransformer Demo - KB Nova Pipeline Integration")
    print("=" * 60)
    print("Note: This demo uses mock implementations to show integration patterns")
    print("=" * 60)
    
    # 1. Your original code
    original_model, embeddings, similarities = demo_your_original_code()
    
    # 2. KB Model Manager integration
    kb_manager = demo_kb_model_manager_integration()
    
    # 3. Troubleshooting use case
    query_emb, problem_emb = demo_troubleshooting_use_case()
    
    # 4. Production API integration
    demo_api_integration()
    
    # 5. Architecture overview
    demo_architecture_overview()
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("\nKey Integration Points:")
    print("• Your all-MiniLM-L6-v2 model is the core of the system")
    print("• Same encode() and similarity() methods you're familiar with")
    print("• Enhanced with production features:")
    print("  - Thread-safe operations")
    print("  - Model management and fine-tuning")
    print("  - Vector database integration (Chroma)")
    print("  - REST API endpoints")
    print("  - Knowledge base integration")
    print("  - Context-aware chat functionality")
    print("\n🎯 Your Code + KB Nova Pipeline = Production-Ready AI System!")
    
    print(f"\n📁 Key Files in Your Project:")
    print(f"  - src/models/kb_model.py (wraps your SentenceTransformer)")
    print(f"  - src/data/chroma_service.py (uses your embeddings)")
    print(f"  - src/api/main.py (exposes your model via REST API)")
    print(f"  - notebooks/exploratory/02_sentence_transformer_analysis.ipynb")
    print(f"  - examples/sentence_transformer_demo.py (full integration demo)")
    
    print(f"\n🚀 To install and run the full system:")
    print(f"  pip install -r requirements.txt")
    print(f"  python src/api/main.py")


if __name__ == "__main__":
    main() 