"""
Demo script for Chroma DB integration with KB Nova Pipeline Models.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.kb_model import KBModelManager
from data.knowledge_base import KnowledgeBaseService
from data.chroma_service import ChromaService
from api.chat import ChatService
from inference.troubleshoot import TroubleshootService
from models.schemas import Query, KnowledgeBaseItem


def main():
    """Demonstrate Chroma DB integration."""
    print("🚀 KB Nova Pipeline Models - Chroma DB Integration Demo")
    print("=" * 60)
    
    try:
        # Initialize services
        print("📦 Initializing services...")
        
        # Model manager
        model_manager = KBModelManager(
            base_model_name="all-mpnet-base-v2",
            models_dir="models"
        )
        print("✅ Model manager initialized")
        
        # Knowledge base service
        kb_service = KnowledgeBaseService(data_dir="data")
        print("✅ Knowledge base service initialized")
        
        # Chroma service
        chroma_service = ChromaService(
            model_manager=model_manager,
            persist_directory="data/chroma_demo",
            collection_name="demo_collection"
        )
        print("✅ Chroma service initialized")
        
        # Add knowledge base items to Chroma
        print("\n📚 Loading knowledge base into Chroma...")
        kb_items = kb_service.get_all_items()
        if kb_items:
            chroma_service.add_knowledge_base_items(kb_items)
            print(f"✅ Loaded {len(kb_items)} knowledge base items")
        
        # Troubleshoot service
        troubleshoot_service = TroubleshootService(
            model_manager=model_manager,
            kb_service=kb_service,
            chroma_service=chroma_service
        )
        print("✅ Troubleshoot service initialized")
        
        # Chat service
        chat_service = ChatService(
            chroma_service=chroma_service,
            troubleshoot_service=troubleshoot_service
        )
        print("✅ Chat service initialized")
        
        # Demo 1: Vector search in knowledge base
        print("\n🔍 Demo 1: Vector Search in Knowledge Base")
        print("-" * 40)
        
        search_queries = [
            "npm install error",
            "docker container not starting",
            "python import module error"
        ]
        
        for query in search_queries:
            print(f"\nQuery: '{query}'")
            results = chroma_service.search_knowledge_base(query, n_results=2)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['similarity_score']:.3f}] {result['description']}")
                print(f"     Solution: {result['resolution'][:80]}...")
        
        # Demo 2: Chat with context
        print("\n💬 Demo 2: Context-Aware Chat")
        print("-" * 40)
        
        # Create a chat session
        session_id = chat_service.create_session()
        print(f"Created chat session: {session_id}")
        
        chat_queries = [
            "I'm having trouble with npm install",
            "The error says ERESOLVE",
            "What about Docker issues?"
        ]
        
        for query in chat_queries:
            print(f"\nUser: {query}")
            response = chat_service.chat(
                message=query,
                session_id=session_id,
                use_context=True,
                include_chat_history=True
            )
            
            print(f"Assistant: {response['response']}")
            print(f"Confidence: {response['confidence']}")
            print(f"Context sources: {len(response['context_sources'])}")
        
        # Demo 3: Add and search chat history
        print("\n📝 Demo 3: Chat History Search")
        print("-" * 40)
        
        # Add some sample chat messages
        chroma_service.add_chat_message(
            user_message="How do I fix git merge conflicts?",
            assistant_response="Use git status to see conflicts, edit files to resolve, then git add and git commit.",
            session_id="demo_session"
        )
        
        chroma_service.add_chat_message(
            user_message="Database connection timeout issue",
            assistant_response="Check database server status, network connectivity, and connection string parameters.",
            session_id="demo_session"
        )
        
        # Search chat history
        search_query = "git problems"
        print(f"\nSearching chat history for: '{search_query}'")
        chat_results = chroma_service.search_chat_history(search_query, n_results=3)
        
        for i, result in enumerate(chat_results, 1):
            print(f"  {i}. [{result['similarity_score']:.3f}] User: {result['user_message']}")
            print(f"     Assistant: {result['assistant_response'][:80]}...")
        
        # Demo 4: Statistics
        print("\n📊 Demo 4: Collection Statistics")
        print("-" * 40)
        
        stats = chroma_service.get_collection_stats()
        print(f"Total documents: {stats['total_documents']}")
        print(f"Knowledge base items: {stats['knowledge_base_items']}")
        print(f"Chat messages: {stats['chat_messages']}")
        print(f"Collection: {stats['collection_name']}")
        print(f"Storage: {stats['persist_directory']}")
        
        # Demo 5: Enhanced troubleshooting with Chroma
        print("\n🔧 Demo 5: Enhanced Troubleshooting")
        print("-" * 40)
        
        test_query = Query(text="My npm package won't install and shows dependency errors")
        response = troubleshoot_service.troubleshoot(test_query)
        
        print(f"Query: {test_query.text}")
        print(f"Response: {response.response}")
        print(f"Confidence: {response.confidence}")
        print(f"Similarity Score: {response.similarity_score:.3f}")
        print(f"Source: {response.source}")
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 