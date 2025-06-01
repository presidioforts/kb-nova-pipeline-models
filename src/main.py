"""
Production-ready LangChain Hybrid Knowledge Base Service
FastAPI application with intelligent routing between in-memory and LangChain ChromaDB storage
"""

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router
from .models.langchain_hybrid_kb import get_langchain_hybrid_kb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
    # Startup
    logger.info("🚀 Starting LangChain Hybrid Knowledge Base Service...")
    
    try:
        # Initialize LangChain hybrid knowledge base
        langchain_kb = await get_langchain_hybrid_kb()
        logger.info("✅ LangChain hybrid knowledge base initialized successfully")
        
        # Log initial system status
        health = await langchain_kb.health_check()
        logger.info(f"📊 System Status: {health['overall_status']}")
        
        # Log component status
        for component, status in health['components'].items():
            if isinstance(status, dict):
                logger.info(f"   {component}: {status.get('status', 'unknown')}")
            else:
                logger.info(f"   {component}: {status}")
        
        # Log LangChain integration status
        if 'langchain_integration' in health:
            langchain_info = health['langchain_integration']
            logger.info(f"🔗 LangChain Integration: {langchain_info.get('status', 'unknown')}")
            logger.info(f"   Vector Store: {langchain_info.get('vector_store', 'unknown')}")
            logger.info(f"   Embeddings: {langchain_info.get('embeddings', 'unknown')}")
        
        # Log performance baseline
        stats = langchain_kb.get_performance_stats()
        logger.info(f"📈 Performance baseline established")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize LangChain hybrid knowledge base: {e}")
        raise
    
    # Shutdown
    logger.info("🛑 Shutting down LangChain Hybrid Knowledge Base Service...")
    logger.info("✅ Shutdown complete")

# Create FastAPI application with LangChain hybrid system
app = FastAPI(
    title="LangChain Hybrid Knowledge Base Service",
    description="""
    Production-ready knowledge base service with LangChain integration and intelligent routing.
    
    ## Features
    
    * **LangChain Integration**: Full LangChain ecosystem compatibility with ChromaDB
    * **Intelligent Routing**: Automatically routes queries to optimal storage tier
    * **Multi-tier Performance**: 1-5ms (hot), 10-15ms (warm), 25-50ms (cold)
    * **Scalable Storage**: In-memory + LangChain ChromaDB for unlimited capacity
    * **Advanced Search**: Semantic similarity with SentenceTransformers + LangChain retrievers
    * **Model Training**: Fine-tuning with background job processing
    * **Production Ready**: Health checks, metrics, monitoring
    
    ## LangChain Architecture
    
    | Component | Technology | Purpose |
    |-----------|------------|---------|
    | **Vector Store** | Chroma | Persistent vector storage |
    | **Embeddings** | SentenceTransformerEmbeddings | Consistent embedding generation |
    | **Retriever** | BaseRetriever | Intelligent document retrieval |
    | **Hot Cache** | SentenceTransformers | Ultra-fast in-memory search |
    
    ## Performance Tiers
    
    | Tier | Storage | Latency | Technology |
    |------|---------|---------|------------|
    | **Hot Memory** | In-Memory | 1-5ms | SentenceTransformers direct |
    | **Warm Cache** | LangChain + Cache | 10-15ms | LangChain retriever + caching |
    | **Cold Storage** | LangChain ChromaDB | 25-50ms | Full LangChain pipeline |
    | **Massive Dataset** | LangChain ChromaDB | 25-50ms | Comprehensive corpus search |
    
    ## Intelligent Routing Logic
    
    The system automatically determines the best storage tier based on:
    - Query frequency and recency patterns
    - Result quality scores and relevance
    - Critical keywords detection
    - Access patterns and usage history
    - LangChain retriever performance metrics
    
    ## API Endpoints
    
    * `POST /troubleshoot` - Intelligent troubleshooting with LangChain routing
    * `POST /train` - Start model fine-tuning with LangChain integration
    * `GET /train/{job_id}` - Check training status
    * `GET /health` - Comprehensive system health including LangChain components
    * `GET /performance` - Detailed performance metrics with LangChain analysis
    * `POST /knowledge` - Add new knowledge items via LangChain
    """,
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← Security risk in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with LangChain system information"""
    try:
        langchain_kb = await get_langchain_hybrid_kb()
        health = await langchain_kb.health_check()
        stats = langchain_kb.get_performance_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "service": "LangChain Hybrid Knowledge Base Service",
                "version": "2.1.0",
                "status": health.get('overall_status', 'unknown'),
                "features": {
                    "langchain_integration": True,
                    "intelligent_routing": True,
                    "multi_tier_storage": True,
                    "scalable_chromadb": True,
                    "in_memory_performance": True,
                    "model_fine_tuning": True,
                    "production_monitoring": True
                },
                "langchain_components": {
                    "vector_store": "Chroma",
                    "embeddings": "SentenceTransformerEmbeddings",
                    "retriever": "BaseRetriever with similarity_score_threshold",
                    "document_processing": "LangChain Document pipeline"
                },
                "performance_summary": {
                    "total_queries_processed": stats.get('total_queries', 0),
                    "hot_cache_hit_rate": f"{stats.get('hot_hit_rate', 0) * 100:.1f}%",
                    "langchain_warm_hit_rate": f"{stats.get('warm_hit_rate', 0) * 100:.1f}%",
                    "langchain_cold_hit_rate": f"{stats.get('cold_hit_rate', 0) * 100:.1f}%",
                    "system_performance": "optimal" if stats.get('hot_hit_rate', 0) > 0.8 else "good"
                },
                "endpoints": {
                    "troubleshoot": "/api/v1/troubleshoot",
                    "train": "/api/v1/train",
                    "health": "/api/v1/health",
                    "performance": "/api/v1/performance",
                    "knowledge": "/api/v1/knowledge",
                    "docs": "/docs"
                },
                "architecture": {
                    "storage_tiers": [
                        {"name": "Hot Memory", "latency": "1-5ms", "technology": "SentenceTransformers", "use_case": "Critical/frequent queries"},
                        {"name": "Warm Cache", "latency": "10-15ms", "technology": "LangChain + Cache", "use_case": "Regular queries"},
                        {"name": "Cold Storage", "latency": "25-50ms", "technology": "LangChain ChromaDB", "use_case": "Comprehensive search"}
                    ],
                    "routing_intelligence": "Automatic based on usage patterns and query characteristics",
                    "langchain_benefits": "Full ecosystem compatibility, advanced retrievers, document processing"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "service": "LangChain Hybrid Knowledge Base Service",
                "version": "2.1.0",
                "status": "unhealthy",
                "error": "Service initialization failed",
                "message": "Please check system health at /api/v1/health",
                "langchain_integration": "failed"
            }
        )

@app.get("/api/v1/system/info")
async def system_info():
    """Detailed system information endpoint with LangChain details"""
    try:
        langchain_kb = await get_langchain_hybrid_kb()
        health = await langchain_kb.health_check()
        stats = langchain_kb.get_performance_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "system_info": {
                    "service_name": "LangChain Hybrid Knowledge Base Service",
                    "version": "2.1.0",
                    "architecture": "LangChain + ChromaDB Hybrid",
                    "intelligent_routing": True,
                    "langchain_integration": True
                },
                "langchain_components": {
                    "vector_store": health.get('components', {}).get('langchain_chromadb', {}).get('vector_store', 'Chroma'),
                    "embeddings": health.get('components', {}).get('langchain_chromadb', {}).get('embeddings', 'SentenceTransformerEmbeddings'),
                    "retriever_type": health.get('components', {}).get('retriever', {}).get('type', 'BaseRetriever'),
                    "document_processing": "LangChain Document pipeline"
                },
                "component_health": health.get('components', {}),
                "performance_metrics": {
                    "total_queries": stats.get('total_queries', 0),
                    "tier_hit_rates": {
                        "hot_memory": f"{stats.get('hot_hit_rate', 0) * 100:.1f}%",
                        "langchain_warm_cache": f"{stats.get('warm_hit_rate', 0) * 100:.1f}%",
                        "langchain_cold_storage": f"{stats.get('cold_hit_rate', 0) * 100:.1f}%"
                    },
                    "average_response_times": {
                        "hot_memory": f"{stats.get('hot_memory_avg_ms', 0):.2f}ms",
                        "langchain_warm_cache": f"{stats.get('warm_cache_avg_ms', 0):.2f}ms",
                        "langchain_cold_storage": f"{stats.get('cold_storage_avg_ms', 0):.2f}ms"
                    }
                },
                "storage_info": {
                    "hot_cache_size": health.get('components', {}).get('hot_cache', {}).get('item_count', 0),
                    "langchain_chromadb_size": health.get('components', {}).get('langchain_chromadb', {}).get('item_count', 0),
                    "total_capacity": "Unlimited (LangChain ChromaDB)",
                    "vector_dimensions": "Determined by SentenceTransformer model"
                },
                "routing_intelligence": {
                    "criteria": [
                        "Query frequency and recency patterns",
                        "Result quality scores and relevance", 
                        "Critical keywords detection",
                        "Access patterns and usage history",
                        "LangChain retriever performance metrics"
                    ],
                    "automatic_promotion": "High-quality results promoted to faster tiers",
                    "fallback_capability": "Graceful degradation between tiers",
                    "langchain_benefits": "Advanced retrieval strategies and document processing"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"System info endpoint error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Failed to retrieve system information",
                "message": str(e),
                "langchain_integration": "failed"
            }
        )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for production safety"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please check system health.",
            "request_path": str(request.url.path),
            "health_check": "/api/v1/health",
            "service": "LangChain Hybrid Knowledge Base Service"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting LangChain Hybrid Knowledge Base Service in development mode...")
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 