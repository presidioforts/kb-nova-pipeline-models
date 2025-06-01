"""
API Routes for Hybrid Knowledge Base Service
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from ..models.schemas import Query, TrainingPair, TrainingData, KnowledgeBaseItem
from ..models.hybrid_knowledge_base import get_hybrid_kb, HybridKnowledgeBase
from ..models.sentence_transformer import SentenceTransformerModel, get_model

logger = logging.getLogger(__name__)

router = APIRouter()

# Training job storage (in production, use Redis or database)
training_jobs: Dict[str, Dict[str, Any]] = {}

@router.post("/troubleshoot")
async def troubleshoot_query(
    query: Query,
    hybrid_kb: HybridKnowledgeBase = Depends(get_hybrid_kb)
) -> JSONResponse:
    """
    Intelligent troubleshooting with hybrid knowledge base routing
    
    Routes queries to optimal storage tier:
    - Hot Memory: 1-5ms for critical/frequent queries (SentenceTransformers)
    - Warm Cache: 10-15ms for regular queries (ChromaDB + Cache)
    - Cold Storage: 25-50ms for comprehensive search (ChromaDB)
    """
    try:
        start_time = datetime.now()
        
        # Search using intelligent routing
        results = await hybrid_kb.search(query.text, top_k=5)
        
        if not results:
            logger.warning(f"No results found for query: {query.text}")
            return JSONResponse(
                status_code=200,
                content={
                    "query": query.text,
                    "results": [],
                    "message": "No matching troubleshooting solutions found. Try rephrasing your query.",
                    "search_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "routing_info": {
                        "tier_used": "no_results",
                        "total_items_searched": 0,
                        "chromadb_integration": True
                    }
                }
            )
        
        # Format results
        formatted_results = []
        for item, similarity in results:
            formatted_results.append({
                "issue": item.issue,
                "resolution": item.resolution,
                "category": item.category,
                "tags": item.tags,
                "similarity_score": round(similarity, 4),
                "confidence": "high" if similarity > 0.8 else "medium" if similarity > 0.6 else "low"
            })
        
        # Get performance stats for response
        perf_stats = hybrid_kb.get_performance_stats()
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response_data = {
            "query": query.text,
            "results": formatted_results,
            "search_time_ms": round(search_time, 2),
            "routing_info": {
                "total_items_searched": len(results),
                "chromadb_integration": True,
                "vector_store": "ChromaDB",
                "embeddings": "SentenceTransformers",
                "performance_tier_hit_rates": {
                    "hot_memory": round(perf_stats.get('hot_hit_rate', 0) * 100, 1),
                    "warm_cache": round(perf_stats.get('warm_hit_rate', 0) * 100, 1),
                    "cold_storage": round(perf_stats.get('cold_hit_rate', 0) * 100, 1)
                }
            },
            "metadata": {
                "timestamp": start_time.isoformat(),
                "total_queries_processed": perf_stats.get('total_queries', 0),
                "system_performance": "optimal" if search_time < 10 else "good" if search_time < 50 else "acceptable",
                "architecture": "In-Memory + ChromaDB Hybrid"
            }
        }
        
        logger.info(f"Hybrid troubleshoot query processed in {search_time:.2f}ms: {query.text[:50]}...")
        return JSONResponse(status_code=200, content=response_data)
        
    except Exception as e:
        logger.error(f"Error processing troubleshoot query: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "overall_status": "unhealthy",
                "error": "Internal server error",
                "timestamp": datetime.now().isoformat(),
                "chromadb_integration": "failed"
            }
        )

@router.post("/train")
async def start_training(
    training_data: TrainingData,
    background_tasks: BackgroundTasks,
    model: SentenceTransformerModel = Depends(get_model),
    hybrid_kb: HybridKnowledgeBase = Depends(get_hybrid_kb)
) -> JSONResponse:
    """
    Start fine-tuning with hybrid knowledge base integration
    
    Trains the model and updates both in-memory and ChromaDB storage
    """
    try:
        if not training_data.pairs:
            raise HTTPException(status_code=400, detail="No training pairs provided")
        
        # Generate job ID
        job_id = f"train_{int(datetime.now().timestamp())}_{len(training_data.pairs)}"
        
        # Initialize job status
        training_jobs[job_id] = {
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "training_pairs_count": len(training_data.pairs),
            "progress": 0,
            "current_step": "initializing",
            "estimated_completion": None,
            "error": None,
            "model_version": None,
            "hybrid_integration": {
                "hot_cache_updated": False,
                "chromadb_updated": False,
                "new_items_added": 0,
                "vector_store": "ChromaDB",
                "embeddings": "SentenceTransformers"
            }
        }
        
        # Start training in background
        background_tasks.add_task(
            run_hybrid_training,
            job_id,
            training_data.pairs,
            model,
            hybrid_kb
        )
        
        logger.info(f"Started hybrid training job {job_id} with {len(training_data.pairs)} pairs")
        
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "started",
                "message": f"Hybrid training started with {len(training_data.pairs)} pairs",
                "estimated_duration_minutes": len(training_data.pairs) // 10 + 5,  # Rough estimate
                "hybrid_features": {
                    "will_update_hot_cache": True,
                    "will_update_chromadb": True,
                    "intelligent_routing_enabled": True,
                    "vector_store": "ChromaDB",
                    "embeddings": "SentenceTransformers"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "overall_status": "unhealthy",
                "error": "Training service error",
                "timestamp": datetime.now().isoformat(),
                "chromadb_integration": "failed"
            }
        )

@router.get("/train/{job_id}")
async def get_training_status(job_id: str) -> JSONResponse:
    """
    Get training job status with hybrid system integration details
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job_info = training_jobs[job_id].copy()
    
    # Add additional context for completed jobs
    if job_info["status"] == "completed":
        job_info["next_steps"] = {
            "recommendation": "Test the updated model with sample queries",
            "hybrid_benefits": "New model integrated into both hot cache and ChromaDB",
            "performance_impact": "Improved accuracy with maintained speed via intelligent routing"
        }
    
    return JSONResponse(status_code=200, content=job_info)

@router.get("/health")
async def health_check(
    hybrid_kb: HybridKnowledgeBase = Depends(get_hybrid_kb)
) -> JSONResponse:
    """
    Comprehensive health check for hybrid knowledge base system
    """
    try:
        # Get hybrid system health
        health_info = await hybrid_kb.health_check()
        
        # Add API-level health info
        health_info["api"] = {
            "status": "healthy",
            "active_training_jobs": len([j for j in training_jobs.values() if j["status"] == "running"]),
            "total_training_jobs": len(training_jobs)
        }
        
        # Add hybrid specific info
        health_info["hybrid_integration"] = {
            "status": "active",
            "vector_store": "ChromaDB",
            "embeddings": "SentenceTransformers",
            "retriever": "BaseRetriever with similarity_score_threshold"
        }
        
        # Determine overall status
        component_statuses = [
            health_info["components"].get("hot_cache", {}).get("status", "unknown"),
            health_info["components"].get("chromadb", {}).get("status", "unknown"),
            health_info["components"].get("models", {}).get("sentence_transformer", "unknown"),
            health_info["components"].get("retriever", {}).get("status", "unknown"),
            health_info["api"]["status"]
        ]
        
        if all(status == "healthy" for status in component_statuses):
            overall_status = "healthy"
        elif any(status == "unhealthy" for status in component_statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        health_info["overall_status"] = overall_status
        
        return JSONResponse(
            status_code=200 if overall_status == "healthy" else 503,
            content=health_info
        )
        
    except Exception as e:
        logger.error(f"Hybrid health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "overall_status": "unhealthy",
                "error": "Health check failed",
                "timestamp": datetime.now().isoformat(),
                "chromadb_integration": "failed"
            }
        )

@router.get("/performance")
async def get_performance_metrics(
    hybrid_kb: HybridKnowledgeBase = Depends(get_hybrid_kb)
) -> JSONResponse:
    """
    Get detailed performance metrics for the hybrid system
    """
    try:
        stats = hybrid_kb.get_performance_stats()
        
        # Add performance analysis
        analysis = {
            "performance_grade": "A",  # Default
            "recommendations": [],
            "tier_efficiency": {},
            "hybrid_performance": {}
        }
        
        # Analyze hot cache performance
        hot_hit_rate = stats.get('hot_hit_rate', 0)
        if hot_hit_rate > 0.8:
            analysis["tier_efficiency"]["hot_cache"] = "excellent"
        elif hot_hit_rate > 0.6:
            analysis["tier_efficiency"]["hot_cache"] = "good"
        else:
            analysis["tier_efficiency"]["hot_cache"] = "needs_improvement"
            analysis["recommendations"].append("Consider increasing hot cache size or adjusting promotion criteria")
        
        # Analyze hybrid performance
        warm_hit_rate = stats.get('warm_hit_rate', 0)
        cold_hit_rate = stats.get('cold_hit_rate', 0)
        
        analysis["hybrid_performance"] = {
            "warm_cache_efficiency": "excellent" if warm_hit_rate > 0.6 else "good" if warm_hit_rate > 0.3 else "needs_improvement",
            "cold_storage_usage": "balanced" if cold_hit_rate < 0.5 else "high",
            "vector_store": "ChromaDB",
            "embeddings": "SentenceTransformers"
        }
        
        # Analyze response times
        hot_avg = stats.get('hot_memory_avg_ms', 0)
        if hot_avg > 10:
            analysis["recommendations"].append("Hot cache response time is higher than expected")
        
        warm_avg = stats.get('warm_cache_avg_ms', 0)
        if warm_avg > 30:
            analysis["recommendations"].append("Hybrid warm cache may need optimization")
        
        cold_avg = stats.get('cold_storage_avg_ms', 0)
        if cold_avg > 100:
            analysis["recommendations"].append("Hybrid cold storage performance could be improved")
        
        # Overall grade
        if hot_hit_rate > 0.8 and hot_avg < 5:
            analysis["performance_grade"] = "A+"
        elif hot_hit_rate > 0.6 and hot_avg < 10:
            analysis["performance_grade"] = "A"
        elif hot_hit_rate > 0.4:
            analysis["performance_grade"] = "B"
        else:
            analysis["performance_grade"] = "C"
        
        return JSONResponse(
            status_code=200,
            content={
                "performance_metrics": stats,
                "analysis": analysis,
                "hybrid_integration": {
                    "vector_store": "ChromaDB",
                    "embeddings": "SentenceTransformers",
                    "intelligent_routing": True
                },
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting hybrid performance metrics: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "overall_status": "unhealthy",
                "error": "Internal server error",
                "timestamp": datetime.now().isoformat(),
                "chromadb_integration": "failed"
            }
        )

@router.post("/knowledge")
async def add_knowledge_item(
    item: KnowledgeBaseItem,
    hybrid_kb: HybridKnowledgeBase = Depends(get_hybrid_kb)
) -> JSONResponse:
    """
    Add new knowledge item to the hybrid system
    """
    try:
        success = await hybrid_kb.add_knowledge_item(item)
        
        if success:
            return JSONResponse(
                status_code=201,
                content={
                    "message": "Knowledge item added successfully via hybrid",
                    "item": {
                        "issue": item.issue,
                        "category": item.category,
                        "tags": item.tags
                    },
                    "storage_info": {
                        "added_to_chromadb": True,
                        "vector_store": "ChromaDB",
                        "embeddings": "SentenceTransformers",
                        "will_be_promoted_to_cache": "based on usage patterns"
                    }
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to add knowledge item to hybrid system"
            )
            
    except Exception as e:
        logger.error(f"Error adding knowledge item to hybrid: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "overall_status": "unhealthy",
                "error": "Internal server error",
                "timestamp": datetime.now().isoformat(),
                "chromadb_integration": "failed"
            }
        )

async def run_hybrid_training(
    job_id: str,
    training_pairs: List[TrainingPair],
    model: SentenceTransformerModel,
    hybrid_kb: HybridKnowledgeBase
):
    """
    Background task for training with hybrid system integration
    """
    try:
        # Update job status
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["current_step"] = "preparing_training_data"
        training_jobs[job_id]["progress"] = 10
        
        # Run the actual training
        await asyncio.sleep(1)  # Simulate preparation
        training_jobs[job_id]["current_step"] = "training_model"
        training_jobs[job_id]["progress"] = 30
        
        # Train the model
        model_version = await model.fine_tune(training_pairs)
        
        training_jobs[job_id]["progress"] = 70
        training_jobs[job_id]["current_step"] = "updating_hybrid_system"
        
        # Extract new knowledge items from training pairs
        new_items = []
        for pair in training_pairs:
            # Convert training pairs to knowledge items
            item = KnowledgeBaseItem(
                issue=pair.query,
                resolution=pair.positive_example,
                category="trained",
                tags=["fine_tuned", "user_provided", "hybrid"]
            )
            new_items.append(item)
        
        # Add new items to hybrid system
        added_count = 0
        for item in new_items:
            if await hybrid_kb.add_knowledge_item(item):
                added_count += 1
        
        training_jobs[job_id]["progress"] = 90
        training_jobs[job_id]["current_step"] = "finalizing"
        
        # Update job completion
        training_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "current_step": "completed",
            "completion_time": datetime.now().isoformat(),
            "model_version": model_version,
            "hybrid_integration": {
                "hot_cache_updated": True,
                "chromadb_updated": True,
                "new_items_added": added_count,
                "vector_store": "ChromaDB",
                "embeddings": "SentenceTransformers"
            }
        })
        
        logger.info(f"Hybrid training job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Hybrid training job {job_id} failed: {e}")
        training_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completion_time": datetime.now().isoformat()
        })