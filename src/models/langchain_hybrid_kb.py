"""
LangChain-based Hybrid Knowledge Base with Intelligent Routing
Combines in-memory performance with LangChain + ChromaDB scalability
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

# LangChain imports
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.retriever import BaseRetriever

import numpy as np
from sentence_transformers import SentenceTransformer

from ..models.schemas import KnowledgeBaseItem, Query
from ..data.knowledge_base import KNOWLEDGE_BASE
from ..utils.file_utils import get_model_path

logger = logging.getLogger(__name__)

class StorageTier(Enum):
    """Storage tier enumeration for routing decisions"""
    HOT_MEMORY = "hot_memory"      # 1-5ms, most critical/frequent
    WARM_CACHE = "warm_cache"      # 10-15ms, regular queries  
    COLD_STORAGE = "cold_storage"  # 25-50ms, comprehensive search
    MASSIVE_DATASET = "massive"    # 25-50ms, full corpus

@dataclass
class QueryMetrics:
    """Track query performance and usage patterns"""
    query_hash: str
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    avg_response_time: float = 0.0
    storage_tier_used: StorageTier = StorageTier.COLD_STORAGE
    result_quality_score: float = 0.0

@dataclass
class RoutingDecision:
    """Routing decision with reasoning"""
    tier: StorageTier
    reason: str
    confidence: float
    fallback_tier: Optional[StorageTier] = None

class IntelligentRouter:
    """Smart routing logic for LangChain hybrid knowledge base"""
    
    def __init__(self):
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.access_patterns = defaultdict(int)
        self.recent_queries = deque(maxlen=1000)
        self.lock = threading.RLock()
        
        # Routing thresholds (configurable)
        self.hot_access_threshold = 10      # Promote to hot after 10 accesses
        self.warm_access_threshold = 3      # Use warm cache after 3 accesses
        self.recent_access_window = 300     # 5 minutes for "recent"
        self.quality_threshold = 0.8        # Good result quality threshold
        
    def get_query_hash(self, query: str) -> str:
        """Generate consistent hash for query tracking"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:16]
    
    def record_query_result(self, query: str, response_time: float, 
                          tier_used: StorageTier, quality_score: float):
        """Record query metrics for learning"""
        with self.lock:
            query_hash = self.get_query_hash(query)
            
            if query_hash not in self.query_metrics:
                self.query_metrics[query_hash] = QueryMetrics(query_hash=query_hash)
            
            metrics = self.query_metrics[query_hash]
            metrics.access_count += 1
            metrics.last_access = datetime.now()
            metrics.storage_tier_used = tier_used
            metrics.result_quality_score = quality_score
            
            # Update rolling average response time
            if metrics.avg_response_time == 0:
                metrics.avg_response_time = response_time
            else:
                metrics.avg_response_time = (metrics.avg_response_time * 0.8 + 
                                           response_time * 0.2)
            
            self.recent_queries.append((query_hash, datetime.now()))
            self.access_patterns[query_hash] += 1
    
    def should_use_hot_memory(self, query: str) -> Tuple[bool, str]:
        """Determine if query should use hot memory tier"""
        query_hash = self.get_query_hash(query)
        
        # Check if frequently accessed
        if query_hash in self.query_metrics:
            metrics = self.query_metrics[query_hash]
            
            # High access count
            if metrics.access_count >= self.hot_access_threshold:
                return True, f"High access count: {metrics.access_count}"
            
            # Recent frequent access
            recent_time = datetime.now() - timedelta(seconds=self.recent_access_window)
            if metrics.last_access > recent_time and metrics.access_count >= 3:
                return True, f"Recent frequent access: {metrics.access_count} times"
            
            # High quality results with some usage
            if (metrics.result_quality_score >= self.quality_threshold and 
                metrics.access_count >= 2):
                return True, f"High quality results: {metrics.result_quality_score:.2f}"
        
        # Check for critical keywords
        critical_keywords = ['error', 'crash', 'urgent', 'production', 'down', 'fail']
        if any(keyword in query.lower() for keyword in critical_keywords):
            return True, "Critical keywords detected"
        
        return False, "No hot memory criteria met"
    
    def should_use_warm_cache(self, query: str) -> Tuple[bool, str]:
        """Determine if query should use warm cache tier"""
        query_hash = self.get_query_hash(query)
        
        if query_hash in self.query_metrics:
            metrics = self.query_metrics[query_hash]
            
            # Moderate access count
            if metrics.access_count >= self.warm_access_threshold:
                return True, f"Moderate access: {metrics.access_count} times"
            
            # Recent access
            recent_time = datetime.now() - timedelta(seconds=self.recent_access_window)
            if metrics.last_access > recent_time:
                return True, "Recent access detected"
        
        # Check for common troubleshooting patterns
        common_patterns = ['install', 'setup', 'config', 'npm', 'node', 'package']
        if any(pattern in query.lower() for pattern in common_patterns):
            return True, "Common troubleshooting pattern"
        
        return False, "No warm cache criteria met"
    
    def route_query(self, query: str) -> RoutingDecision:
        """Make intelligent routing decision"""
        
        # Check for hot memory usage
        use_hot, hot_reason = self.should_use_hot_memory(query)
        if use_hot:
            return RoutingDecision(
                tier=StorageTier.HOT_MEMORY,
                reason=hot_reason,
                confidence=0.9,
                fallback_tier=StorageTier.WARM_CACHE
            )
        
        # Check for warm cache usage
        use_warm, warm_reason = self.should_use_warm_cache(query)
        if use_warm:
            return RoutingDecision(
                tier=StorageTier.WARM_CACHE,
                reason=warm_reason,
                confidence=0.7,
                fallback_tier=StorageTier.COLD_STORAGE
            )
        
        # Default to cold storage for comprehensive search
        return RoutingDecision(
            tier=StorageTier.COLD_STORAGE,
            reason="Comprehensive search required",
            confidence=0.6,
            fallback_tier=None
        )

class HotMemoryCache:
    """Ultra-fast in-memory cache for hot data using SentenceTransformers directly"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.items: List[KnowledgeBaseItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model: Optional[SentenceTransformer] = None
        self.lock = threading.RLock()
        self.access_count = defaultdict(int)
        
    def initialize(self, model: SentenceTransformer, initial_items: List[KnowledgeBaseItem]):
        """Initialize with model and initial hot data"""
        with self.lock:
            self.model = model
            self.items = initial_items[:self.max_size]
            self._rebuild_embeddings()
            logger.info(f"Hot memory cache initialized with {len(self.items)} items")
    
    def _rebuild_embeddings(self):
        """Rebuild embeddings matrix"""
        if not self.items or not self.model:
            return
        
        texts = [f"{item.issue} {item.resolution}" for item in self.items]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        logger.debug(f"Rebuilt embeddings for {len(self.items)} items")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeBaseItem, float]]:
        """Ultra-fast similarity search"""
        if not self.items or not self.model or self.embeddings is None:
            return []
        
        start_time = time.time()
        
        with self.lock:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Compute similarities
            similarities = np.dot(query_embedding, self.embeddings.T).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [
                (self.items[idx], float(similarities[idx]))
                for idx in top_indices
                if similarities[idx] > 0.3  # Minimum similarity threshold
            ]
        
        response_time = (time.time() - start_time) * 1000
        logger.debug(f"Hot memory search completed in {response_time:.2f}ms")
        
        return results
    
    def add_item(self, item: KnowledgeBaseItem) -> bool:
        """Add item to hot cache (LRU eviction)"""
        with self.lock:
            # Check if already exists
            for existing in self.items:
                if existing.issue == item.issue:
                    return False
            
            # Add new item
            self.items.insert(0, item)
            
            # Evict if over capacity
            if len(self.items) > self.max_size:
                evicted = self.items.pop()
                logger.debug(f"Evicted item from hot cache: {evicted.issue[:50]}...")
            
            # Rebuild embeddings
            self._rebuild_embeddings()
            return True
    
    def promote_item(self, item: KnowledgeBaseItem):
        """Promote item to front of cache"""
        with self.lock:
            # Remove if exists
            self.items = [i for i in self.items if i.issue != item.issue]
            # Add to front
            self.items.insert(0, item)
            self._rebuild_embeddings()

class LangChainWarmCache:
    """Warm cache tier using LangChain + ChromaDB with local caching"""
    
    def __init__(self, cache_size: int = 5000):
        self.cache_size = cache_size
        self.local_cache: Dict[str, List[Tuple[KnowledgeBaseItem, float]]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(minutes=30)  # 30 minute TTL
        self.lock = threading.RLock()
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for query"""
        return f"{hashlib.md5(query.encode()).hexdigest()[:16]}_{top_k}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        return datetime.now() - self.cache_timestamps[cache_key] < self.cache_ttl
    
    async def search(self, query: str, langchain_retriever, top_k: int = 5) -> List[Tuple[KnowledgeBaseItem, float]]:
        """Search with LangChain retriever and local caching"""
        cache_key = self._get_cache_key(query, top_k)
        
        # Check local cache first
        with self.lock:
            if cache_key in self.local_cache and self._is_cache_valid(cache_key):
                logger.debug(f"Warm cache hit for query: {query[:50]}...")
                return self.local_cache[cache_key]
        
        # Query using LangChain retriever
        start_time = time.time()
        try:
            # Use LangChain's retriever interface
            documents = await asyncio.get_event_loop().run_in_executor(
                None, langchain_retriever.get_relevant_documents, query
            )
            
            # Convert LangChain documents to KnowledgeBaseItem format
            items = []
            for doc in documents[:top_k]:
                # Extract metadata from LangChain document
                metadata = doc.metadata
                similarity = metadata.get('score', 0.8)  # Default similarity if not provided
                
                item = KnowledgeBaseItem(
                    issue=metadata.get('issue', ''),
                    resolution=metadata.get('resolution', doc.page_content),
                    category=metadata.get('category', 'general'),
                    tags=metadata.get('tags', [])
                )
                items.append((item, similarity))
            
            # Cache results
            with self.lock:
                self.local_cache[cache_key] = items
                self.cache_timestamps[cache_key] = datetime.now()
                
                # Evict old cache entries if needed
                if len(self.local_cache) > self.cache_size:
                    oldest_key = min(self.cache_timestamps.keys(), 
                                   key=lambda k: self.cache_timestamps[k])
                    del self.local_cache[oldest_key]
                    del self.cache_timestamps[oldest_key]
            
            response_time = (time.time() - start_time) * 1000
            logger.debug(f"LangChain warm cache search completed in {response_time:.2f}ms")
            
            return items
            
        except Exception as e:
            logger.error(f"LangChain warm cache search failed: {e}")
            return []

class LangChainHybridKnowledgeBase:
    """Production-ready LangChain-based hybrid knowledge base with intelligent routing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.router = IntelligentRouter()
        self.hot_cache = HotMemoryCache(max_size=self.config.get('hot_cache_size', 1000))
        self.warm_cache = LangChainWarmCache(cache_size=self.config.get('warm_cache_size', 5000))
        
        # LangChain components
        self.embeddings: Optional[SentenceTransformerEmbeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self.retriever: Optional[BaseRetriever] = None
        
        # SentenceTransformer model for hot cache
        self.model: Optional[SentenceTransformer] = None
        
        # Performance tracking
        self.performance_stats = {
            'hot_hits': 0,
            'warm_hits': 0, 
            'cold_hits': 0,
            'total_queries': 0,
            'avg_response_times': defaultdict(list)
        }
        
        self.lock = threading.RLock()
        
    async def initialize(self):
        """Initialize the LangChain hybrid knowledge base"""
        try:
            # Load SentenceTransformer model for hot cache
            model_path = get_model_path()
            self.model = SentenceTransformer(model_path)
            logger.info(f"Loaded SentenceTransformer model from {model_path}")
            
            # Initialize LangChain embeddings (same model for consistency)
            self.embeddings = SentenceTransformerEmbeddings(model_name=model_path)
            logger.info("Initialized LangChain SentenceTransformer embeddings")
            
            # Initialize ChromaDB with LangChain
            await self._initialize_langchain_chromadb()
            
            # Initialize hot cache with current knowledge base
            self.hot_cache.initialize(self.model, KNOWLEDGE_BASE)
            
            # Populate ChromaDB if empty
            await self._populate_chromadb_if_needed()
            
            logger.info("LangChain hybrid knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LangChain hybrid knowledge base: {e}")
            raise
    
    async def _initialize_langchain_chromadb(self):
        """Initialize ChromaDB using LangChain"""
        try:
            persist_directory = self.config.get('chromadb_path', './chromadb_data')
            
            # Initialize Chroma vector store with LangChain
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory,
                collection_name="knowledge_base"
            )
            
            # Create retriever from vector store
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 10,  # Return more results for better selection
                    "score_threshold": 0.3  # Minimum similarity threshold
                }
            )
            
            logger.info("LangChain ChromaDB vector store initialized successfully")
                
        except Exception as e:
            logger.error(f"LangChain ChromaDB initialization failed: {e}")
            raise
    
    async def _populate_chromadb_if_needed(self):
        """Populate ChromaDB with initial data if empty using LangChain"""
        try:
            # Check if collection has documents
            existing_docs = self.vectorstore._collection.count()
            
            if existing_docs == 0:
                logger.info("Populating ChromaDB with initial knowledge base using LangChain...")
                await self.bulk_add_to_chromadb(KNOWLEDGE_BASE)
                logger.info(f"Added {len(KNOWLEDGE_BASE)} items to ChromaDB via LangChain")
            else:
                logger.info(f"ChromaDB already contains {existing_docs} items")
                
        except Exception as e:
            logger.error(f"Failed to populate ChromaDB via LangChain: {e}")
    
    async def bulk_add_to_chromadb(self, items: List[KnowledgeBaseItem]):
        """Bulk add items to ChromaDB using LangChain"""
        if not items:
            return
        
        try:
            # Convert KnowledgeBaseItems to LangChain Documents
            documents = []
            metadatas = []
            
            for item in items:
                # Create document content
                doc_content = f"Issue: {item.issue}\nResolution: {item.resolution}"
                
                # Create metadata
                metadata = {
                    'issue': item.issue,
                    'resolution': item.resolution,
                    'category': item.category,
                    'tags': item.tags,
                    'source': 'initial_knowledge_base'
                }
                
                documents.append(doc_content)
                metadatas.append(metadata)
            
            # Add documents to vector store using LangChain
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.vectorstore.add_texts,
                documents,
                metadatas
            )
            
            # Persist the vector store
            self.vectorstore.persist()
            
            logger.info(f"Successfully added {len(items)} items to ChromaDB via LangChain")
            
        except Exception as e:
            logger.error(f"Bulk add to ChromaDB via LangChain failed: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeBaseItem, float]]:
        """Main search method with intelligent routing using LangChain"""
        start_time = time.time()
        
        with self.lock:
            self.performance_stats['total_queries'] += 1
        
        try:
            # Get routing decision
            routing = self.router.route_query(query)
            logger.debug(f"Routing decision: {routing.tier.value} - {routing.reason}")
            
            results = []
            tier_used = routing.tier
            
            # Execute search based on routing decision
            if routing.tier == StorageTier.HOT_MEMORY:
                results = await self._search_hot_memory(query, top_k)
                if not results and routing.fallback_tier:
                    logger.debug("Hot memory failed, falling back to warm cache")
                    results = await self._search_warm_cache(query, top_k)
                    tier_used = StorageTier.WARM_CACHE
                    
            elif routing.tier == StorageTier.WARM_CACHE:
                results = await self._search_warm_cache(query, top_k)
                if not results and routing.fallback_tier:
                    logger.debug("Warm cache failed, falling back to cold storage")
                    results = await self._search_cold_storage(query, top_k)
                    tier_used = StorageTier.COLD_STORAGE
                    
            else:  # COLD_STORAGE or MASSIVE_DATASET
                results = await self._search_cold_storage(query, top_k)
                tier_used = StorageTier.COLD_STORAGE
            
            # Record performance metrics
            response_time = (time.time() - start_time) * 1000
            quality_score = self._calculate_quality_score(results)
            
            self.router.record_query_result(query, response_time, tier_used, quality_score)
            self._update_performance_stats(tier_used, response_time)
            
            # Promote good results to higher tiers
            await self._consider_promotion(query, results, tier_used, quality_score)
            
            logger.info(f"LangChain search completed in {response_time:.2f}ms using {tier_used.value}")
            return results
            
        except Exception as e:
            logger.error(f"LangChain search failed: {e}")
            return []
    
    async def _search_hot_memory(self, query: str, top_k: int) -> List[Tuple[KnowledgeBaseItem, float]]:
        """Search hot memory cache"""
        results = self.hot_cache.search(query, top_k)
        if results:
            with self.lock:
                self.performance_stats['hot_hits'] += 1
        return results
    
    async def _search_warm_cache(self, query: str, top_k: int) -> List[Tuple[KnowledgeBaseItem, float]]:
        """Search warm cache using LangChain retriever"""
        results = await self.warm_cache.search(query, self.retriever, top_k)
        if results:
            with self.lock:
                self.performance_stats['warm_hits'] += 1
        return results
    
    async def _search_cold_storage(self, query: str, top_k: int) -> List[Tuple[KnowledgeBaseItem, float]]:
        """Search cold storage using LangChain retriever directly"""
        try:
            start_time = time.time()
            
            # Use LangChain retriever for comprehensive search
            documents = await asyncio.get_event_loop().run_in_executor(
                None, self.retriever.get_relevant_documents, query
            )
            
            # Convert to KnowledgeBaseItem format
            items = []
            for doc in documents[:top_k]:
                metadata = doc.metadata
                similarity = metadata.get('score', 0.7)  # Default similarity
                
                item = KnowledgeBaseItem(
                    issue=metadata.get('issue', ''),
                    resolution=metadata.get('resolution', doc.page_content),
                    category=metadata.get('category', 'general'),
                    tags=metadata.get('tags', [])
                )
                items.append((item, similarity))
            
            if items:
                with self.lock:
                    self.performance_stats['cold_hits'] += 1
            
            response_time = (time.time() - start_time) * 1000
            logger.debug(f"LangChain cold storage search completed in {response_time:.2f}ms")
            
            return items
            
        except Exception as e:
            logger.error(f"LangChain cold storage search failed: {e}")
            return []
    
    def _calculate_quality_score(self, results: List[Tuple[KnowledgeBaseItem, float]]) -> float:
        """Calculate quality score for results"""
        if not results:
            return 0.0
        
        # Average similarity score weighted by position
        total_score = 0.0
        total_weight = 0.0
        
        for i, (_, similarity) in enumerate(results):
            weight = 1.0 / (i + 1)  # Higher weight for top results
            total_score += similarity * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _update_performance_stats(self, tier: StorageTier, response_time: float):
        """Update performance statistics"""
        with self.lock:
            self.performance_stats['avg_response_times'][tier.value].append(response_time)
            
            # Keep only recent measurements (last 1000)
            if len(self.performance_stats['avg_response_times'][tier.value]) > 1000:
                self.performance_stats['avg_response_times'][tier.value] = \
                    self.performance_stats['avg_response_times'][tier.value][-1000:]
    
    async def _consider_promotion(self, query: str, results: List[Tuple[KnowledgeBaseItem, float]], 
                                tier_used: StorageTier, quality_score: float):
        """Consider promoting good results to higher tiers"""
        if not results or quality_score < 0.8:
            return
        
        # Promote to hot cache if accessed from warm/cold and high quality
        if tier_used in [StorageTier.WARM_CACHE, StorageTier.COLD_STORAGE]:
            query_hash = self.router.get_query_hash(query)
            if (query_hash in self.router.query_metrics and 
                self.router.query_metrics[query_hash].access_count >= 3):
                
                # Promote top result to hot cache
                top_item, _ = results[0]
                self.hot_cache.promote_item(top_item)
                logger.debug(f"Promoted item to hot cache: {top_item.issue[:50]}...")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            stats = self.performance_stats.copy()
            
            # Calculate averages
            for tier, times in stats['avg_response_times'].items():
                if times:
                    stats[f'{tier}_avg_ms'] = sum(times) / len(times)
                    stats[f'{tier}_p95_ms'] = np.percentile(times, 95) if len(times) > 1 else times[0]
                else:
                    stats[f'{tier}_avg_ms'] = 0
                    stats[f'{tier}_p95_ms'] = 0
            
            # Calculate hit rates
            total = stats['total_queries']
            if total > 0:
                stats['hot_hit_rate'] = stats['hot_hits'] / total
                stats['warm_hit_rate'] = stats['warm_hits'] / total  
                stats['cold_hit_rate'] = stats['cold_hits'] / total
            
            return stats
    
    async def add_knowledge_item(self, item: KnowledgeBaseItem) -> bool:
        """Add new knowledge item to the LangChain system"""
        try:
            # Create LangChain document
            doc_content = f"Issue: {item.issue}\nResolution: {item.resolution}"
            metadata = {
                'issue': item.issue,
                'resolution': item.resolution,
                'category': item.category,
                'tags': item.tags,
                'source': 'user_added',
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to vector store using LangChain
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.vectorstore.add_texts,
                [doc_content],
                [metadata]
            )
            
            # Persist changes
            self.vectorstore.persist()
            
            logger.info(f"Added new knowledge item via LangChain: {item.issue[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add knowledge item via LangChain: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for LangChain hybrid system"""
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check hot cache
            health['components']['hot_cache'] = {
                'status': 'healthy' if self.hot_cache.model else 'unhealthy',
                'item_count': len(self.hot_cache.items),
                'max_size': self.hot_cache.max_size
            }
            
            # Check LangChain ChromaDB
            try:
                count = self.vectorstore._collection.count()
                health['components']['langchain_chromadb'] = {
                    'status': 'healthy',
                    'item_count': count,
                    'vector_store': 'Chroma',
                    'embeddings': 'SentenceTransformerEmbeddings'
                }
            except Exception as e:
                health['components']['langchain_chromadb'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health['status'] = 'degraded'
            
            # Check models
            health['components']['models'] = {
                'sentence_transformer': 'healthy' if self.model else 'unhealthy',
                'langchain_embeddings': 'healthy' if self.embeddings else 'unhealthy',
                'model_path': get_model_path() if self.model else None
            }
            
            # Check retriever
            health['components']['retriever'] = {
                'status': 'healthy' if self.retriever else 'unhealthy',
                'type': 'LangChain BaseRetriever'
            }
            
            # Add performance stats
            health['performance'] = self.get_performance_stats()
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health

# Global instance
langchain_hybrid_kb: Optional[LangChainHybridKnowledgeBase] = None

async def get_langchain_hybrid_kb() -> LangChainHybridKnowledgeBase:
    """Get or create LangChain hybrid knowledge base instance"""
    global langchain_hybrid_kb
    if langchain_hybrid_kb is None:
        langchain_hybrid_kb = LangChainHybridKnowledgeBase()
        await langchain_hybrid_kb.initialize()
    return langchain_hybrid_kb 