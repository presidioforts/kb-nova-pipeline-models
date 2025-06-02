"""
Hybrid Knowledge Base with Intelligent Routing
Combines in-memory performance with ChromaDB scalability
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

import chromadb
# Remove deprecated Settings import
# from chromadb.config import Settings
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
    """Smart routing logic for hybrid knowledge base"""
    
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
    """Ultra-fast in-memory cache for hot data"""
    
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
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Threshold for relevance
                    results.append((self.items[idx], float(similarities[idx])))
                    self.access_count[idx] += 1
        
        logger.debug(f"Hot cache search took {(time.time() - start_time) * 1000:.2f}ms")
        return results
    
    def add_item(self, item: KnowledgeBaseItem) -> bool:
        """Add item to hot cache"""
        with self.lock:
            if len(self.items) >= self.max_size:
                # Remove least accessed item
                min_access_idx = min(range(len(self.items)), 
                                   key=lambda i: self.access_count[i])
                self.items.pop(min_access_idx)
                
                # Shift access counts
                new_counts = defaultdict(int)
                for i, count in self.access_count.items():
                    if i < min_access_idx:
                        new_counts[i] = count
                    elif i > min_access_idx:
                        new_counts[i-1] = count
                self.access_count = new_counts
            
            self.items.append(item)
            self._rebuild_embeddings()
            logger.debug(f"Added item to hot cache: {item.issue[:50]}...")
            return True
    
    def promote_item(self, item: KnowledgeBaseItem):
        """Promote item to front of hot cache"""
        with self.lock:
            if item not in self.items:
                self.add_item(item)
            else:
                idx = self.items.index(item)
                self.access_count[idx] += 10  # Boost access count

class WarmCache:
    """Intermediate cache layer for ChromaDB queries"""
    
    def __init__(self, cache_size: int = 5000):
        self.cache: Dict[str, Tuple[List[Tuple[KnowledgeBaseItem, float]], datetime]] = {}
        self.cache_size = cache_size
        self.lock = threading.RLock()
        self.ttl_seconds = 3600  # 1 hour TTL
        
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for query"""
        return f"{hashlib.md5(query.encode()).hexdigest()}_{top_k}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.cache:
            return False
        _, timestamp = self.cache[cache_key]
        return (datetime.now() - timestamp).seconds < self.ttl_seconds
    
    def _parse_tags_from_metadata(self, tags_value) -> List[str]:
        """Convert tags from ChromaDB metadata format back to list"""
        if isinstance(tags_value, str) and tags_value:
            return [tag.strip() for tag in tags_value.split(',') if tag.strip()]
        elif isinstance(tags_value, list):
            return tags_value
        else:
            return []
    
    async def search(self, query: str, chromadb_client, top_k: int = 5) -> List[Tuple[KnowledgeBaseItem, float]]:
        """Search with caching layer"""
        cache_key = self._get_cache_key(query, top_k)
        
        # Check cache first
        with self.lock:
            if self._is_cache_valid(cache_key):
                results, _ = self.cache[cache_key]
                logger.debug(f"Warm cache hit for query: {query[:30]}...")
                return results
        
        # Cache miss - query ChromaDB
        try:
            if chromadb_client is None:
                logger.warning("ChromaDB client not available for warm cache")
                return []
                
            collection = chromadb_client.get_collection("hybrid_kb")
            chroma_results = collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            items = []
            if chroma_results['documents'] and chroma_results['documents'][0]:
                for doc, metadata, distance in zip(
                    chroma_results['documents'][0],
                    chroma_results['metadatas'][0],
                    chroma_results['distances'][0]
                ):
                    similarity = 1.0 - distance
                    if similarity > 0.3:
                        item = KnowledgeBaseItem(
                            issue=metadata.get('issue', ''),
                            resolution=metadata.get('resolution', ''),
                            category=metadata.get('category', 'general'),
                            tags=self._parse_tags_from_metadata(metadata.get('tags', []))
                        )
                        items.append((item, similarity))
            
            # Cache results
            with self.lock:
                if len(self.cache) >= self.cache_size:
                    # Remove oldest entry
                    oldest_key = min(self.cache.keys(), 
                                   key=lambda k: self.cache[k][1])
                    del self.cache[oldest_key]
                
                self.cache[cache_key] = (items, datetime.now())
            
            logger.debug(f"Warm cache stored {len(items)} results for: {query[:30]}...")
            return items
            
        except Exception as e:
            logger.error(f"Warm cache search failed: {e}")
            return []

class HybridKnowledgeBase:
    """Hybrid knowledge base with intelligent routing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model: Optional[SentenceTransformer] = None
        self.chromadb_client = None
        self.collection = None
        self.collection_name = "hybrid_kb"
        
        # Initialize components
        self.hot_cache = HotMemoryCache(max_size=1000)
        self.warm_cache = WarmCache(cache_size=5000)
        self.router = IntelligentRouter()
        
        # Performance tracking
        self.lock = threading.RLock()
        self.performance_stats = {
            'total_queries': 0,
            'hot_hits': 0,
            'warm_hits': 0,
            'cold_hits': 0,
            'avg_response_times': defaultdict(list)
        }

    def _parse_tags_from_metadata(self, tags_value) -> List[str]:
        """Convert tags from ChromaDB metadata format back to list"""
        if isinstance(tags_value, str) and tags_value:
            return [tag.strip() for tag in tags_value.split(',') if tag.strip()]
        elif isinstance(tags_value, list):
            return tags_value
        else:
            return []

    async def initialize(self):
        """Initialize the hybrid knowledge base"""
        try:
            # Load model
            model_path = get_model_path()
            self.model = SentenceTransformer(model_path)
            logger.info(f"Loaded model from {model_path}")
            
            # Initialize ChromaDB
            await self._initialize_chromadb()
            
            # Initialize hot cache with current knowledge base
            self.hot_cache.initialize(self.model, KNOWLEDGE_BASE)
            
            # Populate ChromaDB if empty
            await self._populate_chromadb_if_needed()
            
            logger.info("Hybrid knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid knowledge base: {e}")
            raise

    async def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Use new ChromaDB client initialization (no deprecated Settings)
            persist_directory = self.config.get('chromadb_path', './chromadb_data')
            
            # Create persistent client with the new pattern
            self.chromadb_client = chromadb.PersistentClient(path=persist_directory)
            
            # Create or get collection
            try:
                self.collection = self.chromadb_client.get_collection(self.collection_name)
                logger.info(f"Connected to existing ChromaDB collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.chromadb_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Hybrid knowledge base storage"}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise

    async def _populate_chromadb_if_needed(self):
        """Populate ChromaDB with initial data if empty"""
        try:
            count = self.collection.count()
            if count == 0:
                logger.info("Populating ChromaDB with initial knowledge base...")
                await self.bulk_add_to_chromadb(KNOWLEDGE_BASE)
                logger.info(f"Added {len(KNOWLEDGE_BASE)} items to ChromaDB")
            else:
                logger.info(f"ChromaDB already contains {count} items")
                
        except Exception as e:
            logger.error(f"Failed to populate ChromaDB: {e}")

    async def bulk_add_to_chromadb(self, items: List[KnowledgeBaseItem]):
        """Bulk add items to ChromaDB"""
        if not items:
            return
        
        try:
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, item in enumerate(items):
                doc_text = f"{item.issue} {item.resolution}"
                documents.append(doc_text)
                metadatas.append({
                    'issue': item.issue,
                    'resolution': item.resolution,
                    'category': item.category,
                    'tags': ','.join(item.tags) if item.tags else ''  # Convert list to comma-separated string
                })
                ids.append(f"kb_item_{i}_{hashlib.md5(item.issue.encode()).hexdigest()[:8]}")
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(items)} items to ChromaDB")
            
        except Exception as e:
            logger.error(f"Bulk add to ChromaDB failed: {e}")
            raise

    async def search(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeBaseItem, float]]:
        """Main search method with intelligent routing"""
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
            
            logger.info(f"Search completed in {response_time:.2f}ms using {tier_used.value}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _search_hot_memory(self, query: str, top_k: int) -> List[Tuple[KnowledgeBaseItem, float]]:
        """Search hot memory cache"""
        results = self.hot_cache.search(query, top_k)
        if results:
            with self.lock:
                self.performance_stats['hot_hits'] += 1
        return results
    
    async def _search_warm_cache(self, query: str, top_k: int) -> List[Tuple[KnowledgeBaseItem, float]]:
        """Search warm cache"""
        results = await self.warm_cache.search(query, self.chromadb_client, top_k)
        if results:
            with self.lock:
                self.performance_stats['warm_hits'] += 1
        return results
    
    async def _search_cold_storage(self, query: str, top_k: int) -> List[Tuple[KnowledgeBaseItem, float]]:
        """Search cold storage (direct ChromaDB)"""
        try:
            collection = self.chromadb_client.get_collection(self.collection_name)
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            items = []
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                ):
                    similarity = 1.0 - distance
                    if similarity > 0.3:
                        item = KnowledgeBaseItem(
                            issue=metadata.get('issue', ''),
                            resolution=metadata.get('resolution', ''),
                            category=metadata.get('category', 'general'),
                            tags=self._parse_tags_from_metadata(metadata.get('tags', []))
                        )
                        items.append((item, similarity))
            
            if items:
                with self.lock:
                    self.performance_stats['cold_hits'] += 1
            
            return items
            
        except Exception as e:
            logger.error(f"Cold storage search failed: {e}")
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
        """Add new knowledge item to the system"""
        try:
            # Add to ChromaDB
            doc_text = f"{item.issue} {item.resolution}"
            item_id = f"kb_item_{int(time.time())}_{hashlib.md5(item.issue.encode()).hexdigest()[:8]}"
            
            self.collection.add(
                documents=[doc_text],
                metadatas=[{
                    'issue': item.issue,
                    'resolution': item.resolution,
                    'category': item.category,
                    'tags': ','.join(item.tags) if item.tags else ''  # Convert list to comma-separated string
                }],
                ids=[item_id]
            )
            
            logger.info(f"Added new knowledge item: {item.issue[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add knowledge item: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
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
            
            # Check ChromaDB
            try:
                count = self.collection.count()
                health['components']['chromadb'] = {
                    'status': 'healthy',
                    'item_count': count
                }
            except Exception as e:
                health['components']['chromadb'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health['status'] = 'degraded'
            
            # Check model
            health['components']['model'] = {
                'status': 'healthy' if self.model else 'unhealthy',
                'model_path': get_model_path() if self.model else None
            }
            
            # Add performance stats
            health['performance'] = self.get_performance_stats()
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health

# Global instance
hybrid_kb: Optional[HybridKnowledgeBase] = None

async def get_hybrid_kb() -> HybridKnowledgeBase:
    """Get or create hybrid knowledge base instance"""
    global hybrid_kb
    if hybrid_kb is None:
        hybrid_kb = HybridKnowledgeBase()
        await hybrid_kb.initialize()
    return hybrid_kb 