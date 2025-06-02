#!/usr/bin/env python3
"""
Populate Knowledge Base with Synthetic Data
Uses the same generator as the test suite to add diverse content
"""

import asyncio
import json
import requests
from test_comprehensive import SyntheticDataGenerator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def populate_knowledge_base(count: int = 50, base_url: str = "http://localhost:8000"):
    """Populate the knowledge base with synthetic data"""
    generator = SyntheticDataGenerator()
    
    # Generate diverse synthetic items
    logger.info(f"Generating {count} synthetic knowledge base items...")
    items = generator.generate_synthetic_items(count)
    
    # Add items to knowledge base
    session = requests.Session()
    added_count = 0
    failed_count = 0
    
    for i, item in enumerate(items):
        try:
            response = session.post(
                f"{base_url}/api/v1/knowledge",
                json=item,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                added_count += 1
                if added_count % 10 == 0:
                    logger.info(f"Added {added_count}/{count} items...")
            else:
                failed_count += 1
                logger.warning(f"Failed to add item {i+1}: HTTP {response.status_code}")
                
        except Exception as e:
            failed_count += 1
            logger.error(f"Error adding item {i+1}: {e}")
    
    # Summary
    logger.info(f"✅ Populate complete: {added_count} added, {failed_count} failed")
    
    # Test a few queries to verify
    logger.info("Testing queries with new data...")
    test_queries = [
        "python import error",
        "kubernetes pod not starting", 
        "aws s3 access denied",
        "database connection timeout",
        "docker container restart"
    ]
    
    for query in test_queries:
        try:
            response = session.post(
                f"{base_url}/api/v1/troubleshoot",
                json={"text": query},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results_count = len(data.get("results", []))
                search_time = data.get("search_time_ms", 0)
                logger.info(f"Query '{query}': {results_count} results in {search_time:.2f}ms")
            else:
                logger.warning(f"Query '{query}' failed: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")
    
    return {"added": added_count, "failed": failed_count, "total": count}

if __name__ == "__main__":
    result = asyncio.run(populate_knowledge_base(count=100))
    print(f"\n🎉 Knowledge Base Population Results:")
    print(f"Successfully added: {result['added']}")
    print(f"Failed: {result['failed']} ")
    print(f"Total attempted: {result['total']}")
    print(f"Success rate: {result['added']/result['total']*100:.1f}%") 