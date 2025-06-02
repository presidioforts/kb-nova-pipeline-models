#!/usr/bin/env python3
"""
Integration Tests for Simplified Knowledge Base Service
Tests core endpoints and functionality
"""

import pytest
import requests
import json
import time
from typing import Dict, Any
import concurrent.futures

# Test configuration
BASE_URL = "http://localhost:8080"
TEST_TIMEOUT = 30

class TestKnowledgeBaseAPI:
    """Integration tests for the knowledge base API"""
    
    @pytest.fixture(scope="class")
    def api_client(self):
        """Setup API client and verify service is running"""
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            assert response.status_code == 200
            return requests.Session()
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running. Start with: python run.py")
    
    def test_service_health(self, api_client):
        """Test basic service health"""
        response = api_client.get(f"{BASE_URL}/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "Simplified Knowledge Base"
        assert data["version"] == "3.0.0"
        assert data["status"] == "healthy"
        assert "endpoints" in data
    
    def test_troubleshoot_endpoint(self, api_client):
        """Test troubleshooting search functionality"""
        # Test basic query
        query_data = {"text": "npm install hanging"}
        response = api_client.post(f"{BASE_URL}/troubleshoot", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "problem" in data
        assert "solution" in data
        assert "similarity_score" in data
        assert "source" in data
        assert data["problem"] == query_data["text"]
    
    def test_training_workflow(self, api_client):
        """Test complete training workflow"""
        # Prepare training data
        training_data = {
            "data": [
                {
                    "problem": "Docker container won't start",
                    "solution": "Check docker logs and verify port availability",
                    "category": "docker",
                    "source": "test"
                },
                {
                    "problem": "Python import error",
                    "solution": "Verify module installation and PYTHONPATH",
                    "category": "python",
                    "source": "test"
                }
            ]
        }
        
        # Start training
        response = api_client.post(f"{BASE_URL}/train", json=training_data)
        assert response.status_code == 200
        
        train_result = response.json()
        assert "job_id" in train_result
        assert train_result["status"] == "queued"
        job_id = train_result["job_id"]
        
        # Wait for training to complete
        max_wait = TEST_TIMEOUT
        while max_wait > 0:
            response = api_client.get(f"{BASE_URL}/train/{job_id}")
            assert response.status_code == 200
            
            status_data = response.json()
            if status_data["status"] == "completed":
                assert "message" in status_data
                assert "completion_time" in status_data
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Training failed: {status_data.get('message', 'Unknown error')}")
            
            time.sleep(1)
            max_wait -= 1
        
        if max_wait <= 0:
            pytest.fail("Training did not complete within timeout")
        
        # Test that trained data is searchable
        query_data = {"text": "Docker container issue"}
        response = api_client.post(f"{BASE_URL}/troubleshoot", json=query_data)
        assert response.status_code == 200
        
        search_result = response.json()
        assert search_result["similarity_score"] > 0.3  # Should find some similarity
    
    def test_document_bulk_import(self, api_client):
        """Test bulk document import functionality"""
        documents = [
            {
                "title": "Git Troubleshooting Guide",
                "content": "When git push fails, check your remote URL. Use git remote -v to verify. If authentication fails, check your SSH keys or personal access tokens.",
                "category": "git",
                "source": "test"
            },
            {
                "title": "Database Connection Issues",
                "content": "Database connection timeouts can be caused by network issues. Check your connection string and firewall settings. Verify the database server is running.",
                "category": "database",
                "source": "test"
            }
        ]
        
        response = api_client.post(f"{BASE_URL}/documents/bulk", json=documents)
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "success"
        assert result["documents_added"] == 2
        assert result["chunks_created"] > 0
        
        # Test searching the imported documents
        query_data = {"text": "git push authentication problem"}
        response = api_client.post(f"{BASE_URL}/troubleshoot", json=query_data)
        assert response.status_code == 200
        
        search_result = response.json()
        assert search_result["similarity_score"] > 0.3
        assert "git" in search_result["solution"].lower() or "ssh" in search_result["solution"].lower()
    
    def test_training_job_not_found(self, api_client):
        """Test training job status for non-existent job"""
        fake_job_id = "non-existent-job-id"
        response = api_client.get(f"{BASE_URL}/train/{fake_job_id}")
        assert response.status_code == 404
    
    def test_empty_training_data(self, api_client):
        """Test training with empty data"""
        empty_data = {"data": []}
        response = api_client.post(f"{BASE_URL}/train", json=empty_data)
        assert response.status_code == 400
    
    def test_malformed_requests(self, api_client):
        """Test API with malformed requests"""
        # Missing required field
        response = api_client.post(f"{BASE_URL}/troubleshoot", json={})
        assert response.status_code == 422  # Validation error
        
        # Invalid JSON
        response = api_client.post(
            f"{BASE_URL}/troubleshoot",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

class TestPerformance:
    """Basic performance tests"""
    
    @pytest.fixture(scope="class")
    def api_client(self):
        """Setup API client"""
        try:
            response = requests.get(f"{BASE_URL}/", timeout=5)
            assert response.status_code == 200
            return requests.Session()
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running")
    
    def test_search_performance(self, api_client):
        """Test search response time"""
        query_data = {"text": "performance test query"}
        
        start_time = time.time()
        response = api_client.post(f"{BASE_URL}/troubleshoot", json=query_data)
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should respond within 1 second for most queries
        assert response_time < 1000, f"Search took {response_time:.2f}ms, expected < 1000ms"
    
    def test_concurrent_searches(self, api_client):
        """Test multiple concurrent searches"""
        def search_query(query_text):
            query_data = {"text": f"concurrent test {query_text}"}
            response = api_client.post(f"{BASE_URL}/troubleshoot", json=query_data)
            return response.status_code == 200
        
        # Run 5 concurrent searches
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search_query, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All searches should succeed
        assert all(results), "Some concurrent searches failed"

def run_integration_tests():
    """Run integration tests with proper setup"""
    print("🧪 Running Integration Tests for Simplified Knowledge Base")
    print("=" * 60)
    
    # Check if service is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print("❌ Service not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Service not running. Please start with: python run.py")
        return False
    
    # Run tests
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    
    if exit_code == 0:
        print("\n✅ All integration tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1) 