#!/usr/bin/env python3
"""
Comprehensive Test Suite for Hybrid Knowledge Base
Generates synthetic data and tests all system components
"""

import asyncio
import json
import time
import random
import string
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    name: str
    query: str
    expected_category: str = None
    expected_min_results: int = 1
    expected_max_time_ms: float = 1000.0
    description: str = ""

class SyntheticDataGenerator:
    """Generate diverse synthetic test data"""
    
    def __init__(self):
        self.categories = [
            "javascript", "python", "docker", "kubernetes", "aws", "database", 
            "networking", "security", "performance", "deployment", "testing",
            "frontend", "backend", "mobile", "devops", "monitoring"
        ]
        
        # Technology-specific issue patterns
        self.issue_templates = {
            "javascript": [
                "npm ERR! peer dep missing: {package}",
                "TypeError: Cannot read property '{prop}' of undefined",
                "Module '{module}' not found",
                "Unexpected token '{token}' in JSON",
                "Promise rejection unhandled: {error}",
                "React hook '{hook}' is called conditionally",
                "ESLint error: '{rule}' rule violation",
                "Webpack build failed: {reason}",
                "Node.js memory leak in {component}",
                "Express middleware {middleware} not working"
            ],
            "python": [
                "ModuleNotFoundError: No module named '{module}'",
                "IndentationError: expected an indented block",
                "KeyError: '{key}' not found in dictionary",
                "AttributeError: '{object}' has no attribute '{attr}'",
                "ImportError: cannot import name '{name}'",
                "SyntaxError: invalid syntax in {file}",
                "ValueError: {function} received invalid argument",
                "TypeError: {function}() missing required argument",
                "FileNotFoundError: {file} does not exist",
                "ConnectionError: Failed to connect to {service}"
            ],
            "docker": [
                "Docker build failed: layer {layer} error",
                "Container {name} keeps restarting",
                "Port {port} already in use",
                "Volume mount permission denied",
                "Image {image} not found",
                "Docker daemon not running",
                "Out of disk space during build",
                "Network {network} connection failed",
                "Docker-compose service {service} unhealthy",
                "Registry authentication failed"
            ],
            "kubernetes": [
                "Pod {pod} stuck in Pending state",
                "Service {service} endpoint not found",
                "Ingress {ingress} returning 404",
                "PVC {pvc} in Pending state",
                "Node {node} not ready",
                "ImagePullBackOff for {image}",
                "CrashLoopBackOff in {container}",
                "ConfigMap {configmap} not found",
                "Secret {secret} permission denied",
                "HPA {hpa} not scaling"
            ],
            "aws": [
                "S3 bucket {bucket} access denied",
                "EC2 instance {instance} connection timeout",
                "Lambda function {function} timeout",
                "RDS connection pool exhausted",
                "CloudFormation stack {stack} rollback",
                "ELB health check failing",
                "IAM role {role} insufficient permissions",
                "API Gateway {api} throttling",
                "SQS queue {queue} message delay",
                "EKS cluster {cluster} upgrade failed"
            ],
            "database": [
                "Connection pool exhausted for {database}",
                "Table {table} deadlock detected",
                "Index {index} optimization needed",
                "Query timeout on {query}",
                "Foreign key constraint violation",
                "Disk space full on {server}",
                "Replication lag on {replica}",
                "Transaction isolation conflict",
                "Column {column} data type mismatch",
                "Backup restoration failed"
            ]
        }
        
        # Resolution templates
        self.resolution_templates = [
            "Update {component} to version {version}",
            "Clear cache and restart {service}",
            "Check {config_file} configuration",
            "Increase {resource} allocation to {amount}",
            "Install missing dependency: {dependency}",
            "Fix permissions on {path}",
            "Restart {service} with --{flag} option",
            "Use {alternative} instead of {current}",
            "Add {parameter} to {config}",
            "Migrate from {old} to {new} approach"
        ]
        
        # Common tech terms for substitution
        self.tech_terms = {
            "package": ["react", "express", "lodash", "axios", "moment"],
            "module": ["auth", "database", "utils", "config", "logger"],
            "service": ["nginx", "redis", "postgresql", "elasticsearch", "rabbitmq"],
            "component": ["API", "frontend", "backend", "middleware", "cache"],
            "version": ["2.1.4", "3.0.0", "1.8.2", "4.2.1", "0.9.5"],
            "dependency": ["python3", "nodejs", "gcc", "make", "cmake"],
            "config_file": ["nginx.conf", "app.yaml", "docker-compose.yml", "package.json"]
        }

    def generate_issue(self, category: str) -> str:
        """Generate a realistic issue description"""
        if category not in self.issue_templates:
            category = random.choice(list(self.issue_templates.keys()))
        
        template = random.choice(self.issue_templates[category])
        
        # Substitute placeholders with realistic values
        for placeholder, options in self.tech_terms.items():
            if f"{{{placeholder}}}" in template:
                template = template.replace(f"{{{placeholder}}}", random.choice(options))
        
        # Handle remaining placeholders with generic terms
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        for placeholder in placeholders:
            if placeholder not in self.tech_terms:
                template = template.replace(f"{{{placeholder}}}", f"example_{placeholder}")
        
        return template

    def generate_resolution(self, issue: str, category: str) -> str:
        """Generate a resolution based on the issue"""
        template = random.choice(self.resolution_templates)
        
        # Context-aware substitutions
        substitutions = {
            "component": category,
            "service": f"{category}_service",
            "resource": random.choice(["memory", "CPU", "disk", "network"]),
            "amount": random.choice(["2GB", "4 cores", "100GB", "1Gbps"]),
            "dependency": random.choice(self.tech_terms["dependency"]),
            "path": f"/var/{category}",
            "flag": random.choice(["verbose", "debug", "force", "quiet"]),
            "config": f"{category}.conf",
            "parameter": f"max_{random.choice(['connections', 'memory', 'timeout'])}",
            "alternative": f"new_{category}_lib",
            "current": f"old_{category}_lib",
            "old": f"{category}_v1",
            "new": f"{category}_v2"
        }
        
        for key, value in substitutions.items():
            template = template.replace(f"{{{key}}}", value)
        
        return template

    def generate_tags(self, category: str, issue: str) -> List[str]:
        """Generate relevant tags"""
        base_tags = [category]
        
        # Add technology-specific tags
        tech_tags = {
            "javascript": ["node", "npm", "webpack", "babel"],
            "python": ["pip", "virtualenv", "django", "flask"],
            "docker": ["container", "image", "volume", "network"],
            "kubernetes": ["pod", "service", "deployment", "configmap"],
            "aws": ["cloud", "ec2", "s3", "lambda"],
            "database": ["sql", "nosql", "migration", "backup"]
        }
        
        if category in tech_tags:
            base_tags.extend(random.sample(tech_tags[category], random.randint(1, 3)))
        
        # Add severity/type tags
        severity_tags = ["critical", "warning", "info", "debug"]
        type_tags = ["configuration", "performance", "security", "compatibility"]
        
        base_tags.append(random.choice(severity_tags))
        base_tags.append(random.choice(type_tags))
        
        return list(set(base_tags))  # Remove duplicates

    def generate_synthetic_items(self, count: int) -> List[Dict[str, Any]]:
        """Generate a list of synthetic knowledge base items"""
        items = []
        
        for i in range(count):
            category = random.choice(self.categories)
            issue = self.generate_issue(category)
            resolution = self.generate_resolution(issue, category)
            tags = self.generate_tags(category, issue)
            
            item = {
                "issue": issue,
                "resolution": resolution,
                "category": category,
                "tags": tags
            }
            items.append(item)
        
        return items

class PerformanceTester:
    """Test performance and routing behavior"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    async def test_search_performance(self, queries: List[str], iterations: int = 3) -> Dict[str, Any]:
        """Test search performance across multiple queries"""
        results = {
            "total_queries": len(queries) * iterations,
            "avg_response_time": 0.0,
            "min_response_time": float('inf'),
            "max_response_time": 0.0,
            "success_rate": 0.0,
            "routing_stats": {"hot_hits": 0, "warm_hits": 0, "cold_hits": 0},
            "query_results": []
        }
        
        successful_queries = 0
        total_time = 0.0
        
        for iteration in range(iterations):
            logger.info(f"Performance test iteration {iteration + 1}/{iterations}")
            
            for i, query in enumerate(queries):
                start_time = time.time()
                
                try:
                    response = self.session.post(
                        f"{self.base_url}/api/v1/troubleshoot",
                        json={"text": query},
                        timeout=10
                    )
                    
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        successful_queries += 1
                        data = response.json()
                        
                        results["query_results"].append({
                            "query": query,
                            "response_time_ms": response_time,
                            "results_count": len(data.get("results", [])),
                            "routing_info": data.get("routing_info", {}),
                            "iteration": iteration
                        })
                        
                        total_time += response_time
                        results["min_response_time"] = min(results["min_response_time"], response_time)
                        results["max_response_time"] = max(results["max_response_time"], response_time)
                    else:
                        logger.warning(f"Query failed with status {response.status_code}: {query}")
                        
                except Exception as e:
                    logger.error(f"Query error: {e} for query: {query}")
        
        if successful_queries > 0:
            results["avg_response_time"] = total_time / successful_queries
            results["success_rate"] = successful_queries / results["total_queries"]
        
        return results
    
    async def test_concurrent_queries(self, query: str, concurrent_count: int = 10) -> Dict[str, Any]:
        """Test concurrent query handling"""
        import concurrent.futures
        
        start_time = time.time()
        
        def make_request():
            try:
                response = self.session.post(
                    f"{self.base_url}/api/v1/troubleshoot",
                    json={"text": query},
                    timeout=10
                )
                return {
                    "status_code": response.status_code,
                    "response_time": time.time(),
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {"status_code": 0, "error": str(e), "success": False}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_count) as executor:
            futures = [executor.submit(make_request) for _ in range(concurrent_count)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r["success"])
        
        return {
            "concurrent_requests": concurrent_count,
            "successful_requests": successful,
            "total_time_seconds": total_time,
            "requests_per_second": concurrent_count / total_time,
            "success_rate": successful / concurrent_count
        }

class ComprehensiveTestSuite:
    """Main test suite coordinator"""
    
    def __init__(self):
        self.generator = SyntheticDataGenerator()
        self.tester = PerformanceTester()
        self.test_results = {}
        
    def create_diverse_test_queries(self) -> List[TestCase]:
        """Create a comprehensive set of test queries"""
        test_cases = [
            # Basic functionality tests
            TestCase("npm_error", "npm install error", "javascript", 1, 100),
            TestCase("python_import", "cannot import module", "python", 1, 100),
            TestCase("docker_build", "docker build failed", "docker", 1, 100),
            
            # Category-specific tests
            TestCase("kubernetes_pod", "pod not starting kubernetes", "kubernetes", 1, 200),
            TestCase("aws_s3", "s3 bucket access denied", "aws", 1, 200),
            TestCase("database_connection", "database connection timeout", "database", 1, 200),
            
            # Complex multi-word queries
            TestCase("complex_error", "react component not rendering after state update", "javascript", 1, 300),
            TestCase("performance_issue", "slow database query optimization needed", "database", 1, 300),
            TestCase("deployment_problem", "kubernetes deployment rollback after failed update", "kubernetes", 1, 300),
            
            # Edge cases
            TestCase("short_query", "error", None, 0, 500),
            TestCase("very_specific", "ModuleNotFoundError: No module named 'specific_package_v2'", "python", 1, 200),
            TestCase("mixed_tech", "docker container running python flask app connection refused", None, 1, 400),
            
            # Natural language queries
            TestCase("natural_1", "How do I fix a broken npm installation?", "javascript", 1, 300),
            TestCase("natural_2", "My server keeps crashing, what should I check?", None, 1, 500),
            TestCase("natural_3", "Database is running slow, need optimization tips", "database", 1, 300),
            
            # Performance testing queries
            TestCase("frequent_1", "npm ERR code ERESOLVE", "javascript", 1, 50),  # Should hit hot cache
            TestCase("frequent_2", "docker container not starting", "docker", 1, 100),  # Should hit warm cache
            TestCase("rare_query", "very specific uncommon technical issue", None, 0, 1000),  # Cold storage
        ]
        
        return test_cases
    
    async def run_basic_functionality_tests(self) -> Dict[str, Any]:
        """Test basic search functionality"""
        logger.info("Running basic functionality tests...")
        
        test_cases = self.create_diverse_test_queries()
        results = {
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_details": []
        }
        
        for test_case in test_cases:
            try:
                response = self.tester.session.post(
                    f"{self.tester.base_url}/api/v1/troubleshoot",
                    json={"text": test_case.query},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    search_time = data.get("search_time_ms", 0)
                    results_count = len(data.get("results", []))
                    
                    # Validate results
                    passed = True
                    issues = []
                    
                    if results_count < test_case.expected_min_results:
                        passed = False
                        issues.append(f"Expected min {test_case.expected_min_results} results, got {results_count}")
                    
                    if search_time > test_case.expected_max_time_ms:
                        passed = False
                        issues.append(f"Expected max {test_case.expected_max_time_ms}ms, took {search_time}ms")
                    
                    if passed:
                        results["passed"] += 1
                    else:
                        results["failed"] += 1
                    
                    results["test_details"].append({
                        "name": test_case.name,
                        "query": test_case.query,
                        "passed": passed,
                        "issues": issues,
                        "search_time_ms": search_time,
                        "results_count": results_count,
                        "routing_info": data.get("routing_info", {})
                    })
                    
                else:
                    results["failed"] += 1
                    results["test_details"].append({
                        "name": test_case.name,
                        "query": test_case.query,
                        "passed": False,
                        "issues": [f"HTTP {response.status_code}"],
                        "search_time_ms": None,
                        "results_count": 0
                    })
                    
            except Exception as e:
                results["failed"] += 1
                results["test_details"].append({
                    "name": test_case.name,
                    "query": test_case.query,
                    "passed": False,
                    "issues": [f"Exception: {str(e)}"],
                    "search_time_ms": None,
                    "results_count": 0
                })
        
        return results
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance tests"""
        logger.info("Running performance tests...")
        
        # Create test queries
        queries = [tc.query for tc in self.create_diverse_test_queries()[:10]]
        
        # Basic performance test
        perf_results = await self.tester.test_search_performance(queries, iterations=3)
        
        # Concurrent request test
        concurrent_results = await self.tester.test_concurrent_queries(
            "npm install error", concurrent_count=10
        )
        
        return {
            "sequential_performance": perf_results,
            "concurrent_performance": concurrent_results
        }
    
    async def run_synthetic_data_tests(self, item_count: int = 50) -> Dict[str, Any]:
        """Test with synthetic data"""
        logger.info(f"Generating and testing with {item_count} synthetic items...")
        
        # Generate synthetic items
        synthetic_items = self.generator.generate_synthetic_items(item_count)
        
        # Test adding items (if endpoint exists)
        added_items = 0
        for item in synthetic_items[:10]:  # Add first 10 as test
            try:
                response = self.tester.session.post(
                    f"{self.tester.base_url}/api/v1/knowledge",
                    json=item,
                    timeout=10
                )
                if response.status_code in [200, 201]:
                    added_items += 1
            except Exception as e:
                logger.warning(f"Failed to add item: {e}")
        
        # Test queries against categories in synthetic data
        category_queries = [f"{cat} error" for cat in self.generator.categories[:5]]
        category_results = await self.tester.test_search_performance(category_queries, iterations=1)
        
        return {
            "synthetic_items_generated": item_count,
            "items_added_successfully": added_items,
            "category_search_results": category_results,
            "sample_items": synthetic_items[:3]  # Show first 3 as examples
        }
    
    async def run_health_monitoring(self) -> Dict[str, Any]:
        """Test health and monitoring endpoints"""
        logger.info("Testing health and monitoring...")
        
        try:
            # Health check
            health_response = self.tester.session.get(
                f"{self.tester.base_url}/api/v1/health",
                timeout=10
            )
            
            health_data = health_response.json() if health_response.status_code == 200 else {}
            
            return {
                "health_status": health_response.status_code,
                "health_data": health_data,
                "components_healthy": len([
                    c for c in health_data.get("components", {}).values() 
                    if c.get("status") == "healthy"
                ]) if health_data else 0
            }
            
        except Exception as e:
            return {
                "health_status": "error",
                "error": str(e)
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("🧪 Starting Comprehensive Test Suite...")
        start_time = time.time()
        
        results = {
            "test_suite_version": "1.0",
            "start_time": datetime.now().isoformat(),
            "tests": {}
        }
        
        try:
            # Basic functionality
            results["tests"]["basic_functionality"] = await self.run_basic_functionality_tests()
            
            # Performance tests
            results["tests"]["performance"] = await self.run_performance_tests()
            
            # Synthetic data tests
            results["tests"]["synthetic_data"] = await self.run_synthetic_data_tests(100)
            
            # Health monitoring
            results["tests"]["health_monitoring"] = await self.run_health_monitoring()
            
            # Summary
            total_time = time.time() - start_time
            results["summary"] = {
                "total_test_time_seconds": total_time,
                "basic_tests_passed": results["tests"]["basic_functionality"]["passed"],
                "basic_tests_failed": results["tests"]["basic_functionality"]["failed"],
                "avg_response_time_ms": results["tests"]["performance"]["sequential_performance"]["avg_response_time"],
                "system_health": results["tests"]["health_monitoring"]["health_status"],
                "overall_status": "PASS" if results["tests"]["basic_functionality"]["failed"] == 0 else "FAIL"
            }
            
            logger.info(f"✅ Test suite completed in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            results["error"] = str(e)
        
        return results

async def main():
    """Main test execution"""
    test_suite = ComprehensiveTestSuite()
    
    # Run all tests
    results = await test_suite.run_all_tests()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📊 Test results saved to: {results_file}")
    
    # Print summary
    summary = results.get("summary", {})
    print("\n" + "="*60)
    print("🧪 COMPREHENSIVE TEST SUITE RESULTS")
    print("="*60)
    print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
    print(f"Total Test Time: {summary.get('total_test_time_seconds', 0):.2f}s")
    print(f"Basic Tests Passed: {summary.get('basic_tests_passed', 0)}")
    print(f"Basic Tests Failed: {summary.get('basic_tests_failed', 0)}")
    print(f"Avg Response Time: {summary.get('avg_response_time_ms', 0):.2f}ms")
    print(f"System Health: {summary.get('system_health', 'UNKNOWN')}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    asyncio.run(main()) 