# API Documentation

## Overview

The Knowledge Base Nova Pipeline Models API provides three main endpoints for semantic search and model training. All endpoints accept and return JSON data.

## Base URL

```
http://localhost:8080
```

## Authentication

Currently, no authentication is required. This should be implemented for production use.

## Endpoints

### 1. Troubleshoot Query

Find the most relevant solution for a given problem description using semantic similarity.

**Endpoint:** `POST /troubleshoot`

**Content-Type:** `application/json`

#### Request

```json
{
  "text": "string"
}
```

**Parameters:**
- `text` (string, required): The problem description or query text

#### Response

```json
{
  "query": "string",
  "response": "string", 
  "similarity_score": "float"
}
```

**Fields:**
- `query`: The original query text
- `response`: The recommended solution
- `similarity_score`: Cosine similarity score (0.0 to 1.0)

#### Examples

**Example 1: NPM Issue**
```bash
curl -X POST "http://localhost:8080/troubleshoot" \
  -H "Content-Type: application/json" \
  -d '{"text": "npm install is hanging and not completing"}'
```

Response:
```json
{
  "query": "npm install is hanging and not completing",
  "response": "Clear npm cache (npm cache clean --force) or check network.",
  "similarity_score": 0.87
}
```

**Example 2: Python Import Error**
```bash
curl -X POST "http://localhost:8080/troubleshoot" \
  -H "Content-Type: application/json" \
  -d '{"text": "ModuleNotFoundError: No module named requests"}'
```

Response:
```json
{
  "query": "ModuleNotFoundError: No module named requests",
  "response": "Check if module is installed (pip list) and verify PYTHONPATH.",
  "similarity_score": 0.92
}
```

#### Error Responses

**400 Bad Request**
```json
{
  "detail": "Request body is required"
}
```

**500 Internal Server Error**
```json
{
  "detail": "internal error"
}
```

### 2. Submit Training Data

Submit new training pairs to fine-tune the model with domain-specific knowledge.

**Endpoint:** `POST /train`

**Content-Type:** `application/json`

#### Request

```json
{
  "data": [
    {
      "input": "string",
      "target": "string"
    }
  ]
}
```

**Parameters:**
- `data` (array, required): List of training pairs
  - `input` (string, required): Problem description or query
  - `target` (string, required): Expected solution or response

#### Response

```json
{
  "job_id": "string",
  "note": "string"
}
```

**Fields:**
- `job_id`: Unique identifier for tracking the training job
- `note`: Confirmation message with number of pairs accepted

#### Examples

**Example 1: Single Training Pair**
```bash
curl -X POST "http://localhost:8080/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "input": "Docker container exits immediately",
        "target": "Check the container logs with docker logs <container_id> and verify the entry point command"
      }
    ]
  }'
```

Response:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "note": "1 pairs accepted"
}
```

**Example 2: Multiple Training Pairs**
```bash
curl -X POST "http://localhost:8080/train" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "input": "Git merge conflict in package.json",
        "target": "Manually edit package.json to resolve conflicts, then git add package.json && git commit"
      },
      {
        "input": "Database connection timeout",
        "target": "Check database server status, network connectivity, and connection string parameters"
      },
      {
        "input": "SSL certificate verification failed",
        "target": "Update certificates or temporarily disable SSL verification for testing"
      }
    ]
  }'
```

Response:
```json
{
  "job_id": "456e7890-e89b-12d3-a456-426614174001",
  "note": "3 pairs accepted"
}
```

#### Error Responses

**400 Bad Request**
```json
{
  "detail": "No training data received"
}
```

### 3. Check Training Status

Monitor the progress of a training job using the job ID returned from the training submission.

**Endpoint:** `GET /train/{job_id}`

#### Request

**Path Parameters:**
- `job_id` (string, required): The job ID returned from training submission

#### Response

```json
{
  "status": "string",
  "msg": "string"
}
```

**Fields:**
- `status`: Current job status
- `msg`: Status message or error details

**Status Values:**
- `queued`: Job is waiting to start
- `running`: Training is in progress
- `finished`: Training completed successfully
- `failed`: Training failed (check `msg` for error details)

#### Examples

**Example 1: Successful Training**
```bash
curl -X GET "http://localhost:8080/train/123e4567-e89b-12d3-a456-426614174000"
```

Response:
```json
{
  "status": "finished",
  "msg": "saved to breakfix-kb-model/all-mpnet-base-v2/fine-tuned-runs/fine-tuned-20231201-143022"
}
```

**Example 2: Training in Progress**
```bash
curl -X GET "http://localhost:8080/train/456e7890-e89b-12d3-a456-426614174001"
```

Response:
```json
{
  "status": "running",
  "msg": ""
}
```

**Example 3: Failed Training**
```bash
curl -X GET "http://localhost:8080/train/789e0123-e89b-12d3-a456-426614174002"
```

Response:
```json
{
  "status": "failed",
  "msg": "Training failed: CUDA out of memory"
}
```

#### Error Responses

**404 Not Found**
```json
{
  "detail": "job id not found"
}
```

## Usage Patterns

### 1. Basic Troubleshooting Workflow

```python
import requests

# Query for a solution
response = requests.post(
    "http://localhost:8080/troubleshoot",
    json={"text": "Python script crashes with segmentation fault"}
)

result = response.json()
print(f"Solution: {result['response']}")
print(f"Confidence: {result['similarity_score']:.2f}")
```

### 2. Training and Monitoring Workflow

```python
import requests
import time

# Submit training data
training_data = {
    "data": [
        {
            "input": "React component not re-rendering",
            "target": "Check if state is being mutated directly. Use setState or useState hook properly."
        },
        {
            "input": "API request returns 401 unauthorized",
            "target": "Verify authentication token is valid and included in request headers."
        }
    ]
}

response = requests.post(
    "http://localhost:8080/train",
    json=training_data
)

job_id = response.json()["job_id"]
print(f"Training job submitted: {job_id}")

# Monitor training progress
while True:
    status_response = requests.get(f"http://localhost:8080/train/{job_id}")
    status = status_response.json()["status"]
    
    print(f"Training status: {status}")
    
    if status in ["finished", "failed"]:
        print(f"Final message: {status_response.json()['msg']}")
        break
    
    time.sleep(5)  # Check every 5 seconds
```

### 3. Batch Processing

```python
import requests

# Process multiple queries
queries = [
    "Docker build fails with permission denied",
    "MySQL connection refused",
    "JavaScript undefined variable error"
]

solutions = []
for query in queries:
    response = requests.post(
        "http://localhost:8080/troubleshoot",
        json={"text": query}
    )
    solutions.append(response.json())

# Display results
for solution in solutions:
    print(f"Q: {solution['query']}")
    print(f"A: {solution['response']}")
    print(f"Score: {solution['similarity_score']:.2f}")
    print("-" * 50)
```

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing:
- Request rate limits per IP
- Concurrent training job limits
- Model inference request queuing

## Error Handling

The API uses standard HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (invalid job ID)
- `500`: Internal Server Error

All error responses include a `detail` field with a description of the error.

## OpenAPI/Swagger Documentation

When the service is running, interactive API documentation is available at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc` 