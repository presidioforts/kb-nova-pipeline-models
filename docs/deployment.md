# Deployment Guide

## Overview

This guide covers deploying the Knowledge Base Nova Pipeline Models service in production environments. The service can be deployed using various methods depending on your infrastructure requirements.

## Prerequisites

### System Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 10GB+ for models and data
- **Python**: 3.8 or higher
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, or macOS

### Optional Requirements

- **GPU**: NVIDIA GPU with CUDA support for faster training
- **Docker**: For containerized deployment
- **Load Balancer**: For high-availability setups

## Deployment Methods

### 1. Direct Python Deployment

#### Step 1: Environment Setup

```bash
# Create application user
sudo useradd -m -s /bin/bash kb-service
sudo su - kb-service

# Clone repository
git clone <repository-url>
cd kb-nova-pipeline-models

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Configuration

Create environment configuration:

```bash
# Create .env file
cat > .env << EOF
# Server Configuration
HOST=0.0.0.0
PORT=8080
WORKERS=4

# Model Configuration
MODEL_DIR=/opt/kb-service/models
BASE_MODEL=all-mpnet-base-v2

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/kb-service/app.log

# Environment
ENVIRONMENT=production
HF_HUB_DISABLE_PROGRESS_BARS=1
HF_HUB_OFFLINE=1
EOF
```

#### Step 3: Create Systemd Service

```bash
sudo tee /etc/systemd/system/kb-service.service << EOF
[Unit]
Description=Knowledge Base Service
After=network.target

[Service]
Type=exec
User=kb-service
Group=kb-service
WorkingDirectory=/home/kb-service/kb-nova-pipeline-models
Environment=PATH=/home/kb-service/kb-nova-pipeline-models/venv/bin
ExecStart=/home/kb-service/kb-nova-pipeline-models/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8080 --workers 4
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable kb-service
sudo systemctl start kb-service
```

#### Step 4: Verify Deployment

```bash
# Check service status
sudo systemctl status kb-service

# Test API
curl http://localhost:8080/docs

# Check logs
sudo journalctl -u kb-service -f
```

### 2. Docker Deployment

#### Step 1: Create Dockerfile

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/docs || exit 1

# Start application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Step 2: Build and Run

```bash
# Build image
docker build -t kb-service:latest .

# Run container
docker run -d \
  --name kb-service \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  kb-service:latest
```

#### Step 3: Docker Compose (Recommended)

```yaml
# docker-compose.yml
version: '3.8'

services:
  kb-service:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
      - HF_HUB_DISABLE_PROGRESS_BARS=1
      - HF_HUB_OFFLINE=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/docs"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - kb-service
    restart: unless-stopped
```

### 3. Kubernetes Deployment

#### Step 1: Create Deployment Manifest

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kb-service
  labels:
    app: kb-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kb-service
  template:
    metadata:
      labels:
        app: kb-service
    spec:
      containers:
      - name: kb-service
        image: kb-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: HF_HUB_DISABLE_PROGRESS_BARS
          value: "1"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /docs
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /docs
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: kb-service-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: kb-service
spec:
  selector:
    app: kb-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

#### Step 2: Deploy to Kubernetes

```bash
# Apply deployment
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -l app=kb-service
kubectl get services kb-service

# View logs
kubectl logs -l app=kb-service -f
```

## Production Configuration

### 1. Environment Variables

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8080
WORKERS=4

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com,api.your-domain.com

# Database (if implemented)
DATABASE_URL=postgresql://user:pass@localhost/kb_service

# Model Configuration
MODEL_DIR=/opt/models
BASE_MODEL=all-mpnet-base-v2
MAX_TRAINING_PAIRS=1000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/kb-service/app.log

# Monitoring
METRICS_ENABLED=true
HEALTH_CHECK_INTERVAL=30

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Environment
ENVIRONMENT=production
DEBUG=false
```

### 2. Nginx Configuration

```nginx
# nginx.conf
upstream kb_service {
    server 127.0.0.1:8080;
    # Add more servers for load balancing
    # server 127.0.0.1:8081;
    # server 127.0.0.1:8082;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    client_max_body_size 10M;
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;

    location / {
        proxy_pass http://kb_service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        access_log off;
        proxy_pass http://kb_service/docs;
    }
}
```

### 3. Monitoring and Logging

#### Prometheus Metrics (Future Enhancement)

```python
# Add to requirements.txt
# prometheus-client==0.17.1

# Example metrics endpoint
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

#### Log Aggregation

```yaml
# docker-compose.yml addition for ELK stack
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Security Considerations

### 1. Authentication and Authorization

```python
# Future enhancement: Add JWT authentication
from fastapi.security import HTTPBearer
from jose import JWTError, jwt

security = HTTPBearer()

@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    # Implement authentication logic
    pass
```

### 2. Input Validation

```python
# Enhanced validation in schemas.py
from pydantic import validator, Field

class Query(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    
    @validator('text')
    def validate_text(cls, v):
        # Add custom validation logic
        return v.strip()
```

### 3. Rate Limiting

```python
# Add to requirements.txt
# slowapi==0.1.9

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/troubleshoot")
@limiter.limit("10/minute")
def troubleshoot(request: Request, q: Query):
    # Existing logic
    pass
```

## Backup and Recovery

### 1. Model Backup

```bash
#!/bin/bash
# backup-models.sh

BACKUP_DIR="/backup/models"
MODEL_DIR="/opt/models"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
tar -czf "$BACKUP_DIR/models_backup_$DATE.tar.gz" -C "$MODEL_DIR" .

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "models_backup_*.tar.gz" -mtime +7 -delete
```

### 2. Database Backup (if implemented)

```bash
#!/bin/bash
# backup-db.sh

pg_dump kb_service > "/backup/db/kb_service_$(date +%Y%m%d_%H%M%S).sql"
```

## Scaling Considerations

### 1. Horizontal Scaling

- Use load balancer (Nginx, HAProxy, or cloud LB)
- Deploy multiple service instances
- Implement session affinity for training jobs
- Use shared storage for models

### 2. Vertical Scaling

- Increase CPU/RAM for faster inference
- Use GPU for accelerated training
- Optimize batch sizes and model parameters

### 3. Database Scaling (Future)

- Implement read replicas
- Use connection pooling
- Consider sharding for large datasets

## Troubleshooting

### Common Issues

1. **Service won't start**
   - Check port availability: `netstat -tlnp | grep 8080`
   - Verify Python dependencies: `pip check`
   - Check logs: `journalctl -u kb-service`

2. **High memory usage**
   - Monitor model loading: Check if multiple models are loaded
   - Adjust batch sizes in training
   - Consider model quantization

3. **Slow responses**
   - Check CPU/GPU utilization
   - Monitor network latency
   - Optimize model inference

### Health Checks

```bash
# Basic health check
curl -f http://localhost:8080/docs

# Detailed service check
curl -X POST http://localhost:8080/troubleshoot \
  -H "Content-Type: application/json" \
  -d '{"text": "test query"}'
```

## Maintenance

### Regular Tasks

1. **Log Rotation**
   ```bash
   # Add to /etc/logrotate.d/kb-service
   /var/log/kb-service/*.log {
       daily
       rotate 30
       compress
       delaycompress
       missingok
       notifempty
       postrotate
           systemctl reload kb-service
       endscript
   }
   ```

2. **Model Cleanup**
   ```bash
   # Clean old model versions (keep last 5)
   find /opt/models/fine-tuned-runs -type d -name "fine-tuned-*" | sort -r | tail -n +6 | xargs rm -rf
   ```

3. **Security Updates**
   ```bash
   # Update dependencies
   pip install --upgrade -r requirements.txt
   
   # Rebuild Docker image
   docker build -t kb-service:latest .
   ``` 