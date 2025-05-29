# Hybrid Knowledge Base Deployment Guide

## 🚀 Production Deployment Guide

This guide covers deploying the production-ready hybrid knowledge base system with intelligent routing between in-memory and ChromaDB storage.

## 📋 Prerequisites

### System Requirements
```bash
# Minimum Production Requirements
CPU: 4 cores (8 recommended)
RAM: 8GB (16GB recommended)
Storage: 50GB SSD (100GB+ for large datasets)
Network: 1Gbps (for distributed deployments)

# Software Requirements
Python: 3.9+
Docker: 20.10+
Docker Compose: 2.0+
```

### Dependencies Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify ChromaDB installation
python -c "import chromadb; print('ChromaDB installed successfully')"

# Verify SentenceTransformers
python -c "from sentence_transformers import SentenceTransformer; print('SentenceTransformers ready')"
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (nginx)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                FastAPI Application                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Intelligent     │  │   Hot Memory    │  │ Warm Cache   │ │
│  │ Router          │  │   (1-5ms)       │  │ (10-15ms)    │ │
│  │                 │  │                 │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                    ChromaDB                                 │
│              (Cold Storage 25-50ms)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Vector DB     │  │   Metadata      │  │  Persistence │ │
│  │   (DuckDB)      │  │   Storage       │  │   Layer      │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🐳 Docker Deployment

### 1. Create Docker Configuration

**Dockerfile**
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY breakfix-kb-model/ ./breakfix-kb-model/

# Create directories for ChromaDB
RUN mkdir -p /app/chromadb_data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml**
```yaml
version: '3.8'

services:
  hybrid-kb:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
      - CHROMADB_PATH=/app/chromadb_data
      - HOT_CACHE_SIZE=1000
      - WARM_CACHE_SIZE=5000
    volumes:
      - ./chromadb_data:/app/chromadb_data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - hybrid-kb
    restart: unless-stopped

volumes:
  chromadb_data:
  logs:
```

### 2. Nginx Configuration

**nginx.conf**
```nginx
events {
    worker_connections 1024;
}

http {
    upstream hybrid_kb {
        server hybrid-kb:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name your-domain.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # API routes with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://hybrid_kb;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts for different endpoints
            proxy_read_timeout 60s;
            proxy_connect_timeout 10s;
            proxy_send_timeout 10s;
        }

        # Health check (no rate limiting)
        location /api/v1/health {
            proxy_pass http://hybrid_kb;
            proxy_set_header Host $host;
        }

        # Documentation
        location /docs {
            proxy_pass http://hybrid_kb;
            proxy_set_header Host $host;
        }

        # Root
        location / {
            proxy_pass http://hybrid_kb;
            proxy_set_header Host $host;
        }
    }
}
```

### 3. Deploy with Docker Compose

```bash
# Build and start services
docker-compose up -d --build

# Check service status
docker-compose ps

# View logs
docker-compose logs -f hybrid-kb

# Check health
curl http://localhost:8000/api/v1/health
```

## ☸️ Kubernetes Deployment

### 1. Kubernetes Manifests

**namespace.yaml**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: hybrid-kb
```

**configmap.yaml**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hybrid-kb-config
  namespace: hybrid-kb
data:
  LOG_LEVEL: "INFO"
  HOT_CACHE_SIZE: "1000"
  WARM_CACHE_SIZE: "5000"
  CHROMADB_PATH: "/app/chromadb_data"
```

**deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hybrid-kb
  namespace: hybrid-kb
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hybrid-kb
  template:
    metadata:
      labels:
        app: hybrid-kb
    spec:
      containers:
      - name: hybrid-kb
        image: your-registry/hybrid-kb:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: hybrid-kb-config
        volumeMounts:
        - name: chromadb-storage
          mountPath: /app/chromadb_data
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: chromadb-storage
        persistentVolumeClaim:
          claimName: chromadb-pvc
```

**service.yaml**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: hybrid-kb-service
  namespace: hybrid-kb
spec:
  selector:
    app: hybrid-kb
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

**ingress.yaml**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hybrid-kb-ingress
  namespace: hybrid-kb
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "10"
    nginx.ingress.kubernetes.io/rate-limit-window: "1s"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: hybrid-kb-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hybrid-kb-service
            port:
              number: 80
```

### 2. Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods -n hybrid-kb
kubectl get services -n hybrid-kb
kubectl get ingress -n hybrid-kb

# View logs
kubectl logs -f deployment/hybrid-kb -n hybrid-kb
```

## 🔧 Configuration Management

### Environment Variables

```bash
# Core Configuration
PYTHONPATH=/app
LOG_LEVEL=INFO
PORT=8000

# Hybrid System Configuration
HOT_CACHE_SIZE=1000          # Number of items in hot cache
WARM_CACHE_SIZE=5000         # Number of items in warm cache
CHROMADB_PATH=./chromadb_data # ChromaDB storage path

# Performance Tuning
HOT_ACCESS_THRESHOLD=10      # Accesses needed for hot promotion
WARM_ACCESS_THRESHOLD=3      # Accesses needed for warm cache
RECENT_ACCESS_WINDOW=300     # Seconds for "recent" access
QUALITY_THRESHOLD=0.8        # Quality score for promotion

# Model Configuration
MODEL_PATH=./breakfix-kb-model/all-mpnet-base-v2
MODEL_CACHE_DIR=./model_cache

# Security
CORS_ORIGINS=["https://your-domain.com"]
API_KEY_REQUIRED=false       # Set to true for API key auth
```

### Production Configuration File

**config/production.yaml**
```yaml
# Production Configuration for Hybrid Knowledge Base

app:
  name: "Hybrid Knowledge Base Service"
  version: "2.0.0"
  debug: false
  log_level: "INFO"

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 60

hybrid_system:
  hot_cache:
    max_size: 1000
    promotion_threshold: 10
  warm_cache:
    max_size: 5000
    ttl_minutes: 30
    promotion_threshold: 3
  cold_storage:
    chromadb_path: "/app/chromadb_data"
    collection_name: "knowledge_base"
  routing:
    recent_access_window: 300
    quality_threshold: 0.8
    critical_keywords: ["error", "crash", "urgent", "production", "down", "fail"]

model:
  path: "./breakfix-kb-model/all-mpnet-base-v2"
  cache_dir: "./model_cache"
  device: "cpu"  # or "cuda" for GPU

monitoring:
  health_check_interval: 30
  metrics_retention_hours: 24
  performance_logging: true

security:
  cors_origins: ["*"]  # Configure for production
  api_key_required: false
  rate_limiting:
    requests_per_minute: 60
    burst_size: 20
```

## 📊 Monitoring and Observability

### 1. Health Monitoring

```bash
# Basic health check
curl http://localhost:8000/api/v1/health

# Detailed performance metrics
curl http://localhost:8000/api/v1/performance

# System information
curl http://localhost:8000/api/v1/system/info
```

### 2. Prometheus Metrics (Optional)

**prometheus.yml**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hybrid-kb'
    static_configs:
      - targets: ['hybrid-kb:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### 3. Grafana Dashboard

Key metrics to monitor:
- Query response times by tier
- Cache hit rates (hot/warm/cold)
- Total queries processed
- System resource usage
- Error rates and availability

## 🔒 Security Considerations

### 1. Network Security
```bash
# Firewall rules (example for Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 8000/tcp   # Block direct app access
sudo ufw enable
```

### 2. Application Security
- Enable HTTPS with valid certificates
- Implement API rate limiting
- Use environment variables for secrets
- Regular security updates
- Input validation and sanitization

### 3. Data Security
- Encrypt ChromaDB data at rest
- Secure backup procedures
- Access logging and monitoring
- Regular security audits

## 🚀 Performance Optimization

### 1. System Tuning

```bash
# Linux kernel parameters
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' >> /etc/sysctl.conf
echo 'vm.swappiness = 10' >> /etc/sysctl.conf
sysctl -p
```

### 2. Application Tuning

```python
# Uvicorn production settings
uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --timeout-keep-alive 5
```

### 3. ChromaDB Optimization

```python
# ChromaDB settings for production
settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chromadb_data",
    anonymized_telemetry=False,
    allow_reset=False,
    chroma_server_host="localhost",
    chroma_server_http_port=8001
)
```

## 📈 Scaling Strategies

### Horizontal Scaling
1. **Load Balancer**: Distribute requests across multiple instances
2. **Shared ChromaDB**: Use external ChromaDB service
3. **Cache Synchronization**: Implement cache warming strategies

### Vertical Scaling
1. **Memory**: Increase RAM for larger hot cache
2. **CPU**: More cores for concurrent processing
3. **Storage**: SSD for faster ChromaDB operations

### Auto-scaling Configuration

**Kubernetes HPA**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hybrid-kb-hpa
  namespace: hybrid-kb
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hybrid-kb
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🔄 Backup and Recovery

### 1. ChromaDB Backup

```bash
#!/bin/bash
# backup_chromadb.sh

BACKUP_DIR="/backups/chromadb"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SOURCE_DIR="/app/chromadb_data"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Create compressed backup
tar -czf "$BACKUP_DIR/chromadb_backup_$TIMESTAMP.tar.gz" -C "$SOURCE_DIR" .

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "chromadb_backup_*.tar.gz" -mtime +7 -delete

echo "Backup completed: chromadb_backup_$TIMESTAMP.tar.gz"
```

### 2. Model Backup

```bash
#!/bin/bash
# backup_models.sh

BACKUP_DIR="/backups/models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SOURCE_DIR="/app/breakfix-kb-model"

# Create backup
tar -czf "$BACKUP_DIR/models_backup_$TIMESTAMP.tar.gz" -C "$SOURCE_DIR" .

echo "Model backup completed: models_backup_$TIMESTAMP.tar.gz"
```

### 3. Automated Backup with Cron

```bash
# Add to crontab
0 2 * * * /scripts/backup_chromadb.sh
0 3 * * 0 /scripts/backup_models.sh  # Weekly model backup
```

## 🚨 Troubleshooting

### Common Issues

1. **ChromaDB Connection Failed**
   ```bash
   # Check ChromaDB directory permissions
   ls -la /app/chromadb_data
   
   # Verify disk space
   df -h
   
   # Check logs
   docker logs hybrid-kb
   ```

2. **High Memory Usage**
   ```bash
   # Monitor memory usage
   docker stats hybrid-kb
   
   # Reduce hot cache size
   export HOT_CACHE_SIZE=500
   ```

3. **Slow Response Times**
   ```bash
   # Check performance metrics
   curl http://localhost:8000/api/v1/performance
   
   # Monitor cache hit rates
   curl http://localhost:8000/api/v1/health
   ```

### Performance Debugging

```python
# Enable debug logging
import logging
logging.getLogger("src.models.hybrid_knowledge_base").setLevel(logging.DEBUG)

# Monitor query routing decisions
curl -X POST http://localhost:8000/api/v1/troubleshoot \
  -H "Content-Type: application/json" \
  -d '{"text": "npm install error"}' | jq .routing_info
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**:
   - Review performance metrics
   - Check error logs
   - Verify backup integrity

2. **Monthly**:
   - Update dependencies
   - Review security patches
   - Optimize cache configurations

3. **Quarterly**:
   - Performance tuning review
   - Capacity planning
   - Security audit

### Monitoring Checklist

- [ ] Service health status
- [ ] Response time metrics
- [ ] Cache hit rates
- [ ] Error rates
- [ ] Resource utilization
- [ ] Backup status
- [ ] Security alerts

This deployment guide provides a comprehensive foundation for running the hybrid knowledge base system in production with optimal performance, security, and reliability. 