# CustomerIQ Deployment Guide

## Deployment Options

CustomerIQ can be deployed using multiple methods depending on your needs and infrastructure requirements.

## 🚀 Quick Deployment Options

### 1. Streamlit Cloud (Recommended for Demo/Development)

**Pros**: Free, easy setup, automatic deployments
**Cons**: Limited resources, public repositories only

**Steps**:
1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository and branch
5. Set main file path: `src/app.py`
6. Deploy!

**Configuration**:
```toml
# .streamlit/config.toml
[server]
port = 8501
headless = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### 2. Docker Deployment

**Pros**: Consistent environment, scalable, works anywhere
**Cons**: Requires Docker knowledge

**Quick Start**:
```bash
# Build and run with Docker
docker build -t customeriq .
docker run -p 8501:8501 customeriq

# Or use Docker Compose
docker-compose up -d
```

**Production Configuration**:
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  customeriq:
    image: customeriq:latest
    ports:
      - "80:8501"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/customeriq
    volumes:
      - ./data:/app/data
    restart: always
  
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: customeriq
      POSTGRES_USER: customeriq_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

### 3. Heroku Deployment

**Pros**: Easy scaling, managed infrastructure
**Cons**: Costs money, limited free tier

**Setup Files**:

`Procfile`:
```
web: streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0
```

`runtime.txt`:
```
python-3.9.7
```

**Deploy Commands**:
```bash
# Install Heroku CLI
heroku login
heroku create your-app-name
heroku addons:create heroku-postgresql:hobby-dev
git push heroku main
heroku open
```

### 4. AWS Deployment

#### Option A: AWS ECS with Fargate

**Benefits**: Serverless containers, auto-scaling

**Setup**:
1. Push Docker image to ECR
2. Create ECS task definition
3. Create ECS service
4. Configure Application Load Balancer

**Sample Task Definition**:
```json
{
  "family": "customeriq",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "customeriq",
      "image": "your-account.dkr.ecr.region.amazonaws.com/customeriq:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://..."
        }
      ]
    }
  ]
}
```

#### Option B: AWS EC2

**Instance Setup**:
```bash
# Ubuntu 20.04 LTS
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker ubuntu

# Clone repository
git clone https://github.com/yourusername/customeriq.git
cd customeriq

# Run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

### 5. Google Cloud Platform

#### Option A: Cloud Run

**Benefits**: Serverless, pay-per-use, auto-scaling

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/customeriq
gcloud run deploy --image gcr.io/PROJECT-ID/customeriq --platform managed
```

#### Option B: Google Kubernetes Engine (GKE)

**Kubernetes Configuration**:
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: customeriq
spec:
  replicas: 3
  selector:
    matchLabels:
      app: customeriq
  template:
    metadata:
      labels:
        app: customeriq
    spec:
      containers:
      - name: customeriq
        image: gcr.io/PROJECT-ID/customeriq:latest
        ports:
        - containerPort: 8501
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: customeriq-secrets
              key: database-url
---
apiVersion: v1
kind: Service
metadata:
  name: customeriq-service
spec:
  selector:
    app: customeriq
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

### 6. Microsoft Azure

#### Option A: Azure Container Instances

```bash
# Create resource group
az group create --name customeriq-rg --location eastus

# Deploy container
az container create \
  --resource-group customeriq-rg \
  --name customeriq \
  --image your-registry/customeriq:latest \
  --ports 8501 \
  --dns-name-label customeriq-app \
  --environment-variables DATABASE_URL=postgresql://...
```

#### Option B: Azure App Service

```bash
# Create App Service plan
az appservice plan create --name customeriq-plan --resource-group customeriq-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group customeriq-rg --plan customeriq-plan --name customeriq-app --deployment-container-image-name your-registry/customeriq:latest
```

## 🔧 Environment Configuration

### Environment Variables

Create `.env` file for local development:
```bash
# Database
DATABASE_URL=sqlite:///data/customeriq.db
# DATABASE_URL=postgresql://user:password@localhost:5432/customeriq

# Application
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
PYTHONPATH=/app/src

# Security
SECRET_KEY=your-secret-key-change-me
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

# External APIs (if used)
# API_KEY=your-api-key
```

### Production Environment Variables

```bash
# Database (Production)
DATABASE_URL=postgresql://user:password@prod-db-host:5432/customeriq

# Application
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Security
SECRET_KEY=complex-secret-key-for-production
ALLOWED_HOSTS=your-production-domain.com

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance
WORKERS=2
MAX_CONNECTIONS=100
```

## 📊 Monitoring and Logging

### Health Checks

Add health check endpoint:
```python
# src/health.py
from datetime import datetime
import psutil
import streamlit as st

def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent(),
        "version": "1.0.0"
    }
```

### Logging Configuration

```python
# config/logging.py
import logging
import sys
from datetime import datetime

def setup_logging(log_level="INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/customeriq_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )
```

### Monitoring with Prometheus

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('customeriq_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('customeriq_request_duration_seconds', 'Request duration')

def metrics_endpoint():
    return generate_latest()
```

## 🔒 Security Considerations

### 1. Environment Security

```bash
# Use secrets management
export DATABASE_URL=$(aws secretsmanager get-secret-value --secret-id prod/customeriq/db --query SecretString --output text)
```

### 2. Network Security

```yaml
# docker-compose with network isolation
networks:
  customeriq-network:
    driver: bridge
    internal: true
  web-network:
    driver: bridge

services:
  customeriq:
    networks:
      - customeriq-network
      - web-network
```

### 3. SSL/TLS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://customeriq:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Port conflicts**:
   ```bash
   # Check what's using port 8501
   lsof -i :8501
   # Use different port
   streamlit run src/app.py --server.port 8502
   ```

2. **Memory issues**:
   ```bash
   # Monitor memory usage
   docker stats customeriq
   # Increase container memory
   docker run -m 2g customeriq
   ```

3. **Database connection issues**:
   ```python
   # Test database connection
   python -c "from src.database import DatabaseManager; db = DatabaseManager(); db.test_connection()"
   ```

### Performance Optimization

1. **Enable caching**:
   ```python
   @st.cache_data(ttl=3600)
   def load_data():
       # Expensive data loading
       pass
   ```

2. **Use connection pooling**:
   ```python
   # sqlalchemy connection pool
   engine = create_engine(
       DATABASE_URL,
       pool_size=20,
       max_overflow=30,
       pool_recycle=3600
   )
   ```

3. **Resource limits**:
   ```yaml
   # docker-compose resource limits
   deploy:
     resources:
       limits:
         cpus: '1.0'
         memory: 2G
       reservations:
         cpus: '0.5'
         memory: 1G
   ```

## 📈 Scaling Strategies

### Horizontal Scaling

```yaml
# docker-compose scale
version: '3.8'
services:
  customeriq:
    image: customeriq:latest
    deploy:
      replicas: 3
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    depends_on:
      - customeriq
```

### Load Balancing

```nginx
# nginx load balancer
upstream customeriq_backend {
    server customeriq_1:8501;
    server customeriq_2:8501;
    server customeriq_3:8501;
}

server {
    listen 80;
    location / {
        proxy_pass http://customeriq_backend;
    }
}
```

### Auto-scaling (Kubernetes)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: customeriq-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: customeriq
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 📋 Deployment Checklist

### Pre-deployment

- [ ] Code review completed
- [ ] Tests passing (unit, integration)
- [ ] Security scan completed
- [ ] Performance testing done
- [ ] Documentation updated
- [ ] Environment variables configured
- [ ] Database migrations ready
- [ ] Backup strategy in place

### Deployment

- [ ] Deploy to staging first
- [ ] Verify staging functionality
- [ ] Deploy to production
- [ ] Verify production deployment
- [ ] Monitor application metrics
- [ ] Check error logs
- [ ] Test critical user flows
- [ ] Notify stakeholders

### Post-deployment

- [ ] Monitor performance metrics
- [ ] Check error rates
- [ ] Verify data integrity
- [ ] Update monitoring dashboards
- [ ] Document any issues
- [ ] Plan next iteration

Choose the deployment method that best fits your requirements, budget, and technical expertise. For quick demos, use Streamlit Cloud. For production applications, consider Docker with cloud providers like AWS, GCP, or Azure.
