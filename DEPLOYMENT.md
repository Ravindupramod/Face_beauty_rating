# Deployment Guide

Complete guide for deploying the Beauty Prediction API to production.

## Quick Start (Docker)

The fastest way to get started:

```bash
# 1. Clone repository
git clone <repository-url>
cd beauty-prediction

# 2. Build and run with Docker Compose
docker-compose up -d

# 3. Access the API
open http://localhost
```

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **CPU**: 2+ cores (4+ recommended)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 10GB free space
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows with WSL2

### Software Requirements
- Python 3.10+
- Docker 20.10+ (for containerized deployment)
- Docker Compose 2.0+ (optional)
- Git

## Local Development

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install production dependencies
pip install -r requirements-prod.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
nano .env
```

### 3. Run API Server

```bash
# Development mode (with auto-reload)
python api.py

# Production mode with Uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4

# Production mode with Gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 4. Test API

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/info

# Predict with image
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict

# Open web UI
open http://localhost:8000
```

## Docker Deployment

### Build Docker Image

```bash
# Build image
docker build -t beauty-prediction-api:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  --name beauty-api \
  beauty-prediction-api:latest

# Check logs
docker logs beauty-api

# Stop container
docker stop beauty-api
```

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### Docker Compose Services

- **api**: Beauty Prediction API (port 8000)
- **nginx**: Reverse proxy (ports 80, 443)

## Cloud Deployment

### AWS Deployment

#### Option 1: AWS EC2

```bash
# 1. Launch EC2 instance (Ubuntu 22.04, t3.medium or larger)

# 2. SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# 4. Clone repository
git clone <repository-url>
cd beauty-prediction

# 5. Run with Docker Compose
docker-compose up -d

# 6. Configure security group
# Allow inbound: TCP 80, 443 from 0.0.0.0/0
```

#### Option 2: AWS ECS (Fargate)

```bash
# 1. Build and push image to ECR
aws ecr create-repository --repository-name beauty-prediction
docker tag beauty-prediction-api:latest <account-id>.dkr.ecr.region.amazonaws.com/beauty-prediction:latest
docker push <account-id>.dkr.ecr.region.amazonaws.com/beauty-prediction:latest

# 2. Create ECS task definition and service
# Use AWS Console or CLI
```

### Google Cloud Platform

```bash
# 1. Build and push to Container Registry
gcloud builds submit --tag gcr.io/<project-id>/beauty-prediction

# 2. Deploy to Cloud Run
gcloud run deploy beauty-prediction \
  --image gcr.io/<project-id>/beauty-prediction \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Microsoft Azure

```bash
# 1. Create Container Registry
az acr create --resource-group mygroup --name myregistry --sku Basic

# 2. Build and push image
az acr build --registry myregistry --image beauty-prediction:latest .

# 3. Deploy to Container Instances
az container create \
  --resource-group mygroup \
  --name beauty-api \
  --image myregistry.azurecr.io/beauty-prediction:latest \
  --dns-name-label beauty-api \
  --ports 8000
```

### DigitalOcean

```bash
# 1. Create Droplet (Ubuntu 22.04, 2GB/2CPU or larger)

# 2. Install Docker and Docker Compose

# 3. Deploy with Docker Compose
docker-compose up -d
```

## Configuration

### Environment Variables

Create `.env` file (copy from `.env.example`):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model paths
MODEL_PATH=./models/best_model.pth
ONNX_MODEL_PATH=./models/beauty_model_int8.onnx

# Logging
LOG_LEVEL=info

# CORS (update for production)
CORS_ORIGINS=["https://yourdomain.com"]
```

### Nginx Configuration

For production with SSL:

```nginx
# /etc/nginx/sites-available/beauty-api
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL/HTTPS Setup

#### Using Let's Encrypt (Free)

```bash
# Install Certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Container health
docker ps
```

### Logging

```bash
# View API logs
docker-compose logs -f api

# View Nginx logs
docker-compose logs -f nginx

# View specific container
docker logs -f <container-id>
```

### Performance Monitoring

```bash
# Install monitoring tools
pip install prometheus-client grafana-client

# Monitor CPU/Memory
docker stats

# API metrics (if implemented)
curl http://localhost:8000/metrics
```

## Production Checklist

Before going to production:

- [ ] Model trained and validated
- [ ] ONNX model exported and quantized
- [ ] Environment variables configured
- [ ] CORS origins updated for production domain
- [ ] SSL/HTTPS certificates installed
- [ ] Firewall configured (allow 80, 443)
- [ ] Monitoring and alerts set up
- [ ] Backup strategy implemented
- [ ] Rate limiting configured
- [ ] Load balancing configured (if needed)
- [ ] Documentation reviewed
- [ ] Security audit completed

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>
```

#### Docker Build Fails
```bash
# Clear Docker cache
docker system prune -a

# Rebuild with no cache
docker build --no-cache -t beauty-prediction-api .
```

#### Out of Memory
```bash
# Reduce workers
API_WORKERS=2 uvicorn api:app

# Or increase container memory
docker run -m 4g beauty-prediction-api
```

#### Model Not Found
```bash
# Check model path
ls -la models/

# Update .env
MODEL_PATH=./path/to/model.pth
```

### Performance Issues

#### High Latency
- Reduce image size before upload
- Use ONNX Int8 quantized model
- Enable model caching
- Add Redis for caching predictions
- Use GPU if available

#### Low Throughput
- Increase API workers
- Use load balancer
- Enable HTTP/2
- Optimize Docker image size
- Use CDN for static files

## Scaling

### Horizontal Scaling

```bash
# Docker Compose scaling
docker-compose up -d --scale api=3

# Kubernetes
kubectl scale deployment beauty-api --replicas=3
```

### Load Balancing

Use Nginx or cloud load balancer to distribute traffic across multiple API instances.

## Security

### Best Practices

1. **Use HTTPS**: Always use SSL/TLS in production
2. **API Authentication**: Implement API keys or OAuth
3. **Rate Limiting**: Prevent abuse
4. **Input Validation**: Validate all uploads
5. **Regular Updates**: Keep dependencies updated
6. **Secrets Management**: Use environment variables, not hardcoded secrets
7. **Firewall**: Restrict access to necessary ports only

### Rate Limiting Example

```python
# In api.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(...):
    ...
```

## Support

For issues and questions:
- GitHub Issues: [repository-url/issues]
- Documentation: [docs-url]
- Email: [support-email]

---

**Last Updated**: 2024-12-06
