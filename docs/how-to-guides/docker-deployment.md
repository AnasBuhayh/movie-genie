# Docker Deployment Guide

This guide covers deploying Movie Genie using Docker and docker-compose for production and development environments.

## Overview

Movie Genie uses a multi-container Docker setup with:

- **Backend**: Flask API with Gunicorn (Python 3.11)
- **Frontend**: React SPA served by Nginx
- **MLflow**: Experiment tracking server
- **Docs**: MkDocs documentation server

All services are orchestrated using docker-compose with proper networking, health checks, and volume management.

## Prerequisites

- **Docker**: Engine 20.10+ and Docker Compose 2.0+
- **Python**: 3.9+ (for DVC pipeline)
- **Git**: For cloning repository
- **Resources**: 4GB+ RAM for containers, 5GB+ disk space

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/AnasBuhayh/movie-genie.git
cd movie-genie

# Install Python dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[ml,llm]"
```

### 2. Build Artifacts with DVC

**IMPORTANT**: You must run the DVC pipeline first to generate required artifacts.

```bash
# Run DVC pipeline to generate:
# - data/processed/*.parquet (processed data)
# - models/*.pt (trained models)
# - mlruns/ (MLflow experiments)
# - movie_genie/frontend/dist/ (built frontend)
dvc repro

# First run takes ~15 minutes:
# - Downloads MovieLens data
# - Generates embeddings with sentence-transformers
# - Trains BERT4Rec and Two-Tower models
# - Builds React frontend
```

**Why DVC first?**

Docker containers mount these artifacts as read-only volumes. They cannot generate them at runtime because:
- Model training requires GPU/significant compute (better done locally)
- DVC provides reproducibility and version control for ML artifacts
- Separates data/ML concerns from deployment concerns

### 3. Configure Environment

```bash
# Create environment file
cp .env.example .env

# Edit .env with your configuration
# At minimum, change SECRET_KEY for production!
nano .env
```

### 4. Build and Start Services

```bash
# Build all containers
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 5. Access Services

Once all containers are healthy:

- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:5001/api
- **MLflow UI**: http://localhost:5002
- **Documentation**: http://localhost:8000

### 6. Verify Health

```bash
# Check service status
docker-compose ps

# Check health checks
docker-compose ps | grep "healthy"

# Test backend health endpoint
curl http://localhost:5001/api/health
```

## DVC + Docker Workflow

### Understanding the Separation

**DVC Pipeline** (Local):
- Data processing and feature engineering
- Model training (GPU/CPU intensive)
- Frontend build (Node.js)
- MLflow experiment tracking

**Docker Deployment** (Containerized):
- Serves pre-trained models
- Hosts frontend static files
- Provides API endpoints
- MLflow UI for viewing experiments

### Complete Workflow

```bash
# 1. Development: Build artifacts with DVC
dvc repro

# 2. Production: Deploy with Docker
docker-compose build
docker-compose up -d

# 3. Updates: Retrain models and redeploy
dvc repro bert4rec_training    # Retrain specific model
docker-compose restart backend  # Reload with new model

# 4. Data changes: Rebuild everything
dvc repro                       # Reprocess data and retrain
docker-compose down
docker-compose build
docker-compose up -d
```

## Architecture

### Multi-Stage Builds

Both backend and frontend use multi-stage Docker builds for optimization:

#### Backend Dockerfile

```dockerfile
# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[ml,llm]"

# Stage 2: Runtime - Minimal production image
FROM python:3.11-slim
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN groupadd -r moviegenie && useradd -r -g moviegenie moviegenie
USER moviegenie
CMD ["gunicorn", ...]
```

**Benefits**:
- Smaller final image (no build tools)
- Faster deployment
- More secure (non-root user)

#### Frontend Dockerfile

```dockerfile
# Stage 1: Builder - Build React app
FROM node:20-alpine as builder
WORKDIR /app
RUN npm ci --only=production
RUN npm run build

# Stage 2: Runtime - Nginx server
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
```

**Benefits**:
- No Node.js in production image
- Optimized static file serving
- Nginx performance and security

### Networking

All services run on an isolated bridge network:

```yaml
networks:
  moviegenie-network:
    driver: bridge
    name: moviegenie-network
```

**Internal communication**:
- Backend → MLflow: `http://mlflow:5002`
- Frontend → Backend: `http://backend:5001/api`

**External access**:
- Controlled via port mappings in docker-compose.yml

### Volumes

#### Backend Volumes

```yaml
volumes:
  - ./data:/app/data:ro          # Read-only data
  - ./models:/app/models:ro      # Read-only models
  - ./mlruns:/app/mlruns:ro      # Read-only MLflow experiments
  - ./configs:/app/configs:ro    # Read-only configs
  - backend-logs:/app/logs       # Writable logs
```

**Why read-only?**
- Prevents accidental data modification
- Clearer separation of concerns
- Better security

#### MLflow Volumes

```yaml
volumes:
  - ./mlruns:/mlflow/mlruns:rw   # Read-write for experiments
  - mlflow-data:/mlflow          # Persistent DB and artifacts
```

### Health Checks

All services have health checks for reliable orchestration:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5001/api/health"]
  interval: 30s      # Check every 30 seconds
  timeout: 10s       # Fail if takes >10s
  retries: 3         # Try 3 times before unhealthy
  start_period: 40s  # Wait 40s before first check
```

**Benefits**:
- Automatic restart on failure
- Load balancer integration
- Dependency management

## Configuration

### Environment Variables

Create `.env` file from template:

```bash
cp .env.example .env
```

**Key variables**:

```bash
# Ports
BACKEND_PORT=5001
FRONTEND_PORT=8080
MLFLOW_PORT=5002
DOCS_PORT=8000

# Backend
FLASK_ENV=production
SECRET_KEY=your-secret-key-here

# Frontend
VITE_API_URL=http://localhost:5001/api
```

**Production checklist**:
1. Change `SECRET_KEY` to random string
2. Set `FLASK_ENV=production`
3. Update `VITE_API_URL` if using reverse proxy
4. Configure firewall for exposed ports

### Custom Configuration

#### Change Ports

Edit `.env`:

```bash
BACKEND_PORT=9000
FRONTEND_PORT=3000
```

Then restart:

```bash
docker-compose down
docker-compose up -d
```

#### Add Environment Variable

1. Add to `.env`:
   ```bash
   NEW_VAR=value
   ```

2. Update `docker-compose.yml`:
   ```yaml
   backend:
     environment:
       - NEW_VAR=${NEW_VAR}
   ```

3. Restart service:
   ```bash
   docker-compose restart backend
   ```

## Common Operations

### Start Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d backend

# Start with logs visible
docker-compose up
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes data!)
docker-compose down -v

# Stop specific service
docker-compose stop backend
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart backend
```

### Rebuild Containers

```bash
# Rebuild all
docker-compose build

# Rebuild specific service
docker-compose build backend

# Rebuild and restart
docker-compose up -d --build
```

### Execute Commands in Container

```bash
# Open shell in backend
docker-compose exec backend /bin/bash

# Run Python script
docker-compose exec backend python scripts/test.py

# Check backend logs directory
docker-compose exec backend ls -la logs/
```

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d

# Verify health
docker-compose ps
```

## Development Workflow

### Development Mode

For active development, use local environment instead of Docker:

```bash
# Use the app manager script
./scripts/app.sh dev

# Or manually:
python scripts/start_server.py &
cd movie_genie/frontend && npm run dev
```

**Why?**
- Faster iteration (no rebuild needed)
- Hot reloading for frontend
- Easier debugging
- Direct file access

### Production Testing

Test production build locally with Docker:

```bash
# Build production images
docker-compose build

# Start services
docker-compose up -d

# Run tests
docker-compose exec backend pytest tests/

# Check production bundle
docker-compose exec frontend ls -lh /usr/share/nginx/html
```

### Hybrid Development

Run some services in Docker, others locally:

```bash
# Start only MLflow in Docker
docker-compose up -d mlflow

# Run backend locally (connects to Docker MLflow)
export MLFLOW_TRACKING_URI=http://localhost:5002
python scripts/start_server.py

# Run frontend locally
cd movie_genie/frontend
npm run dev
```

## Production Deployment

### Prerequisites

1. **Server Requirements**:
   - Ubuntu 20.04+ or similar Linux
   - 4GB+ RAM
   - 20GB+ disk space
   - Docker & docker-compose installed

2. **Domain Setup**:
   - Domain name configured
   - SSL certificate obtained (Let's Encrypt)

3. **Security**:
   - Firewall configured
   - SSH key-based auth
   - Non-root user for Docker

### Deployment Steps

#### 1. Server Setup

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install docker-compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### 2. Clone and Configure

```bash
# Clone repository
git clone https://github.com/AnasBuhayh/movie-genie.git
cd movie-genie

# Create production .env
cp .env.example .env
nano .env

# Generate secure secret key
python3 -c "import secrets; print(secrets.token_hex(32))"
# Copy output to SECRET_KEY in .env
```

#### 3. Prepare Data and Models

```bash
# Copy data to server (if needed)
scp -r data/ user@server:/path/to/movie-genie/
scp -r models/ user@server:/path/to/movie-genie/

# Or use DVC to pull from remote storage
dvc pull
```

#### 4. Build and Start

```bash
# Build containers
docker-compose build --no-cache

# Start services
docker-compose up -d

# Check health
docker-compose ps
docker-compose logs -f
```

#### 5. Configure Reverse Proxy (Optional)

Using Nginx as reverse proxy for SSL:

```nginx
# /etc/nginx/sites-available/moviegenie
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Frontend
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/moviegenie /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 6. Set Up Auto-Start

Create systemd service:

```bash
# /etc/systemd/system/moviegenie.service
[Unit]
Description=Movie Genie Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/movie-genie
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable moviegenie
sudo systemctl start moviegenie
```

### Monitoring

#### View Service Status

```bash
# Docker status
docker-compose ps

# System resources
docker stats

# Disk usage
docker system df
```

#### Log Management

```bash
# View logs
docker-compose logs -f --tail=100

# Log rotation (add to crontab)
0 0 * * * docker-compose logs --no-color > /var/log/moviegenie/$(date +\%Y\%m\%d).log
```

#### Health Checks

```bash
# Check all health endpoints
curl http://localhost:5001/api/health
curl http://localhost:8080/health
curl http://localhost:5002/health
curl http://localhost:8000

# Automated monitoring script
#!/bin/bash
for service in backend:5001/api frontend:8080 mlflow:5002 docs:8000; do
    name=${service%%:*}
    url=http://localhost:${service##*:}/health
    if curl -f -s $url > /dev/null; then
        echo "[OK] $name: healthy"
    else
        echo "[FAIL] $name: unhealthy"
    fi
done
```

## Troubleshooting

### Container Won't Start

**Check logs**:
```bash
docker-compose logs backend
```

**Common issues**:

1. **Port already in use**:
   ```bash
   # Find process using port
   lsof -i :5001

   # Change port in .env
   BACKEND_PORT=5002
   ```

2. **Volume mount errors**:
   ```bash
   # Check permissions
   ls -la data/ models/

   # Fix permissions
   chmod -R 755 data/ models/
   ```

3. **Environment variable issues**:
   ```bash
   # Verify .env file exists
   cat .env

   # Check docker-compose sees variables
   docker-compose config
   ```

### Service Unhealthy

**Check health check**:
```bash
# View health check configuration
docker inspect moviegenie-backend | grep -A 10 Healthcheck

# Manual health check
docker-compose exec backend curl -f http://localhost:5001/api/health
```

**Common fixes**:

1. **Backend not responding**:
   ```bash
   # Check if process running
   docker-compose exec backend ps aux

   # Restart service
   docker-compose restart backend
   ```

2. **MLflow database locked**:
   ```bash
   # Stop all services
   docker-compose down

   # Remove lock file
   rm mlruns/mlflow.db.lock

   # Restart
   docker-compose up -d
   ```

### Network Issues

**Check network**:
```bash
# List networks
docker network ls

# Inspect network
docker network inspect moviegenie-network

# Test connectivity
docker-compose exec backend ping mlflow
docker-compose exec frontend ping backend
```

**Fix network issues**:
```bash
# Recreate network
docker-compose down
docker network rm moviegenie-network
docker-compose up -d
```

### Build Failures

**Clear Docker cache**:
```bash
# Remove old images
docker-compose down --rmi all

# Clear build cache
docker builder prune -a

# Rebuild from scratch
docker-compose build --no-cache
```

**Check build logs**:
```bash
# Build with verbose output
docker-compose build --progress=plain backend
```

### Performance Issues

**Check resource usage**:
```bash
# Real-time stats
docker stats

# Specific container
docker stats moviegenie-backend
```

**Optimize**:

1. **Limit container resources**:
   ```yaml
   # docker-compose.yml
   backend:
     deploy:
       resources:
         limits:
           cpus: '2'
           memory: 2G
         reservations:
           memory: 1G
   ```

2. **Reduce log verbosity**:
   ```yaml
   backend:
     environment:
       - LOG_LEVEL=WARNING
   ```

3. **Enable Docker buildkit**:
   ```bash
   export DOCKER_BUILDKIT=1
   docker-compose build
   ```

### Data Persistence Issues

**Check volumes**:
```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect moviegenie-backend-logs

# Check volume mount
docker-compose exec backend df -h
```

**Backup volumes**:
```bash
# Backup MLflow data
docker run --rm -v moviegenie-mlflow-data:/data -v $(pwd):/backup \
    alpine tar czf /backup/mlflow-backup.tar.gz /data

# Restore
docker run --rm -v moviegenie-mlflow-data:/data -v $(pwd):/backup \
    alpine tar xzf /backup/mlflow-backup.tar.gz -C /
```

## Security Best Practices

### 1. Non-Root Users

All containers run as non-root users:

```dockerfile
RUN groupadd -r moviegenie && useradd -r -g moviegenie moviegenie
USER moviegenie
```

### 2. Read-Only Volumes

Mount sensitive data as read-only:

```yaml
volumes:
  - ./data:/app/data:ro
  - ./models:/app/models:ro
```

### 3. Secret Management

**Never commit secrets!**

```bash
# Use .env file (in .gitignore)
SECRET_KEY=generated-secret-key

# Or use Docker secrets (Swarm mode)
echo "secret-value" | docker secret create flask_secret -
```

### 4. Network Isolation

```yaml
networks:
  moviegenie-network:
    driver: bridge
    internal: true  # No external access
```

### 5. Security Headers

Configured in nginx.conf:

```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
```

### 6. Regular Updates

```bash
# Update base images
docker-compose pull

# Rebuild with latest base images
docker-compose build --pull

# Update dependencies
docker-compose exec backend pip list --outdated
```

## Advanced Topics

### Multi-Stage Production

Separate staging and production:

```bash
# docker-compose.staging.yml
version: '3.8'
services:
  backend:
    environment:
      - FLASK_ENV=staging
```

Deploy:

```bash
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
```

### Load Balancing

Use multiple backend replicas:

```yaml
backend:
  deploy:
    replicas: 3
    update_config:
      parallelism: 1
      delay: 10s
```

### CI/CD Integration

GitHub Actions example:

```yaml
# .github/workflows/docker.yml
name: Docker Build and Push
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build images
        run: docker-compose build
      - name: Push to registry
        run: docker-compose push
```

### Docker Swarm

Deploy across multiple hosts:

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml moviegenie

# Scale service
docker service scale moviegenie_backend=5
```

## Backup and Recovery

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup volumes
docker run --rm -v moviegenie-mlflow-data:/data -v $BACKUP_DIR:/backup \
    alpine tar czf /backup/mlflow.tar.gz /data

docker run --rm -v moviegenie-backend-logs:/data -v $BACKUP_DIR:/backup \
    alpine tar czf /backup/logs.tar.gz /data

# Backup database
docker-compose exec -T backend python scripts/backup_db.py > $BACKUP_DIR/db.sql

echo "Backup completed: $BACKUP_DIR"
```

### Restore

```bash
#!/bin/bash
# restore.sh

BACKUP_DIR="/backups/20241005"

# Stop services
docker-compose down

# Restore volumes
docker run --rm -v moviegenie-mlflow-data:/data -v $BACKUP_DIR:/backup \
    alpine tar xzf /backup/mlflow.tar.gz -C /

# Start services
docker-compose up -d
```

## Performance Optimization

### Build Optimization

1. **Use .dockerignore**: Already configured
2. **Layer caching**: Order Dockerfile commands by change frequency
3. **Multi-stage builds**: Reduce final image size
4. **BuildKit**: Enable for better caching

```bash
export DOCKER_BUILDKIT=1
docker-compose build
```

### Runtime Optimization

1. **Resource limits**: Set appropriate CPU/memory limits
2. **Logging**: Use log rotation
3. **Networking**: Use Docker's DNS caching
4. **Volumes**: Use named volumes instead of bind mounts for better performance

## Conclusion

This Docker setup provides:

- **Production-ready**: Security, health checks, logging
- **Scalable**: Easy to add replicas and load balancing
- **Maintainable**: Clear separation of concerns
- **Documented**: Comprehensive guides and examples

For questions or issues, refer to:
- [Project Documentation](http://localhost:8000)
- [GitHub Issues](https://github.com/AnasBuhayh/movie-genie/issues)
- [Docker Documentation](https://docs.docker.com)
