# ================================================================
# SE JUNIOR - DEVOPS: DOCKER, CI/CD, CLOUD
# ================================================================

# --- 1. LÝ THUYẾT (THEORY) ---
"""
1. Docker:
   - Container: Lightweight, isolated environment
   - Image: Template/snapshot của container
   - Dockerfile: Instructions để build image
   - docker-compose: Multi-container applications

2. CI/CD:
   - Continuous Integration: Auto test on every commit
   - Continuous Deployment: Auto deploy when tests pass
   - Tools: GitHub Actions, GitLab CI, Jenkins

3. Cloud Basics:
   - Compute: EC2, Cloud Run, Lambda
   - Storage: S3, Cloud Storage
   - Database: RDS, Cloud SQL
   - Serverless: Functions, containers

4. Infrastructure as Code:
   - Terraform, Pulumi
   - Version control infrastructure
   - Reproducible environments
"""

# --- 2. CODE MẪU (CODE SAMPLE) ---

# ========== DOCKERFILE ==========

DOCKERFILE = '''
# Multi-stage build for smaller image
FROM python:3.11-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY . .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

# ========== DOCKER COMPOSE ==========

DOCKER_COMPOSE = '''
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=mydb
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - app

volumes:
  postgres_data:
  redis_data:
'''

# ========== GITHUB ACTIONS CI/CD ==========

GITHUB_ACTIONS = '''
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest --cov=app tests/
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/test
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Log in to Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v1
        with:
          service: my-app
          image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          region: asia-southeast1
'''

# ========== TERRAFORM BASICS ==========

TERRAFORM = '''
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-southeast-1"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  
  tags = {
    Name = "my-app-vpc"
  }
}

# EC2 Instance
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"
  
  tags = {
    Name = "my-app-server"
  }
}

# RDS Database
resource "aws_db_instance" "main" {
  identifier        = "my-app-db"
  engine            = "postgres"
  engine_version    = "15"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  
  db_name  = "myapp"
  username = "admin"
  password = var.db_password
  
  skip_final_snapshot = true
}

output "instance_ip" {
  value = aws_instance.web.public_ip
}
'''

# --- 3. BÀI TẬP (EXERCISE) ---
"""
BÀI 1: Dockerize your application:
       - Create optimized Dockerfile
       - Use docker-compose for local dev
       - Push to Docker Hub hoặc GitHub Container Registry

BÀI 2: Setup CI/CD pipeline:
       - Run tests on every PR
       - Build and push image on merge to main
       - Deploy to Cloud Run / Vercel / Railway

BÀI 3: Implement health checks:
       - /health endpoint
       - Liveness và Readiness probes
       - Graceful shutdown

BÀI 4: Setup monitoring:
       - Add Sentry for error tracking
       - Configure logging với JSON format
       - Setup basic alerting
"""

if __name__ == "__main__":
    print("=== DevOps Configuration Files ===")
    print("\n--- Dockerfile ---")
    print(DOCKERFILE[:500] + "...")
    print("\n--- docker-compose.yml ---")
    print(DOCKER_COMPOSE[:500] + "...")
