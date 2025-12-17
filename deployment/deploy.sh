#!/bin/bash

# ADR-MedDRA Deployment Script
# Usage: ./deploy.sh [environment] [version]

set -e  # Exit on any error

# Configuration
DOCKER_IMAGE="adr-meddra"
CONTAINER_NAME="adr-meddra-app"
PORT="8501"
ENV=${1:-"development"}
VERSION=${2:-"latest"}

echo "🚀 Starting ADR-MedDRA deployment..."
echo "Environment: $ENV"
echo "Version: $VERSION"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Validate project structure
validate_structure() {
    log_info "Validating project structure..."
    
    required_files=(
        "../product/app.py"
        "../product/model.py"
        "../product/requirements.txt"
        "../product/data/meddra_terms.csv"
        "../product/data/adr_drug_knowledge.csv"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    log_info "Project structure validation passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    docker build -t ${DOCKER_IMAGE}:${VERSION} -f Dockerfile ..
    docker tag ${DOCKER_IMAGE}:${VERSION} ${DOCKER_IMAGE}:latest
    
    log_info "Docker image built successfully"
}

# Stop existing container
stop_existing() {
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        log_warn "Stopping existing container..."
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
    fi
}

# Deploy application
deploy_app() {
    log_info "Deploying application..."
    
    case $ENV in
        "production")
            deploy_production
            ;;
        "staging")
            deploy_staging
            ;;
        "development")
            deploy_development
            ;;
        *)
            log_error "Unknown environment: $ENV"
            exit 1
            ;;
    esac
}

# Production deployment
deploy_production() {
    log_info "Deploying to production..."
    
    docker run -d \
        --name ${CONTAINER_NAME} \
        --restart unless-stopped \
        -p ${PORT}:8501 \
        -e TOKENIZERS_PARALLELISM=false \
        -e TF_CPP_MIN_LOG_LEVEL=3 \
        -e LOG_LEVEL=INFO \
        -v $(pwd)/../product/data:/app/data:ro \
        -v adr-logs:/app/logs \
        --health-cmd="curl --fail http://localhost:8501/_stcore/health || exit 1" \
        --health-interval=30s \
        --health-timeout=10s \
        --health-retries=3 \
        --health-start-period=60s \
        ${DOCKER_IMAGE}:${VERSION}
}

# Staging deployment
deploy_staging() {
    log_info "Deploying to staging..."
    
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8501 \
        -e TOKENIZERS_PARALLELISM=false \
        -e TF_CPP_MIN_LOG_LEVEL=3 \
        -e LOG_LEVEL=DEBUG \
        -v $(pwd)/../product/data:/app/data:ro \
        ${DOCKER_IMAGE}:${VERSION}
}

# Development deployment
deploy_development() {
    log_info "Deploying to development..."
    
    docker-compose -f docker-compose.yml up -d adr-meddra
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    max_attempts=10
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:${PORT}/_stcore/health &> /dev/null; then
            log_info "Health check passed"
            return 0
        fi
        
        log_warn "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Show deployment info
show_info() {
    log_info "Deployment completed successfully!"
    echo ""
    echo "Application URL: http://localhost:${PORT}"
    echo "Container Name: ${CONTAINER_NAME}"
    echo "Environment: ${ENV}"
    echo "Version: ${VERSION}"
    echo ""
    echo "Useful commands:"
    echo "  View logs:    docker logs -f ${CONTAINER_NAME}"
    echo "  Stop app:     docker stop ${CONTAINER_NAME}"
    echo "  Remove app:   docker rm ${CONTAINER_NAME}"
    echo "  Shell access: docker exec -it ${CONTAINER_NAME} /bin/bash"
}

# Rollback function
rollback() {
    log_warn "Rolling back deployment..."
    
    if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
        docker stop ${CONTAINER_NAME} || true
        docker rm ${CONTAINER_NAME} || true
    fi
    
    # Restore from backup if available
    if [ "$(docker images -q ${DOCKER_IMAGE}:backup)" ]; then
        docker tag ${DOCKER_IMAGE}:backup ${DOCKER_IMAGE}:latest
        deploy_app
        log_info "Rollback completed"
    else
        log_error "No backup image found for rollback"
    fi
}

# Main execution
main() {
    # Trap errors and rollback
    trap 'log_error "Deployment failed! Rolling back..."; rollback; exit 1' ERR
    
    check_prerequisites
    validate_structure
    
    # Create backup of current deployment
    if [ "$(docker images -q ${DOCKER_IMAGE}:latest)" ]; then
        docker tag ${DOCKER_IMAGE}:latest ${DOCKER_IMAGE}:backup
    fi
    
    build_image
    stop_existing
    deploy_app
    
    # Wait for container to start
    sleep 5
    
    if health_check; then
        show_info
    else
        log_error "Deployment health check failed"
        rollback
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    "rollback")
        rollback
        exit 0
        ;;
    "logs")
        docker logs -f ${CONTAINER_NAME}
        exit 0
        ;;
    "stop")
        docker stop ${CONTAINER_NAME}
        exit 0
        ;;
    "status")
        if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
            echo "Container is running"
            docker ps --filter name=${CONTAINER_NAME}
        else
            echo "Container is not running"
        fi
        exit 0
        ;;
    *)
        main
        ;;
esac