# Deployment Guide: ADR-MedDRA Platform

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended)
**Best for**: Quick deployment, prototyping, small-scale usage

### Option 2: Docker Container
**Best for**: Production environments, scalability, enterprise deployment

### Option 3: Local Development
**Best for**: Development, testing, offline usage

---

## 🌐 Streamlit Cloud Deployment

### Prerequisites
- GitHub repository with the project
- Streamlit Cloud account
- Python 3.8+ compatibility

### Deployment Steps

1. **Prepare Repository Structure**
```bash
# Ensure product/ directory contains all necessary files
ADR-MedDRA/
├── product/
│   ├── app.py
│   ├── model.py
│   ├── drug_extractor.py
│   ├── attribution.py
│   ├── requirements.txt
│   └── data/
│       ├── meddra_terms.csv
│       └── adr_drug_knowledge.csv
```

2. **Configure Streamlit**
Create `product/.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

3. **Environment Variables**
Set in Streamlit Cloud dashboard:
```
TOKENIZERS_PARALLELISM=false
TF_CPP_MIN_LOG_LEVEL=3
```

4. **Deploy**
- Connect GitHub repository
- Set main file path: `product/app.py`
- Deploy and monitor logs

### Post-Deployment Checklist
- [ ] Test all functionality
- [ ] Verify data loading
- [ ] Check model initialization
- [ ] Validate drug extraction
- [ ] Test error handling

---

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY product/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY product/ .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  adr-meddra:
    build: .
    ports:
      - "8501:8501"
    environment:
      - TOKENIZERS_PARALLELISM=false
      - TF_CPP_MIN_LOG_LEVEL=3
    volumes:
      - ./product/data:/app/data:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Build and Run
```bash
# Build the Docker image
docker build -t adr-meddra .

# Run the container
docker run -p 8501:8501 adr-meddra

# Or use Docker Compose
docker-compose up -d
```

---

## ☁️ Cloud Platform Deployment

### AWS EC2 Deployment
```bash
# Launch EC2 instance (t3.medium recommended)
# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start

# Deploy application
git clone https://github.com/your-repo/ADR-MedDRA.git
cd ADR-MedDRA
docker build -t adr-meddra .
docker run -d -p 80:8501 --name adr-app adr-meddra
```

### Google Cloud Run
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/adr-meddra

# Deploy to Cloud Run
gcloud run deploy --image gcr.io/PROJECT_ID/adr-meddra \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1
```

### Azure Container Instances
```bash
# Create resource group
az group create --name adr-meddra-rg --location eastus

# Deploy container
az container create \
  --resource-group adr-meddra-rg \
  --name adr-meddra-app \
  --image your-registry/adr-meddra:latest \
  --ports 8501 \
  --memory 2 \
  --cpu 1
```

---

## 📊 Production Configuration

### Environment Variables
```bash
# Required
TOKENIZERS_PARALLELISM=false
TF_CPP_MIN_LOG_LEVEL=3

# Optional
MODEL_CACHE_DIR=/app/cache
MAX_CONCURRENT_USERS=50
ENABLE_ANALYTICS=true
LOG_LEVEL=INFO
```

### Resource Requirements

#### Minimum Configuration
- **CPU**: 1 vCPU
- **RAM**: 2 GB
- **Storage**: 1 GB
- **Network**: 1 Mbps

#### Recommended Configuration
- **CPU**: 2 vCPU
- **RAM**: 4 GB
- **Storage**: 5 GB
- **Network**: 10 Mbps

#### High-Traffic Configuration
- **CPU**: 4+ vCPU
- **RAM**: 8+ GB
- **Storage**: 10+ GB
- **Network**: 100+ Mbps
- **Load Balancer**: Required

### Performance Optimization
```python
# In production, add to app.py
import streamlit as st
from streamlit.runtime.caching import cache_data

# Configure caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_model():
    return MedDRAMatcher("data/meddra_terms.csv")

# Enable compression
st.set_page_config(
    page_title="ADR-MedDRA Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

---

## 🔒 Security Configuration

### SSL/TLS Setup
```nginx
# Nginx configuration
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Authentication (Optional)
```python
# Add to app.py for basic auth
import streamlit_authenticator as stauth

# Configure authentication
authenticator = stauth.Authenticate(
    credentials,
    'cookie_name',
    'signature_key',
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')
if authentication_status == True:
    # Main app content
    pass
elif authentication_status == False:
    st.error('Username/password is incorrect')
```

---

## 📈 Monitoring & Analytics

### Health Checks
```python
# Add health check endpoint
import streamlit as st
from streamlit.web.server.server import Server

@st.cache_data
def health_check():
    try:
        # Test model loading
        matcher = load_model()
        return {"status": "healthy", "timestamp": datetime.now()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Logging Configuration
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Analytics Tracking
```python
# Optional: Add Google Analytics
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
""", unsafe_allow_html=True)
```

---

## 🚨 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check data files exist
ls -la product/data/

# Verify file permissions
chmod 644 product/data/*.csv

# Test model loading
python -c "from model import MedDRAMatcher; m = MedDRAMatcher('data/meddra_terms.csv')"
```

#### Memory Issues
```python
# Optimize memory usage
import gc
import torch

# Clear cache periodically
@st.cache_data(max_entries=10)
def cached_prediction(text):
    result = matcher.predict(text)
    gc.collect()  # Force garbage collection
    return result
```

#### Performance Issues
- Increase server resources
- Enable caching
- Optimize model loading
- Use CDN for static assets

---

## 📝 Deployment Checklist

### Pre-Deployment
- [ ] Test all functionality locally
- [ ] Verify data file integrity
- [ ] Check security configurations
- [ ] Validate environment variables
- [ ] Test error handling

### Post-Deployment
- [ ] Verify application loads correctly
- [ ] Test core functionality
- [ ] Monitor resource usage
- [ ] Check logs for errors
- [ ] Set up monitoring alerts
- [ ] Document deployment details

---

*Deployment Guide Version: 1.0*  
*Last Updated: December 17, 2025*  
*Maintained by: DevOps Team*