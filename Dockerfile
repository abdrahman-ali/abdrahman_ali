# Multi-stage Dockerfile for ML/DL/RL Pipeline

FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/interim data/external \
    models/saved_models models/checkpoints \
    logs \
    docs/reports/figures

# Expose port for API
EXPOSE 8000

# Development stage
FROM base as development
RUN pip install --no-cache-dir jupyter ipython pytest black flake8
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production
CMD ["python", "-m", "src.deployment.api"]

# Training stage
FROM base as training
CMD ["python", "main.py", "--mode", "train"]

# API serving stage
FROM base as api
CMD ["uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
