# Enhanced NVIDIA NIM-compatible Embedding Service
# Multi-stage build for production optimization

# Build stage
FROM ubuntu:24.04 AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA 12.4 support for better RTX 50 series support  
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM ubuntu:24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p logs conversations cache models monitoring/data

# Create non-root user for security
RUN useradd -m -u 1002 embedding_user && \
    chown -R embedding_user:embedding_user /app

# Switch to non-root user
USER embedding_user

# Health check - Optimized for faster response
# Use liveness endpoint instead of readiness for Docker health check
# Readiness is slower due to model loading checks
HEALTHCHECK --interval=20s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8009}/health/live || exit 1

# Expose port
EXPOSE 8009

# Default command with enhanced configuration
CMD ["python3", "embedding_service.py", "--host", "0.0.0.0", "--port", "8009"]
