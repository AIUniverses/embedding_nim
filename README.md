# 🚀 Enhanced Embedding Service

A production-ready, NVIDIA NIM-compatible embedding service with advanced features including caching, monitoring, vector database integration, and multi-modal support.

## ✨ Features

### Core Capabilities
- **OpenAI API Compatible**: Drop-in replacement for OpenAI embeddings API
- **NVIDIA NIM Compliant**: Full compatibility with NVIDIA NIM specification
- **Multi-Model Support**: E5, GTE, SentenceTransformers, NVIDIA Embed models
- **Multi-Modal**: Text and image embeddings (model dependent)

### 🔥 Advanced Features
- **Intelligent Caching**: Redis, Memory, and Disk cache backends
- **Dynamic Batching**: Adaptive batch processing with priority queues
- **Real-time Monitoring**: Prometheus metrics + Grafana dashboards
- **Vector Database**: Built-in Qdrant integration for storage
- **Streaming API**: WebSocket support for real-time processing
- **Security**: Authentication, rate limiting, input validation

### 🏗️ Deployment Modes

#### Simple Mode (Lightweight)
- Single embedding service container
- Memory-based caching
- Perfect for development and testing

#### Full Mode (Production)
- Embedding service with all features
- Redis for advanced caching
- Prometheus + Grafana monitoring
- Qdrant vector database
- Full observability stack

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with CUDA support (for GPU acceleration)
- 8GB+ GPU memory (recommended)

### Required Secrets
`docker-compose` refuses to start until these are set (see `.env.example`):

```bash
cp .env.example .env
# EMBEDDING_API_KEY    - API key clients must send as `Authorization: Bearer ...`
# REDIS_PASSWORD       - Redis auth (full mode)
# GRAFANA_ADMIN_PASSWORD - Grafana admin login (full mode)
```

For local development without authentication, set `ALLOW_UNAUTHENTICATED=true`
instead of `EMBEDDING_API_KEY`.

### Simple Deployment
```bash
# Make scripts executable
chmod +x launch_service.sh

# Start in simple mode
./launch_service.sh -m simple -a up

# Or using docker-compose directly
docker-compose -f docker-compose.simple.yml up -d
```

### Full Deployment
```bash
# Start full stack with rebuild
./launch_service.sh -m full -a up -b

# Or using docker-compose directly
docker-compose up -d
```

### Service Management
```bash
# Check status
./launch_service.sh -a status

# View logs
./launch_service.sh -a logs

# Restart with rebuild
./launch_service.sh -a restart -b

# Stop services
./launch_service.sh -a down
```

## 📡 API Usage

### Basic Embedding Generation
```python
import requests

# Single text embedding
response = requests.post("http://localhost:8009/v1/embeddings", json={
    "model": "e5-large-v2",
    "input": "Hello world",
    "input_type": "query"
})

# Batch processing
response = requests.post("http://localhost:8009/v1/embeddings", json={
    "model": "e5-large-v2", 
    "input": ["Query text", "Document text"],
    "input_type": "query"
})
```

### Advanced Features
```python
# With dimensions and embedding type
response = requests.post("http://localhost:8009/v1/embeddings", json={
    "model": "e5-large-v2",
    "input": "Sample text",
    "dimensions": 512,
    "embedding_type": "float",
    "normalize": True
})

# Batch processing endpoint
response = requests.post("http://localhost:8009/v1/embeddings/batch", json=[
    {"model": "e5-large-v2", "input": "Text 1"},
    {"model": "e5-base-v2", "input": "Text 2"}
])
```

### WebSocket Streaming
```python
import asyncio
import websockets
import json

async def stream_embeddings():
    uri = "ws://localhost:8009/v1/embeddings/stream"
    async with websockets.connect(uri) as websocket:
        # Send request
        await websocket.send(json.dumps({
            "model": "e5-large-v2",
            "input": "Streaming text",
            "input_type": "query"
        }))
        
        # Receive response
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(stream_embeddings())
```

## 🔧 Configuration

### Environment Variables
```bash
# Deployment mode
DEPLOYMENT_MODE=full                    # simple or full
ENABLE_REDIS=true                      # Enable Redis caching
ENABLE_MONITORING=true                 # Enable Prometheus/Grafana
ENABLE_VECTOR_DB=true                  # Enable Qdrant

# Caching configuration
CACHE_BACKEND=redis                    # memory, redis, disk
CACHE_TTL=3600                         # Cache TTL in seconds
CACHE_MAX_SIZE=1000                    # Max cache entries

# Batching configuration
ENABLE_BATCHING=true                   # Enable dynamic batching
MAX_BATCH_SIZE=32                      # Maximum batch size
BATCH_TIMEOUT_MS=10                    # Batch timeout in milliseconds
ADAPTIVE_BATCHING=true                 # Enable adaptive sizing

# GPU configuration
CUDA_VISIBLE_DEVICES=0                 # GPU device selection

# Strict compatibility (hide metadata fields)
STRICT_NIM_MODE=false

# Truncation strategy for overlength inputs (none|head|tail|mid)
TRUNCATE=none

# Enable deterministic placeholder image embeddings (for development only)
ENABLE_FAKE_IMAGE_EMBEDDINGS=false

# API security
EMBEDDING_API_KEY=your-secret-key   # required (>=16 chars) unless ALLOW_UNAUTHENTICATED=true
RATE_LIMIT_PER_MINUTE=100
CORS_ORIGINS=                       # comma-separated allow-list; empty blocks cross-origin
ALLOWED_HOSTS=*                     # comma-separated Host header allow-list
ENABLE_DOCS=false                   # serve /docs, /redoc and /openapi.json
```

### Model Configuration
Edit `config/models.yaml` to add or modify models:

```yaml
models:
  custom-model:
    model_id: "your/custom-model"
    display_name: "Custom Model"
    family: "sentence_transformers"
    device: "cuda"
    embedding_dimension: 768
    max_seq_length: 512
    normalize_embeddings: true
    supports_input_type: false
    supported_embedding_types: ["float", "int8"]
    supported_modalities: ["text"]
    settings:
      batch_size: 64
      max_memory_gb: 2
```

## 📊 Monitoring & Observability

### Access Monitoring (Full Mode)
- **Grafana Dashboard**: http://localhost:3000 (admin / `$GRAFANA_ADMIN_PASSWORD`)
- **Prometheus Metrics**: http://localhost:9090
- **Service Health**: http://localhost:8009/v1/health/ready
- **API Documentation**: http://localhost:8009/docs

### Key Metrics
- Request throughput and latency
- Cache hit rates and performance
- GPU memory usage and utilization
- Model loading and inference times
- Error rates and types

### Custom Metrics Endpoint
```bash
curl http://localhost:8009/v1/embeddings/metrics
```

## 🗄️ Vector Database Integration

### Qdrant Dashboard
Access the Qdrant web interface at http://localhost:6333/dashboard

### Python Client Example
```python
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# Store embeddings
client.upsert(
    collection_name="my_collection",
    points=[
        {
            "id": 1,
            "vector": embedding_vector,
            "payload": {"text": "Sample text"}
        }
    ]
)

# Search similar vectors
results = client.search(
    collection_name="my_collection",
    query_vector=query_vector,
    limit=5
)
```

## 🔒 Security Features

### API Key Authentication (Required)

The service refuses to start unless `EMBEDDING_API_KEY` is set (minimum 16
characters), or `ALLOW_UNAUTHENTICATED=true` is set explicitly for local
development. Only `/`, `/health`, `/health/live` and `/v1/health/*` are public;
`/metrics`, `/docs` and every `/v1` endpoint - including the
`/v1/embeddings/stream` WebSocket - require the key.

```bash
# Set API key
export EMBEDDING_API_KEY="$(openssl rand -hex 32)"

# Use in requests
curl -H "Authorization: Bearer your-secret-key" \
     -H "Content-Type: application/json" \
     -d '{"model": "e5-large-v2", "input": "Hello"}' \
     http://localhost:8009/v1/embeddings
```

### Rate Limiting
```python
# Rate limiting is automatically applied
# Default: 100 requests per minute per IP
# Configure via RATE_LIMIT_PER_MINUTE environment variable
```

## 🧪 Testing

### Run Test Suite
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest test_embedding_service.py -v

# Run specific test categories
pytest test_embedding_service.py::TestEmbeddingService::test_health_endpoints -v
```

### Manual Testing
```bash
# Health check
curl http://localhost:8009/v1/health/ready

# List available models
curl http://localhost:8009/v1/models

# Test embedding generation
curl -X POST http://localhost:8009/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"model": "e5-large-v2", "input": "Hello world"}'
```

## 🚀 Performance Optimization

### GPU Configuration
```bash
# Multi-GPU setup
CUDA_VISIBLE_DEVICES=0,1,2,3

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Caching Strategies
- **Memory Cache**: Fastest, limited by RAM
- **Redis Cache**: Distributed, persistent across restarts
- **Disk Cache**: Large capacity, slower access

### Batch Processing
- Enable dynamic batching for higher throughput
- Adjust batch sizes based on GPU memory
- Use priority queues for SLA-critical requests
 - Cross-request micro-batching merges compatible single-text calls within batch_timeout window

### Strict Mode
Set `STRICT_NIM_MODE=true` để trả về phản hồi tối giản giống NIM/OpenAI (ẩn metadata phụ trợ, cache flag).

## 📚 API Reference

### Endpoints
- `POST /v1/embeddings` - Generate embeddings
- `POST /v1/embeddings/batch` - Batch processing
- `WS /v1/embeddings/stream` - Streaming WebSocket
- `GET /v1/models` - List available models
- `GET /v1/health/ready` - Health check
- `GET /v1/embeddings/metrics` - Service metrics

### Model Suffixes
- `model-name-query` - Optimized for queries
- `model-name-passage` - Optimized for documents

### Response Format
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],
      "index": 0
    }
  ],
  "model": "e5-large-v2",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: See `/docs` folder
- **Examples**: See `/examples` folder

---

**Built with ❤️ for the AI community**