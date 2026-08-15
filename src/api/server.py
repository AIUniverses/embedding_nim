"""FastAPI server application for the embedding service."""

import logging
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
import uvicorn
import time
from collections import defaultdict

from ..config_manager import ConfigManager
from ..model_manager import ModelManager
from ..embedding_manager import EmbeddingManager
from .routes import health, models, embeddings
from .models import ErrorResponse, create_error_response


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    
    # Startup
    logger.info("Starting embedding service...")
    
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        app.state.config_manager = config_manager
        
        # Initialize model manager
        model_manager = ModelManager(config_manager)
        app.state.model_manager = model_manager
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager(config_manager, model_manager)
        app.state.embedding_manager = embedding_manager
        
        # Mark service as ready for basic operations BEFORE model loading
        app.state.service_ready = True
        logger.info("Service ready for basic operations")
        
        # Load default model asynchronously to avoid blocking startup
        default_model = config_manager.get_default_model()
        if default_model:
            logger.info(f"Starting background loading of default model: {default_model}")
            # Create background task for model loading
            async def load_default_model():
                try:
                    success = await model_manager.load_model_async(default_model)
                    if success:
                        logger.info(f"Default model {default_model} loaded successfully")
                        app.state.default_model_loaded = True
                        app.state.default_model_error = None
                    else:
                        reason = model_manager.get_last_load_error() or 'unknown error'
                        logger.error(
                            f"Failed to load default model {default_model}: {reason}"
                        )
                        app.state.default_model_loaded = False
                        app.state.default_model_error = reason
                except Exception as e:
                    logger.error(
                        f"Failed to load default model {default_model}: {str(e)}",
                        exc_info=True
                    )
                    app.state.default_model_loaded = False
                    app.state.default_model_error = str(e)
            
            # Start background model loading
            asyncio.create_task(load_default_model())
        else:
            app.state.default_model_loaded = True
        
        app.state.service_ready = True
        logger.info("Embedding service started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start embedding service: {str(e)}", exc_info=True)
        app.state.service_ready = False
        raise
    
    # Shutdown
    logger.info("Shutting down embedding service...")
    
    shutdown_errors = []

    # Stop the batch processor first so queued callers get an error instead of hanging
    if hasattr(app.state, 'embedding_manager'):
        try:
            await app.state.embedding_manager.stop_batch_processor()
        except Exception as e:
            shutdown_errors.append(f"batch processor: {e}")
            logger.error(f"Error stopping batch processor: {e}", exc_info=True)

    # Unload all models
    if hasattr(app.state, 'model_manager'):
        try:
            await app.state.model_manager.shutdown()
        except Exception as e:
            shutdown_errors.append(f"model manager: {e}")
            logger.error(f"Error shutting down model manager: {e}", exc_info=True)

    if shutdown_errors:
        logger.error(f"Embedding service shutdown completed with errors: {shutdown_errors}")
    else:
        logger.info("Embedding service shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Embedding Service",
        description="NVIDIA NIM-compatible embedding service supporting multiple model families",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    # Include routers
    app.include_router(health.router, tags=["health"])
    app.include_router(models.router, tags=["models"])
    app.include_router(embeddings.router, tags=["embeddings"])
    
    # Add metrics endpoint for Prometheus
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        
        @app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
            
    except ImportError:
        logger.warning("Prometheus client not available, /metrics endpoint disabled")
    
    # Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with consistent error format."""
        return JSONResponse(
            status_code=exc.status_code,
            content=create_error_response(
                message=exc.detail,
                error_type="http_error"
            ).dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        error_details = []
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        logger.error(f"Validation error on {request.url.path}: {error_details}")
        
        # Format error message
        error_messages = []
        for e in error_details:
            error_messages.append(f"{e['field']}: {e['message']}")
        
        return JSONResponse(
            status_code=422,
            content=create_error_response(
                message=f"Request validation failed: {'; '.join(error_messages)}",
                error_type="validation_error"
            ).dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                message="An unexpected error occurred",
                error_type="internal_error"
            ).dict()
        )
    
    # Wrap validation for standardized error format already handled; ensure all non-handled paths include 'error'
    
    # Add middleware to check service readiness
    @app.middleware("http")
    async def service_readiness_middleware(request: Request, call_next):
        """Middleware to check if service is ready before processing requests."""
        
        # Skip readiness check for health endpoints
        if request.url.path.startswith("/health") or request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            response = await call_next(request)
            return response
        
        # Check if service is ready
        if not getattr(request.app.state, 'service_ready', False):
            return JSONResponse(
                status_code=503,
                content=create_error_response(
                    message="Service is not ready",
                    error_type="service_unavailable"
                ).dict()
            )
        
        response = await call_next(request)
        return response

    # API Key + simple rate limit (per-IP) middleware
    API_KEY = os.getenv('EMBEDDING_API_KEY')
    RATE_LIMIT = int(os.getenv('RATE_LIMIT_PER_MINUTE', '100'))
    rl_counters = defaultdict(lambda: {'count':0,'reset':time.time()+60})

    @app.middleware("http")
    async def auth_and_rate_limit(request: Request, call_next):
        path = request.url.path
        if path.startswith('/v1/health') or path in ('/','/docs','/openapi.json','/redoc'):  # exempt
            return await call_next(request)
        # API Key check
        if API_KEY:
            auth_header = request.headers.get('authorization') or request.headers.get('Authorization')
            if not auth_header or not auth_header.lower().startswith('bearer '):
                return JSONResponse(status_code=401, content=create_error_response('Missing bearer token','unauthorized').dict())
            token = auth_header.split(' ',1)[1].strip()
            if token != API_KEY:
                return JSONResponse(status_code=403, content=create_error_response('Invalid API key','forbidden').dict())
        # Rate limit per client IP
        client_ip = request.client.host if request.client else 'unknown'
        bucket = rl_counters[client_ip]
        now = time.time()
        if now > bucket['reset']:
            bucket['count'] = 0
            bucket['reset'] = now + 60
        bucket['count'] += 1
        if bucket['count'] > RATE_LIMIT:
            return JSONResponse(status_code=429, content=create_error_response('Rate limit exceeded','rate_limit_exceeded', code='rate_limit').dict())
        return await call_next(request)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with service information."""
        return {
            "service": "embedding-service",
            "version": "1.0.0",
            "description": "NVIDIA NIM-compatible embedding service",
            "docs": "/docs",
            "health": "/health/ready"
        }
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8009,
    workers: int = 1,
    log_level: str = "info",
    reload: bool = False
):
    """Run the embedding service server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        workers: Number of worker processes
        log_level: Logging level
        reload: Enable auto-reload for development
    """
    
    logger.info(f"Starting embedding service on {host}:{port}")
    
    # Configure uvicorn logging
    uvicorn_config = {
        "app": "src.api.server:create_app",
        "factory": True,
        "host": host,
        "port": port,
        "log_level": log_level,
        "access_log": True,
        "loop": "asyncio"
    }
    
    if reload:
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = ["src"]
    else:
        uvicorn_config["workers"] = workers
    
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    # Development server
    run_server(reload=True)
