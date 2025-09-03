"""Enhanced embedding generation endpoints."""

import logging
import time
import uuid
from typing import Union, List
from fastapi import APIRouter, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from ..models import (
    EmbeddingRequest, 
    EmbeddingResponse, 
    ErrorResponse,
    validate_model_name,
    validate_input_length,
    validate_batch_size,
    create_error_response
)

try:
    from prometheus_client import Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

router = APIRouter()
logger = logging.getLogger(__name__)

# Metrics (if Prometheus is available)
if PROMETHEUS_AVAILABLE:
    api_request_counter = Counter(
        'embedding_api_requests_total',
        'Total API requests',
        ['endpoint', 'method', 'status']
    )
    
    api_request_duration = Histogram(
        'embedding_api_request_duration_seconds',
        'API request duration',
        ['endpoint', 'method']
    )


def track_metrics(endpoint: str, method: str = "POST"):
    """Decorator to track API metrics."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except HTTPException as e:
                status = f"error_{e.status_code}"
                raise
            except Exception as e:
                status = "error_500"
                raise
            finally:
                if PROMETHEUS_AVAILABLE:
                    api_request_counter.labels(
                        endpoint=endpoint,
                        method=method,
                        status=status
                    ).inc()
                    
                    api_request_duration.labels(
                        endpoint=endpoint,
                        method=method
                    ).observe(time.time() - start_time)
        
        return wrapper
    return decorator


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
@track_metrics("embeddings", "POST")
async def create_embeddings(request: EmbeddingRequest, http_request: Request):
    """Generate embeddings for input text(s).
    
    Enhanced version with caching, metrics, and priority handling.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Get service components
        embedding_manager = getattr(http_request.app.state, 'embedding_manager', None)
        config_manager = getattr(http_request.app.state, 'config_manager', None)
        
        if not embedding_manager or not config_manager:
            raise HTTPException(
                status_code=500, 
                detail="Service not properly initialized"
            )
        
        # Validate request parameters
        await _validate_request(request, config_manager)
        
        # Normalize input to list format
        if isinstance(request.input, str):
            inputs = [request.input]
        else:
            inputs = request.input
        
        # Parse model name for input_type suffix
        base_model_name, suffix_input_type = validate_model_name(request.model)
        
        # Determine effective input_type
        effective_input_type = request.input_type or suffix_input_type
        
        # Log request with ID
        logger.info(
            f"[{request_id}] Embedding request: model={request.model}, "
            f"inputs={len(inputs)}, input_type={effective_input_type}, "
            f"embedding_type={request.embedding_type}"
        )
        
        # Generate embeddings with enhanced manager
        try:
            response = await embedding_manager.generate_embeddings(
                inputs=inputs,
                model=request.model,
                input_type=effective_input_type,
                modality=request.modality,
                embedding_type=request.embedding_type,
                dimensions=request.dimensions,
                normalize=request.normalize
            )
            
            # Add request metadata
            response['request_id'] = request_id
            response['processing_time'] = time.time() - start_time
            
            # Convert to API response format
            return EmbeddingResponse(**response)
            
        except ValueError as e:
            logger.error(f"[{request_id}] Validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            logger.error(f"[{request_id}] Runtime error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error in create_embeddings endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.websocket("/v1/embeddings/stream")
async def streaming_embeddings(websocket: WebSocket):
    """WebSocket endpoint for streaming embeddings.
    
    Allows real-time embedding generation with live updates.
    """
    await websocket.accept()
    
    try:
        # Get service components
        embedding_manager = getattr(websocket.app.state, 'embedding_manager', None)
        config_manager = getattr(websocket.app.state, 'config_manager', None)
        
        if not embedding_manager or not config_manager:
            await websocket.send_json({"error": "Service not properly initialized"})
            return
        
        while True:
            try:
                # Receive request
                data = await websocket.receive_json()
                request_id = str(uuid.uuid4())
                
                # Validate basic structure
                if 'model' not in data or 'input' not in data:
                    await websocket.send_json({
                        "error": "Missing required fields: model, input",
                        "request_id": request_id
                    })
                    continue
                
                # Send acknowledgment
                await websocket.send_json({
                    "status": "processing",
                    "request_id": request_id,
                    "message": "Request received and processing started"
                })
                
                # Process embeddings
                try:
                    inputs = data['input'] if isinstance(data['input'], list) else [data['input']]
                    
                    response = await embedding_manager.generate_embeddings(
                        inputs=inputs,
                        model=data['model'],
                        input_type=data.get('input_type'),
                        modality=data.get('modality'),
                        embedding_type=data.get('embedding_type', 'float'),
                        dimensions=data.get('dimensions'),
                        normalize=data.get('normalize', True)
                    )
                    
                    # Send success response
                    response['request_id'] = request_id
                    response['status'] = 'completed'
                    await websocket.send_json(response)
                    
                except Exception as e:
                    # Send error response
                    await websocket.send_json({
                        "status": "error",
                        "request_id": request_id,
                        "error": str(e)
                    })
                
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                await websocket.send_json({
                    "status": "error",
                    "error": f"Processing error: {str(e)}"
                })
                
    except Exception as e:
        logger.error(f"WebSocket handler error: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.post("/v1/embeddings/batch", response_model=List[EmbeddingResponse])
@track_metrics("embeddings_batch", "POST")
async def create_embeddings_batch(requests: List[EmbeddingRequest], http_request: Request):
    """Process multiple embedding requests in a single batch.
    
    Optimized for high-throughput scenarios.
    """
    batch_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Get service components
        embedding_manager = getattr(http_request.app.state, 'embedding_manager', None)
        config_manager = getattr(http_request.app.state, 'config_manager', None)
        
        if not embedding_manager or not config_manager:
            raise HTTPException(
                status_code=500, 
                detail="Service not properly initialized"
            )
        
        # Validate batch size
        max_batch_size = config_manager.get_max_batch_size()
        if len(requests) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(requests)} exceeds maximum {max_batch_size}"
            )
        
        logger.info(f"[{batch_id}] Processing batch of {len(requests)} requests")
        
        # Process all requests
        responses = []
        for i, request in enumerate(requests):
            request_id = f"{batch_id}_{i}"
            
            try:
                # Validate individual request
                await _validate_request(request, config_manager)
                
                # Process request
                inputs = request.input if isinstance(request.input, list) else [request.input]
                base_model_name, suffix_input_type = validate_model_name(request.model)
                effective_input_type = request.input_type or suffix_input_type
                
                response = await embedding_manager.generate_embeddings(
                    inputs=inputs,
                    model=request.model,
                    input_type=effective_input_type,
                    modality=request.modality,
                    embedding_type=request.embedding_type,
                    dimensions=request.dimensions,
                    normalize=request.normalize
                )
                
                response['request_id'] = request_id
                responses.append(EmbeddingResponse(**response))
                
            except Exception as e:
                logger.error(f"[{request_id}] Error processing request: {str(e)}")
                # Add error response for this request
                error_response = EmbeddingResponse(
                    object="error",
                    data=[],
                    model=request.model,
                    usage={"prompt_tokens": 0, "total_tokens": 0},
                    error=str(e)
                )
                responses.append(error_response)
        
        total_time = time.time() - start_time
        logger.info(f"[{batch_id}] Batch completed in {total_time:.2f}s")
        
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{batch_id}] Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")


@router.get("/v1/embeddings/metrics")
@track_metrics("embeddings_metrics", "GET") 
async def get_embedding_metrics(http_request: Request):
    """Get embedding service metrics and statistics."""
    try:
        embedding_manager = getattr(http_request.app.state, 'embedding_manager', None)
        
        if not embedding_manager:
            raise HTTPException(status_code=500, detail="Service not initialized")
        
        # Get cache statistics if available
        cache_stats = {}
        if hasattr(embedding_manager, 'cache') and embedding_manager.cache:
            cache_stats = embedding_manager.cache.get_stats()
        
        # Get processing statistics
        processing_stats = getattr(embedding_manager, '_processing_stats', {})
        
        return {
            "cache_statistics": cache_stats,
            "processing_statistics": processing_stats,
            "service_status": "healthy"
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _validate_request(request: EmbeddingRequest, config_manager) -> None:
    """Validate embedding request parameters.
    
    Args:
        request: Embedding request to validate
        config_manager: Configuration manager for validation
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        # Validate model exists
        base_model_name, _ = validate_model_name(request.model)
        available_models = config_manager.get_available_models()
        
        if base_model_name not in available_models:
            available = ', '.join(available_models.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Model '{base_model_name}' not found. Available models: {available}"
            )
        
        # Validate input length
        max_input_length = config_manager.get_max_input_length()
        validate_input_length(request.input, max_input_length)
        
        # Validate batch size
        max_batch_size = config_manager.get_max_batch_size()
        validate_batch_size(request.input, max_batch_size)
        
        # Validate embedding type for model
        supported_types = config_manager.get_supported_embedding_types(base_model_name)
        if request.embedding_type not in supported_types:
            supported = ', '.join(supported_types)
            raise HTTPException(
                status_code=400,
                detail=f"Embedding type '{request.embedding_type}' not supported by model '{base_model_name}'. Supported: {supported}"
            )
        
        # Validate dimensions for model
        if request.dimensions:
            supported_dims = config_manager.get_supported_dimensions(base_model_name)
            if supported_dims and request.dimensions not in supported_dims:
                supported = ', '.join(map(str, supported_dims))
                raise HTTPException(
                    status_code=400,
                    detail=f"Dimensions {request.dimensions} not supported by model '{base_model_name}'. Supported: {supported}"
                )
        
        # Validate modality for model
        if request.modality:
            modalities = [request.modality] if isinstance(request.modality, str) else request.modality
            supported_modalities = config_manager.get_supported_modalities(base_model_name)
            
            for modality in modalities:
                if modality not in supported_modalities:
                    supported = ', '.join(supported_modalities)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Modality '{modality}' not supported by model '{base_model_name}'. Supported: {supported}"
                    )
        
        # Validate input_type for model
        supports_input_type = config_manager.supports_input_type(base_model_name)
        base_model_name_parsed, suffix_input_type = validate_model_name(request.model)
        effective_input_type = request.input_type or suffix_input_type
        
        if supports_input_type and effective_input_type is None:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{base_model_name}' requires input_type parameter or model name suffix (-query/-passage)"
            )
        
        if not supports_input_type and effective_input_type is not None:
            logger.warning(f"Model '{base_model_name}' ignores input_type parameter")
        
        # Validate parameter combinations
        if request.dimensions and request.embedding_type != 'float':
            raise HTTPException(
                status_code=400,
                detail="The 'dimensions' parameter cannot be used with compressed embedding types. Use either 'dimensions' or 'embedding_type', not both."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Request validation failed: {str(e)}")


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings_alt(request: EmbeddingRequest, http_request: Request):
    """Alternative endpoint for embedding generation (without v1 prefix).
    
    This provides backward compatibility with some clients.
    """
    return await create_embeddings(request, http_request)


@router.get("/v1/embeddings/models")
async def list_embedding_models(http_request: Request):
    """List models specifically for embedding generation.
    
    Returns detailed information about embedding capabilities.
    """
    try:
        config_manager = getattr(http_request.app.state, 'config_manager', None)
        model_manager = getattr(http_request.app.state, 'model_manager', None)
        
        if not config_manager or not model_manager:
            raise HTTPException(status_code=500, detail="Service not properly initialized")
        
        available_models = config_manager.get_available_models()
        detailed_models = []
        
        for model_name in available_models.keys():
            try:
                model_info = model_manager.get_model_info(model_name)
                
                # Add embedding-specific information
                detailed_info = {
                    "id": model_name,
                    "display_name": model_info.get("display_name", model_name),
                    "family": model_info.get("family"),
                    "embedding_dimension": model_info.get("embedding_dimension"),
                    "max_seq_length": model_info.get("max_seq_length"),
                    "supports_input_type": model_info.get("supports_input_type", False),
                    "supported_embedding_types": model_info.get("supported_embedding_types", []),
                    "supported_modalities": model_info.get("supported_modalities", []),
                    "supports_dimensions": config_manager.get_supported_dimensions(model_name),
                    "is_loaded": model_info.get("is_loaded", False)
                }
                
                detailed_models.append(detailed_info)
                
            except Exception as e:
                logger.warning(f"Failed to get info for model {model_name}: {str(e)}")
                continue
        
        return {
            "object": "list",
            "data": detailed_models,
            "total": len(detailed_models)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing embedding models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list embedding models: {str(e)}")
