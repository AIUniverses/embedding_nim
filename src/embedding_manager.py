"""Enhanced embedding manager for the embedding service."""

import logging
import os
import hashlib
import math
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .config_manager import ConfigManager
from .model_manager import ModelManager
from .utils import (
    TextPreprocessor, 
    ImagePreprocessor, 
    ModalityDetector,
    EmbeddingPostprocessor,
    EmbeddingCompressor,
    PreprocessingError
)
from .utils.caching import EmbeddingCache, create_embedding_cache

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3


@dataclass
class EmbeddingRequest:
    """Enhanced embedding request with priority and metadata."""
    inputs: List[Union[str, Dict[str, Any]]]
    model: str
    input_type: Optional[str] = None
    modality: Optional[Union[str, List[str]]] = None
    embedding_type: str = 'float'
    dimensions: Optional[int] = None
    normalize: bool = True
    priority: RequestPriority = RequestPriority.NORMAL
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchRequest:
    """Batch processing request."""
    requests: List[EmbeddingRequest]
    created_at: float
    priority: RequestPriority


class EmbeddingManager:
    """Unified embedding manager with caching, batching, metrics, and preprocessing."""

    def __init__(self, config_manager: ConfigManager, model_manager: ModelManager):
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)

        # Processors
        self.text_processor = TextPreprocessor()
        self.image_processor = ImagePreprocessor()
        self.modality_detector = ModalityDetector()
        self.embedding_processor = EmbeddingPostprocessor()
        self.compressor = EmbeddingCompressor()

        # Caching
        self.cache: Optional[EmbeddingCache] = None
        if self.config_manager.is_caching_enabled():
            try:
                self.cache = create_embedding_cache(self.config_manager)
                self.logger.info(f"Caching enabled ({self.config_manager.get_cache_backend()})")
            except Exception as e:
                self.logger.error(
                    f"Cache init failed, continuing without cache: {e}", exc_info=True
                )

        # Batching
        self.max_batch_size = self.config_manager.get_max_batch_size()
        self.enable_dynamic_batching = self.config_manager.is_dynamic_batching_enabled()
        self.batch_timeout_ms = self.config_manager.get_batch_timeout_ms()
        self._request_queue = asyncio.Queue() if self.enable_dynamic_batching else None
        self._batch_processor_task = None

        # Stats
        self._processing_stats = {
            'total_requests': 0,
            'cached_requests': 0,
            'batched_requests': 0,
            'total_processing_time': 0.0,
            'average_batch_size': 0.0
        }

        # Strict mode (hide extra metadata to mimic NIM/OpenAI)
        self.strict_mode = (os.getenv('STRICT_NIM_MODE', 'false').lower() == 'true')

        self._init_metrics()
        # Internal queue items for cross-request batching
        # Each item: { 'future': Future, 'params': { ... }, 'key': batching_key }
        if self.enable_dynamic_batching and self._request_queue is None:
            self._request_queue = asyncio.Queue()

    def _init_metrics(self):
        if PROMETHEUS_AVAILABLE and self.config_manager.is_monitoring_enabled():
            self.request_counter = Counter('embedding_requests_total','Total embedding requests',['model','status'])
            self.request_duration = Histogram('embedding_request_duration_seconds','Request processing duration',['model'])
            self.cache_hits = Counter('embedding_cache_hits_total','Cache hits',['model'])
            self.batch_size_histogram = Histogram('embedding_batch_size','Batch sizes used for processing')
            self.queue_size_gauge = Gauge('embedding_batch_queue_size','Current dynamic batch queue size')
        else:
            # Create dummy metrics when monitoring is disabled
            class DummyMetric:
                def labels(self, **kwargs): return self
                def inc(self): pass
                def observe(self, value): pass
                def set(self, value): pass
            
            self.request_counter = DummyMetric()
            self.request_duration = DummyMetric()
            self.cache_hits = DummyMetric()
            self.batch_size_histogram = DummyMetric()
            self.queue_size_gauge = DummyMetric()

    async def generate_embeddings(self, inputs: List[Union[str, Dict[str, Any]]], model: str, input_type: Optional[str] = None, modality: Optional[Union[str, List[str]]] = None, embedding_type: str = 'float', dimensions: Optional[int] = None, normalize: bool = True) -> Dict[str, Any]:
        start_time = time.time()
        self._processing_stats['total_requests'] += 1
        try:
            # Normalize model & input_type
            normalized_model, effective_input_type = self.model_manager.validate_model_request(model, input_type)
            # Ensure model loaded (async)
            if hasattr(self.model_manager, 'ensure_model_loaded_async'):
                loaded = await self.model_manager.ensure_model_loaded_async(normalized_model)
            else:
                loaded = self.model_manager.ensure_model_loaded(normalized_model)
            current_model = self.model_manager.get_current_model()
            if not loaded or current_model is None:
                reason = self.model_manager.get_last_load_error() or 'unknown error'
                raise RuntimeError(f"Model '{normalized_model}' is not available: {reason}")
            current_model.validate_embedding_type(embedding_type)

            # Cross-request dynamic batching only when single text input and enabled
            if self.enable_dynamic_batching and self._request_queue and isinstance(inputs, list) and len(inputs) == 1 and isinstance(inputs[0], str):
                fut: asyncio.Future = asyncio.get_event_loop().create_future()
                batching_key = (normalized_model, effective_input_type, embedding_type, dimensions, normalize)
                await self._request_queue.put({
                    'future': fut,
                    'params': {
                        'inputs': inputs,
                        'model': normalized_model,
                        'input_type': effective_input_type,
                        'embedding_type': embedding_type,
                        'dimensions': dimensions,
                        'normalize': normalize,
                        'modality': modality
                    },
                    'key': batching_key,
                    'enqueued_at': time.time()
                })
                if not self._batch_processor_task:
                    await self.start_batch_processor()
                if PROMETHEUS_AVAILABLE:
                    self.queue_size_gauge.set(self._request_queue.qsize())
                result = await fut
                return result

            # Prepare raw text list (before preprocessing for cache key)
            raw_texts = []
            for item in inputs:
                if isinstance(item, str):
                    raw_texts.append(item)
                elif isinstance(item, dict) and 'text' in item:
                    raw_texts.append(item['text'])
                else:
                    raw_texts.append(str(item))

            cache_key_kwargs = {
                'input_type': effective_input_type,
                'embedding_type': embedding_type,
                'dimensions': dimensions,
                'normalize': normalize,
                'modality': modality
            }
            if self.cache:
                cached = await self.cache.get_embeddings(raw_texts, normalized_model, **cache_key_kwargs)
                if cached:
                    self._processing_stats['cached_requests'] += 1
                    if PROMETHEUS_AVAILABLE:
                        self.cache_hits.labels(model=normalized_model).inc()
                    return cached if self.strict_mode else {**cached, 'metadata': {**cached.get('metadata', {}), 'cache': True}}

            processed_inputs = self._preprocess_inputs(inputs, modality)
            embeddings = await self._generate_embeddings_internal(processed_inputs, current_model, effective_input_type, normalize)
            final_embeddings = self._postprocess_embeddings(embeddings, embedding_type, dimensions, current_model)
            generation_time = time.time() - start_time
            response = self._create_response(final_embeddings, model, generation_time, processed_inputs, current_model, cache=False)

            if self.cache:
                await self.cache.set_embeddings(raw_texts, normalized_model, response, **cache_key_kwargs)

            if PROMETHEUS_AVAILABLE:
                self.request_counter.labels(model=normalized_model, status='success').inc()
                self.request_duration.labels(model=normalized_model).observe(generation_time)
                self.batch_size_histogram.observe(len(inputs))

            if self.strict_mode:
                # Remove metadata for strict compatibility
                response.pop('metadata', None)
            return response
        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                self.request_counter.labels(model=model, status='error').inc()
            self.logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise
        
    async def start_batch_processor(self):
        """Start the dynamic batch processor."""
        if self.enable_dynamic_batching and self._batch_processor_task is None:
            self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
            self.logger.info("Dynamic batch processor started")
    
    async def stop_batch_processor(self):
        """Stop the dynamic batch processor."""
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass
            self._batch_processor_task = None
            self._drain_queue(RuntimeError("Batch processor stopped before request was processed"))
            self.logger.info("Dynamic batch processor stopped")

    def _drain_queue(self, error: BaseException) -> None:
        """Fail every queued request so no caller waits on a dead processor."""
        if not self._request_queue:
            return
        pending = []
        while True:
            try:
                pending.append(self._request_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        self._fail_pending(pending, error)
    
    # (old generate_embeddings removed – unified version above)
    
    def _preprocess_inputs(
        self, 
        inputs: List[Union[str, Dict[str, Any]]], 
        modality: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Preprocess input data.
        
        Args:
            inputs: Raw input data
            modality: Modality specification
            
        Returns:
            List of processed input dictionaries
        """
        processed = []
        
        for i, input_data in enumerate(inputs):
            try:
                # Determine modality
                if isinstance(modality, list):
                    input_modality = modality[i] if i < len(modality) else modality[0]
                elif modality:
                    input_modality = modality
                else:
                    # Auto-detect modality
                    input_modality = self.modality_detector.detect_modality(input_data)
                
                # Extract content based on modality
                content = self.modality_detector.extract_content(input_data, input_modality)
                
                # Validate modality support
                current_model = self.model_manager.get_current_model()
                current_model.validate_modality(input_modality)
                
                # Preprocess text if present
                if 'text' in content:
                    max_length_chars = self.config_manager.get_max_input_length()
                    self.text_processor.validate_text(content['text'], max_length_chars)
                    # Truncate strategy
                    strategy = self.config_manager.get_truncate_strategy()
                    if strategy != 'none' and len(content['text']) > max_length_chars:
                        if strategy == 'head':
                            content['text'] = content['text'][:max_length_chars]
                        elif strategy == 'tail':
                            content['text'] = content['text'][-max_length_chars:]
                        elif strategy == 'mid':
                            half = max_length_chars // 2
                            content['text'] = content['text'][:half] + content['text'][-(max_length_chars-half):]
                    content['text'] = self.text_processor.clean_text(content['text'])
                
                # Validate image if present
                if 'image' in content:
                    format_str, image_bytes = self.image_processor.validate_image(content['image'])
                    content['image_format'] = format_str
                    content['image_size'] = len(image_bytes)
                    content['image_bytes'] = image_bytes  # store for embedding
                
                processed.append(content)
                
            except PreprocessingError as e:
                raise ValueError(f"Input {i}: {e}") from e
            except ValueError as e:
                raise ValueError(f"Input {i}: {e}") from e
            except Exception as e:
                self.logger.error(f"Preprocessing failed for input {i}: {e}", exc_info=True)
                raise ValueError(f"Input {i}: Preprocessing failed - {type(e).__name__}: {e}") from e
        
        return processed
    
    async def _generate_embeddings_internal(
        self,
        processed_inputs: List[Dict[str, Any]],
        model: Any,
        input_type: Optional[str],
        normalize: bool
    ) -> np.ndarray:
        """Generate embeddings using the model.
        
        Args:
            processed_inputs: Preprocessed input data
            model: Model instance
            input_type: Input type for the model
            normalize: Whether to normalize embeddings
            
        Returns:
            Generated embeddings
        """
        # Group inputs by modality for efficient processing
        enable_fake_image = os.getenv('ENABLE_FAKE_IMAGE_EMBEDDINGS','false').lower() == 'true'

        # Split inputs
        text_map = {}
        image_map = {}
        multimodal_indices = []
        for idx, inp in enumerate(processed_inputs):
            mod = inp['modality']
            if mod in ('text','text_image') and 'text' in inp:
                text_map[idx] = inp['text']
            if mod in ('image','text_image') and 'image_bytes' in inp:
                image_map[idx] = inp['image_bytes']
            if mod == 'text_image':
                multimodal_indices.append(idx)

        embeddings_text = None
        if text_map:
            ordered_texts = [text_map[i] for i in sorted(text_map.keys())]
            if self.enable_dynamic_batching and len(ordered_texts) > 1:
                embeddings_text = await self._generate_with_batching(ordered_texts, model, input_type, normalize)
            else:
                embeddings_text = model.encode_texts(ordered_texts, input_type=input_type, normalize=normalize)

        # Image embeddings (fake deterministic if enabled)
        embeddings_image = None
        if image_map:
            if not enable_fake_image:
                raise ValueError("Image modality requested but ENABLE_FAKE_IMAGE_EMBEDDINGS not enabled")
            dim = model.embedding_dimension
            img_vecs = []
            for i in sorted(image_map.keys()):
                data = image_map[i]
                h = hashlib.sha256(data).digest()
                # Expand digest deterministically to required dimension
                bytes_needed = dim * 4
                rep = (h * (math.ceil(bytes_needed/len(h))))[:bytes_needed]
                arr = np.frombuffer(rep, dtype=np.uint8).astype(np.float32)
                arr = (arr - 127.5) / 127.5
                arr = arr.reshape(-1)[:dim]
                if normalize:
                    norm = np.linalg.norm(arr) + 1e-9
                    arr = arr / norm
                img_vecs.append(arr)
            embeddings_image = np.vstack(img_vecs)

        # Merge per original order
        result_rows = []
        for i in range(len(processed_inputs)):
            if i in multimodal_indices and embeddings_text is not None and embeddings_image is not None:
                # Average fusion
                t_idx = sorted(text_map.keys()).index(i)
                im_idx = sorted(image_map.keys()).index(i)
                fused = (embeddings_text[t_idx] + embeddings_image[im_idx]) / 2.0
                result_rows.append(fused)
            elif i in text_map and embeddings_text is not None:
                t_idx = sorted(text_map.keys()).index(i)
                result_rows.append(embeddings_text[t_idx])
            elif i in image_map and embeddings_image is not None:
                im_idx = sorted(image_map.keys()).index(i)
                result_rows.append(embeddings_image[im_idx])
            else:
                raise ValueError(f"Cannot produce embedding for input index {i}")
        return np.vstack(result_rows)
    
    async def _generate_with_batching(
        self,
        texts: List[str],
        model: Any,
        input_type: Optional[str],
        normalize: bool
    ) -> np.ndarray:
        """Generate embeddings with dynamic batching.
        
        Args:
            texts: List of texts
            model: Model instance
            input_type: Input type
            normalize: Whether to normalize
            
        Returns:
            Generated embeddings
        """
        # For now, implement simple batching
        # TODO: Implement proper dynamic batching with queue
        
        all_embeddings = []
        batch_size = min(self.max_batch_size, len(texts))
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode_texts(
                batch_texts, input_type=input_type, normalize=normalize
            )
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def _postprocess_embeddings(
        self,
        embeddings: np.ndarray,
        embedding_type: str,
        dimensions: Optional[int],
        model: Any
    ) -> np.ndarray:
        """Postprocess embeddings.
        
        Args:
            embeddings: Raw embeddings
            embedding_type: Target embedding type
            dimensions: Target dimensions
            model: Model instance
            
        Returns:
            Postprocessed embeddings
        """
        # Apply model-specific postprocessing
        processed_embeddings = model.postprocess_embeddings(
            embeddings, embedding_type, dimensions
        )
        
        return processed_embeddings
    
    def _create_response(self, embeddings: np.ndarray, model: str, generation_time: float, processed_inputs: List[Dict[str, Any]], current_model: Any, cache: bool) -> Dict[str, Any]:
        """Create API response.
        
        Args:
            embeddings: Final embeddings
            model: Model name
            generation_time: Generation time in seconds
            processed_inputs: Processed input data
            
        Returns:
            API response dictionary
        """
        # Create embedding objects
        data = []
        for i, embedding in enumerate(embeddings):
            # Convert numpy array to list for JSON serialization
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            
            data.append({
                "index": i,
                "embedding": embedding_list,
                "object": "embedding"
            })
        
        # Calculate token usage (simplified)
        # Token usage via model tokenizer if available
        raw_texts = [pi.get('text','') for pi in processed_inputs]
        try:
            total_tokens = current_model.count_tokens(raw_texts)
        except Exception as e:
            self.logger.warning(
                f"Token counting failed, falling back to whitespace count: {e}", exc_info=True
            )
            total_tokens = sum(len(t.split()) for t in raw_texts)
        
        response = {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            },
            "metadata": {
                "generation_time": generation_time,
                "embedding_dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings),
                "num_embeddings": len(embeddings),
                "cache": cache
            }
        }
        
        return response
    
    async def _batch_processor_loop(self):
        """Main loop for dynamic batch processing."""
        while True:
            batch: List[Dict[str, Any]] = []
            try:
                item = await self._request_queue.get()
                if item is None:
                    continue
                batch = [item]
                deadline = item['enqueued_at'] + (self.batch_timeout_ms / 1000.0)
                key = item['key']
                # Drain queue for compatible items
                while len(batch) < self.max_batch_size and time.time() < deadline:
                    try:
                        wait_remaining = max(0, deadline - time.time())
                        nxt = await asyncio.wait_for(self._request_queue.get(), timeout=wait_remaining)
                        if nxt['key'] == key:
                            batch.append(nxt)
                        else:
                            # Put back if not matching key
                            await self._request_queue.put(nxt)
                            break
                    except asyncio.TimeoutError:
                        break
                # Process batch
                await self._process_request_batch(batch)
            except asyncio.CancelledError:
                self._fail_pending(batch, asyncio.CancelledError("Batch processor stopped"))
                raise
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}", exc_info=True)
                # Never leave callers waiting on a future this loop will no longer complete
                self._fail_pending(batch, e)
    
    async def _process_request_batch(self, requests: List[Dict[str, Any]]):
        """Process a batch of requests.
        
        Args:
            requests: List of request dictionaries
        """
        try:
            if not requests:
                return
            # Combine all single-text inputs
            first_params = requests[0]['params']
            inputs = [r['params']['inputs'][0] for r in requests]
            # Reuse generate logic but avoid recursion (call lower-level path)
            model = first_params['model']
            effective_input_type = first_params['input_type']
            embedding_type = first_params['embedding_type']
            dimensions = first_params['dimensions']
            normalize = first_params['normalize']
            modality = first_params['modality']
            current_model = self.model_manager.get_current_model()
            if current_model is None:
                reason = self.model_manager.get_last_load_error() or 'no model loaded'
                raise RuntimeError(f"Model '{model}' is not available: {reason}")
            processed_inputs = self._preprocess_inputs(inputs, modality)
            embeddings = await self._generate_embeddings_internal(processed_inputs, current_model, effective_input_type, normalize)
            final_embeddings = self._postprocess_embeddings(embeddings, embedding_type, dimensions, current_model)
            response = self._create_response(final_embeddings, model, 0.0, processed_inputs, current_model, cache=False)
            # Dispatch each single embedding
            for idx, req in enumerate(requests):
                single_resp = response.copy()
                single_resp['data'] = [response['data'][idx]]
                req['future'].set_result(single_resp if not self.strict_mode else {k:v for k,v in single_resp.items() if k!='metadata'})
        except Exception as e:
            self.logger.error(f"Batched embedding generation failed: {e}", exc_info=True)
            self._fail_pending(requests, e)

    def _fail_pending(self, requests: List[Dict[str, Any]], error: BaseException) -> None:
        """Propagate an error to every request still awaiting a result."""
        for req in requests:
            if not isinstance(req, dict):
                continue
            future = req.get('future')
            if future is not None and not future.done():
                future.set_exception(error)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding manager statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'max_batch_size': self.max_batch_size,
            'enable_dynamic_batching': self.enable_dynamic_batching,
            'batch_timeout_ms': self.batch_timeout_ms,
            'batch_processor_running': self._batch_processor_task is not None,
            'queue_size': self._request_queue.qsize() if self._request_queue else 0
        }
