"""Utilities package for embedding service."""

from .preprocessing import TextPreprocessor, ImagePreprocessor, ModalityDetector, PreprocessingError
from .postprocessing import EmbeddingPostprocessor
from .compression import EmbeddingCompressor

# Caching utilities (optional import)
try:
    from .caching import EmbeddingCache, create_embedding_cache, create_cache_backend
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

__all__ = [
    'TextPreprocessor',
    'ImagePreprocessor', 
    'ModalityDetector',
    'PreprocessingError',
    'EmbeddingPostprocessor',
    'EmbeddingCompressor'
]

if CACHING_AVAILABLE:
    __all__.extend([
        'EmbeddingCache',
        'create_embedding_cache',
        'create_cache_backend'
    ])
