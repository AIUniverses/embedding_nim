"""Embedding postprocessing utilities."""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union

from .vector_ops import (
    l2_normalize,
    pack_binary,
    quantize_int8,
    quantize_uint8,
    truncate_dimensions
)


class EmbeddingPostprocessor:
    """Utilities for postprocessing embeddings."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Normalized embeddings
        """
        return l2_normalize(embeddings)
    
    def reduce_dimensions(
        self, 
        embeddings: np.ndarray, 
        target_dim: int,
        method: str = 'truncate'
    ) -> np.ndarray:
        """Reduce embedding dimensions.
        
        Args:
            embeddings: Input embeddings
            target_dim: Target dimension size
            method: Reduction method ('truncate', 'pca', 'random')
            
        Returns:
            Dimension-reduced embeddings
        """
        if target_dim >= embeddings.shape[1]:
            return embeddings
        
        if method == 'truncate':
            return truncate_dimensions(embeddings, target_dim)
        elif method == 'pca':
            return self._pca_reduce(embeddings, target_dim)
        elif method == 'random':
            return self._random_projection(embeddings, target_dim)
        else:
            raise ValueError(f"Unknown dimension reduction method: {method}")
    
    def _pca_reduce(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """PCA dimension reduction.
        
        Args:
            embeddings: Input embeddings
            target_dim: Target dimension size
            
        Returns:
            PCA-reduced embeddings
        """
        try:
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=target_dim)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            self.logger.info(f"PCA reduction: explained variance ratio = {pca.explained_variance_ratio_.sum():.3f}")
            return reduced_embeddings
            
        except ImportError:
            self.logger.warning("scikit-learn not available, falling back to truncation")
            return embeddings[:, :target_dim]
    
    def _random_projection(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Random projection dimension reduction.
        
        Args:
            embeddings: Input embeddings
            target_dim: Target dimension size
            
        Returns:
            Random projection reduced embeddings
        """
        try:
            from sklearn.random_projection import GaussianRandomProjection
            
            rp = GaussianRandomProjection(n_components=target_dim, random_state=42)
            reduced_embeddings = rp.fit_transform(embeddings)
            return reduced_embeddings
            
        except ImportError:
            self.logger.warning("scikit-learn not available, falling back to truncation")
            return embeddings[:, :target_dim]
    
    def quantize_embeddings(
        self, 
        embeddings: np.ndarray, 
        quantization_type: str
    ) -> np.ndarray:
        """Quantize embeddings to reduce memory usage.
        
        Args:
            embeddings: Input embeddings (float32)
            quantization_type: Type of quantization
            
        Returns:
            Quantized embeddings
        """
        if quantization_type == 'int8':
            return quantize_int8(embeddings)
        elif quantization_type == 'uint8':
            return quantize_uint8(embeddings)
        elif quantization_type in ('binary', 'ubinary'):
            # Adaptive threshold based on the value distribution
            return pack_binary(
                embeddings,
                signed=(quantization_type == 'binary'),
                threshold_method='median'
            )
        elif quantization_type == 'float':
            return embeddings.astype(np.float32)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    def compute_similarities(
        self, 
        embeddings1: np.ndarray, 
        embeddings2: Optional[np.ndarray] = None,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """Compute similarities between embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings (if None, compute self-similarity)
            metric: Similarity metric ('cosine', 'dot', 'euclidean')
            
        Returns:
            Similarity matrix
        """
        if embeddings2 is None:
            embeddings2 = embeddings1
        
        if metric == 'cosine':
            return self._cosine_similarity(embeddings1, embeddings2)
        elif metric == 'dot':
            return np.dot(embeddings1, embeddings2.T)
        elif metric == 'euclidean':
            return self._euclidean_distance(embeddings1, embeddings2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def _cosine_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        return np.dot(l2_normalize(embeddings1), l2_normalize(embeddings2).T)
    
    def _euclidean_distance(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance (negative for similarity)."""
        # Compute pairwise distances
        diff = embeddings1[:, np.newaxis, :] - embeddings2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        # Return negative distance as similarity
        return -distances
    
    def aggregate_embeddings(
        self, 
        embeddings: List[np.ndarray], 
        method: str = 'mean'
    ) -> np.ndarray:
        """Aggregate multiple embeddings.
        
        Args:
            embeddings: List of embedding arrays
            method: Aggregation method ('mean', 'max', 'sum', 'weighted')
            
        Returns:
            Aggregated embeddings
        """
        if not embeddings:
            raise ValueError("Empty embeddings list")
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        if method == 'mean':
            return np.mean(embeddings, axis=0)
        elif method == 'max':
            return np.max(embeddings, axis=0)
        elif method == 'sum':
            return np.sum(embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def filter_embeddings(
        self, 
        embeddings: np.ndarray, 
        texts: List[str],
        min_length: int = 5,
        max_length: Optional[int] = None,
        remove_duplicates: bool = False
    ) -> tuple[np.ndarray, List[str]]:
        """Filter embeddings based on text criteria.
        
        Args:
            embeddings: Input embeddings
            texts: Corresponding texts
            min_length: Minimum text length
            max_length: Maximum text length
            remove_duplicates: Whether to remove duplicate texts
            
        Returns:
            Tuple of (filtered_embeddings, filtered_texts)
        """
        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts must have same length")
        
        # Create filter mask
        mask = np.ones(len(texts), dtype=bool)
        
        # Length filtering
        for i, text in enumerate(texts):
            text_len = len(text.strip())
            if text_len < min_length:
                mask[i] = False
            if max_length and text_len > max_length:
                mask[i] = False
        
        # Remove duplicates if requested
        if remove_duplicates:
            seen_texts = set()
            for i, text in enumerate(texts):
                if text in seen_texts:
                    mask[i] = False
                else:
                    seen_texts.add(text)
        
        # Apply filter
        filtered_embeddings = embeddings[mask]
        filtered_texts = [text for i, text in enumerate(texts) if mask[i]]
        
        self.logger.info(f"Filtered {np.sum(~mask)} embeddings, kept {len(filtered_texts)}")
        
        return filtered_embeddings, filtered_texts
    
    def validate_embeddings(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Validate and analyze embedding quality.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'has_nan': np.isnan(embeddings).any(),
            'has_inf': np.isinf(embeddings).any(),
            'min_value': float(np.min(embeddings)),
            'max_value': float(np.max(embeddings)),
            'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1)))
        }
        
        # Check for potential issues
        issues = []
        if results['has_nan']:
            issues.append("Contains NaN values")
        if results['has_inf']:
            issues.append("Contains infinite values")
        if results['mean_norm'] < 0.1:
            issues.append("Very small embedding norms")
        if results['std_norm'] > 10:
            issues.append("High variance in embedding norms")
        
        results['issues'] = issues
        results['is_valid'] = len(issues) == 0
        
        return results
