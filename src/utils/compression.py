"""Embedding compression utilities."""

import numpy as np
import logging
from typing import Dict, Any, Optional

from .vector_ops import (
    compute_thresholds,
    int8_scale_factors,
    l2_normalize,
    pack_binary,
    quantize_int8,
    quantize_uint8,
    uint8_scale_factors
)


class EmbeddingCompressor:
    """Utilities for compressing embeddings to save storage and memory."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compress_embeddings(
        self, 
        embeddings: np.ndarray, 
        compression_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Compress embeddings using specified method.
        
        Args:
            embeddings: Input embeddings (float32)
            compression_type: Type of compression
            **kwargs: Additional compression parameters
            
        Returns:
            Dictionary with compressed data and metadata
        """
        if compression_type == 'int8':
            return self._compress_int8(embeddings, **kwargs)
        elif compression_type == 'uint8':
            return self._compress_uint8(embeddings, **kwargs)
        elif compression_type == 'binary':
            return self._compress_binary(embeddings, signed=True, **kwargs)
        elif compression_type == 'ubinary':
            return self._compress_binary(embeddings, signed=False, **kwargs)
        elif compression_type == 'float16':
            return self._compress_float16(embeddings)
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
    
    def decompress_embeddings(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """Decompress embeddings back to float32.
        
        Args:
            compressed_data: Dictionary with compressed data and metadata
            
        Returns:
            Decompressed embeddings in float32
        """
        compression_type = compressed_data['compression_type']
        
        if compression_type == 'int8':
            return self._decompress_int8(compressed_data)
        elif compression_type == 'uint8':
            return self._decompress_uint8(compressed_data)
        elif compression_type in ['binary', 'ubinary']:
            return self._decompress_binary(compressed_data)
        elif compression_type == 'float16':
            return self._decompress_float16(compressed_data)
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
    
    def _compress_int8(self, embeddings: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Compress to int8 with scaling factors."""
        max_abs = int8_scale_factors(embeddings)
        quantized = quantize_int8(embeddings)
        
        return {
            'compression_type': 'int8',
            'data': quantized,
            'scale_factors': max_abs.flatten(),
            'original_shape': embeddings.shape,
            'compression_ratio': embeddings.nbytes / (quantized.nbytes + max_abs.nbytes)
        }
    
    def _decompress_int8(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """Decompress int8 embeddings."""
        quantized = compressed_data['data']
        scale_factors = compressed_data['scale_factors'].reshape(-1, 1)
        
        # Reconstruct original embeddings
        return quantized.astype(np.float32) / 127 * scale_factors
    
    def _compress_uint8(self, embeddings: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Compress to uint8 with min-max scaling."""
        min_vals, max_vals = uint8_scale_factors(embeddings)
        quantized = quantize_uint8(embeddings)
        
        return {
            'compression_type': 'uint8',
            'data': quantized,
            'min_vals': min_vals.flatten(),
            'max_vals': max_vals.flatten(),
            'original_shape': embeddings.shape,
            'compression_ratio': embeddings.nbytes / (quantized.nbytes + min_vals.nbytes + max_vals.nbytes)
        }
    
    def _decompress_uint8(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """Decompress uint8 embeddings."""
        quantized = compressed_data['data']
        min_vals = compressed_data['min_vals'].reshape(-1, 1)
        max_vals = compressed_data['max_vals'].reshape(-1, 1)
        
        ranges = max_vals - min_vals
        
        # Reconstruct original embeddings
        return quantized.astype(np.float32) / 255 * ranges + min_vals
    
    def _compress_binary(self, embeddings: np.ndarray, signed: bool = True, **kwargs) -> Dict[str, Any]:
        """Compress to binary representation."""
        # Use adaptive threshold or zero threshold
        threshold_method = kwargs.get('threshold_method', 'median')
        if threshold_method not in ('median', 'mean'):
            threshold_method = 'zero'
        
        thresholds = compute_thresholds(embeddings, threshold_method)
        packed = pack_binary(embeddings, signed=signed, thresholds=thresholds)
        
        return {
            'compression_type': 'binary' if signed else 'ubinary',
            'data': packed,
            'thresholds': thresholds.flatten(),
            'threshold_method': threshold_method,
            'original_shape': embeddings.shape,
            'compression_ratio': embeddings.nbytes / (packed.nbytes + thresholds.nbytes)
        }
    
    def _decompress_binary(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """Decompress binary embeddings."""
        packed = compressed_data['data']
        thresholds = compressed_data['thresholds'].reshape(-1, 1)
        original_shape = compressed_data['original_shape']
        
        # Unpack bits
        if packed.dtype == np.int8:
            packed = packed.astype(np.uint8)
        
        unpacked = np.unpackbits(packed, axis=1)
        
        # Trim to original dimension
        unpacked = unpacked[:, :original_shape[1]]
        
        # Convert back to float with thresholds
        # Binary 1 -> threshold + small positive, Binary 0 -> threshold - small negative
        reconstructed = np.where(
            unpacked == 1,
            thresholds + 0.1,  # Small positive offset
            thresholds - 0.1   # Small negative offset
        )
        
        return reconstructed.astype(np.float32)
    
    def _compress_float16(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Compress to float16."""
        compressed = embeddings.astype(np.float16)
        
        return {
            'compression_type': 'float16',
            'data': compressed,
            'original_shape': embeddings.shape,
            'compression_ratio': embeddings.nbytes / compressed.nbytes
        }
    
    def _decompress_float16(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """Decompress float16 embeddings."""
        return compressed_data['data'].astype(np.float32)
    
    def analyze_compression_quality(
        self, 
        original: np.ndarray, 
        compressed_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze compression quality metrics.
        
        Args:
            original: Original embeddings
            compressed_data: Compressed data
            
        Returns:
            Dictionary with quality metrics
        """
        # Decompress for comparison
        decompressed = self.decompress_embeddings(compressed_data)
        
        # Compute metrics
        mse = np.mean((original - decompressed) ** 2)
        mae = np.mean(np.abs(original - decompressed))
        
        # Cosine similarity preservation
        original_normalized = l2_normalize(original)
        decompressed_normalized = l2_normalize(decompressed)
        
        cosine_similarities = np.sum(original_normalized * decompressed_normalized, axis=1)
        mean_cosine_similarity = np.mean(cosine_similarities)
        
        return {
            'compression_ratio': compressed_data['compression_ratio'],
            'mse': float(mse),
            'mae': float(mae),
            'mean_cosine_similarity': float(mean_cosine_similarity),
            'min_cosine_similarity': float(np.min(cosine_similarities)),
            'std_cosine_similarity': float(np.std(cosine_similarities))
        }
    
    def recommend_compression(
        self, 
        embeddings: np.ndarray, 
        target_ratio: Optional[float] = None,
        quality_threshold: float = 0.95
    ) -> str:
        """Recommend best compression method based on requirements.
        
        Args:
            embeddings: Input embeddings
            target_ratio: Target compression ratio (None for best quality)
            quality_threshold: Minimum acceptable cosine similarity preservation
            
        Returns:
            Recommended compression type
        """
        # Test different compression methods
        methods = ['float16', 'int8', 'uint8', 'binary']
        results = {}
        
        for method in methods:
            try:
                compressed = self.compress_embeddings(embeddings, method)
                quality = self.analyze_compression_quality(embeddings, compressed)
                results[method] = quality
            except Exception as e:
                self.logger.warning(f"Failed to test {method} compression: {e}")
                continue
        
        if not results:
            return 'float'  # No compression
        
        # Filter by quality threshold
        valid_methods = [
            method for method, quality in results.items()
            if quality['mean_cosine_similarity'] >= quality_threshold
        ]
        
        if not valid_methods:
            # If no method meets quality threshold, return best quality
            best_method = max(results.keys(), key=lambda m: results[m]['mean_cosine_similarity'])
            self.logger.warning(f"No compression method meets quality threshold {quality_threshold}, using {best_method}")
            return best_method
        
        if target_ratio:
            # Find method that meets target ratio with best quality
            candidates = [
                method for method in valid_methods
                if results[method]['compression_ratio'] >= target_ratio
            ]
            if candidates:
                return max(candidates, key=lambda m: results[m]['mean_cosine_similarity'])
            else:
                # Return highest compression ratio if target not achievable
                return max(valid_methods, key=lambda m: results[m]['compression_ratio'])
        else:
            # Return best quality among valid methods
            return max(valid_methods, key=lambda m: results[m]['mean_cosine_similarity'])
    
    def get_compression_info(self, compression_type: str) -> Dict[str, Any]:
        """Get information about a compression method.
        
        Args:
            compression_type: Type of compression
            
        Returns:
            Information dictionary
        """
        info = {
            'float': {
                'description': 'No compression, full precision',
                'typical_ratio': 1.0,
                'quality_loss': 'None',
                'use_case': 'Maximum accuracy required'
            },
            'float16': {
                'description': 'Half precision floating point',
                'typical_ratio': 2.0,
                'quality_loss': 'Minimal',
                'use_case': 'Balanced accuracy and storage'
            },
            'int8': {
                'description': '8-bit integer with per-vector scaling',
                'typical_ratio': 4.0,
                'quality_loss': 'Low',
                'use_case': 'Production deployments'
            },
            'uint8': {
                'description': '8-bit unsigned integer with min-max scaling',
                'typical_ratio': 4.0,
                'quality_loss': 'Low',
                'use_case': 'Non-negative embeddings'
            },
            'binary': {
                'description': 'Binary representation (1 bit per dimension)',
                'typical_ratio': 32.0,
                'quality_loss': 'Moderate',
                'use_case': 'Maximum compression, fast similarity'
            },
            'ubinary': {
                'description': 'Unsigned binary representation',
                'typical_ratio': 32.0,
                'quality_loss': 'Moderate',
                'use_case': 'Maximum compression'
            }
        }
        
        return info.get(compression_type, {'description': 'Unknown compression type'})
