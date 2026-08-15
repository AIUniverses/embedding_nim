"""Shared numeric operations on embedding matrices.

These helpers are the single implementation of the normalization, dimension
reduction and quantization math used by model classes, the postprocessor and
the compressor.
"""

import numpy as np
from typing import Optional


def safe_denominator(values: np.ndarray) -> np.ndarray:
    """Replace zeros with ones so an array can be used as a divisor."""
    return np.where(values == 0, 1, values)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings row-wise."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / safe_denominator(norms)


def truncate_dimensions(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
    """Matryoshka-style truncation to `target_dim` columns."""
    if target_dim >= embeddings.shape[1]:
        return embeddings
    return embeddings[:, :target_dim]


def compute_thresholds(embeddings: np.ndarray, method: str = 'median') -> np.ndarray:
    """Compute per-row binarization thresholds.

    Args:
        embeddings: Input embeddings
        method: 'median', 'mean' or 'zero'

    Returns:
        Column vector of thresholds
    """
    if method == 'median':
        return np.median(embeddings, axis=1, keepdims=True)
    if method == 'mean':
        return np.mean(embeddings, axis=1, keepdims=True)
    if method == 'zero':
        return np.zeros((embeddings.shape[0], 1))
    raise ValueError(f"Unknown threshold method: {method}")


def quantize_int8(embeddings: np.ndarray) -> np.ndarray:
    """Quantize to int8 using per-row absolute maximum scaling."""
    return (embeddings / int8_scale_factors(embeddings) * 127).astype(np.int8)


def int8_scale_factors(embeddings: np.ndarray) -> np.ndarray:
    """Per-row scale factors used by int8 quantization."""
    return safe_denominator(np.max(np.abs(embeddings), axis=1, keepdims=True))


def quantize_uint8(embeddings: np.ndarray) -> np.ndarray:
    """Quantize to uint8 using per-row min-max scaling."""
    min_vals, max_vals = uint8_scale_factors(embeddings)
    ranges = safe_denominator(max_vals - min_vals)
    return ((embeddings - min_vals) / ranges * 255).astype(np.uint8)


def uint8_scale_factors(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row minima and maxima used by uint8 quantization."""
    return (
        np.min(embeddings, axis=1, keepdims=True),
        np.max(embeddings, axis=1, keepdims=True)
    )


def pack_binary(
    embeddings: np.ndarray,
    signed: bool = True,
    thresholds: Optional[np.ndarray] = None,
    threshold_method: str = 'median'
) -> np.ndarray:
    """Binarize embeddings against per-row thresholds and pack bits into bytes.

    Args:
        embeddings: Input embeddings
        signed: Whether to return int8 (signed) or uint8 bytes
        thresholds: Precomputed thresholds (computed from `threshold_method` if None)
        threshold_method: Method used when `thresholds` is None

    Returns:
        Packed binary embeddings
    """
    if thresholds is None:
        thresholds = compute_thresholds(embeddings, threshold_method)

    packed = np.packbits((embeddings > thresholds).astype(np.uint8), axis=1)
    return packed.astype(np.int8) if signed else packed
