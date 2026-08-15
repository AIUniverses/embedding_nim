"""NV-Embed model family implementation for NVIDIA-style models."""

import torch
import numpy as np
from typing import Optional, Dict, Any

from ..utils.vector_ops import pack_binary
from .hf_encoder_model import HFEncoderModel, PrefixedInputTypeMixin, mean_pool


class NVEmbedModel(PrefixedInputTypeMixin, HFEncoderModel):
    """NV-Embed model family implementation for NVIDIA-style embedding models."""

    family_label = "NV-Embed"
    default_batch_size = 32
    default_trust_remote_code = True
    default_query_prefix = 'Represent this sentence for searching relevant passages: '
    default_passage_prefix = 'Represent this sentence for retrieval: '

    def _pool(self, outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """Use the pooler output when the model provides one."""
        if getattr(outputs, 'pooler_output', None) is not None:
            return outputs.pooler_output
        return mean_pool(outputs.last_hidden_state, attention_mask)

    def postprocess_embeddings(
        self,
        embeddings: np.ndarray,
        embedding_type: str = 'float',
        dimensions: Optional[int] = None
    ) -> np.ndarray:
        """Postprocess embeddings with NV-Embed optimizations.

        Args:
            embeddings: Input embeddings
            embedding_type: Target embedding type
            dimensions: Target dimensions

        Returns:
            Processed embeddings
        """
        # NV-Embed models use a median threshold rather than a sign threshold,
        # which keeps the binary representation balanced.
        if embedding_type in ['binary', 'ubinary']:
            return pack_binary(
                embeddings,
                signed=(embedding_type == 'binary'),
                threshold_method='median'
            )

        return super().postprocess_embeddings(embeddings, embedding_type, dimensions)

    def get_family_info(self) -> Dict[str, Any]:
        """NV-Embed specific model information."""
        info = super().get_family_info()
        info['supports_advanced_compression'] = True
        return info
