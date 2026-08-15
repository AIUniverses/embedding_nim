"""GTE model family implementation."""

from typing import Dict, Any

from .hf_encoder_model import (
    HFEncoderModel,
    SENTENCE_TRANSFORMERS_AVAILABLE
)


class GTEModel(HFEncoderModel):
    """GTE (General Text Embeddings) model family implementation."""

    family_label = "GTE"
    default_batch_size = 64
    uses_input_type = False

    def get_family_info(self) -> Dict[str, Any]:
        """GTE specific model information."""
        info = super().get_family_info()
        info['sentence_transformers_available'] = SENTENCE_TRANSFORMERS_AVAILABLE
        return info
