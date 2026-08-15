"""Sentence Transformers model family implementation."""

from typing import Optional, Dict, Any

from .hf_encoder_model import HFEncoderModel


class SentenceTransformersModel(HFEncoderModel):
    """Generic Sentence Transformers model implementation."""

    family_label = "Sentence Transformers"
    default_batch_size = 128
    requires_sentence_transformers = True
    uses_input_type = False

    def __init__(self, model_config: Dict[str, Any]):
        """Initialize Sentence Transformers model.

        Args:
            model_config: Model configuration dictionary
        """
        super().__init__(model_config)

        # This family has no transformers fallback
        self.use_sentence_transformers = True

    def get_family_info(self) -> Dict[str, Any]:
        """Sentence Transformers specific model information."""
        return {
            'sentence_transformers_version': self._get_sentence_transformers_version()
        }

    def _get_sentence_transformers_version(self) -> Optional[str]:
        """Get sentence-transformers library version.

        Returns:
            Version string or None if not available
        """
        try:
            import sentence_transformers
            return sentence_transformers.__version__
        except (ImportError, AttributeError):
            return None
