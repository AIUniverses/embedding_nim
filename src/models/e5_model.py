"""E5 model family implementation."""

from typing import Optional

from .hf_encoder_model import HFEncoderModel, PrefixedInputTypeMixin


class E5Model(PrefixedInputTypeMixin, HFEncoderModel):
    """E5 model family implementation with query/passage support."""

    family_label = "E5"
    default_batch_size = 32
    default_query_prefix = 'query: '
    default_passage_prefix = 'passage: '
    requires_sentence_transformers = False

    def _get_effective_batch_size(
        self, num_texts: int, requested_batch_size: Optional[int] = None
    ) -> int:
        """Get effective batch size based on text count and memory constraints.

        Args:
            num_texts: Number of texts to process
            requested_batch_size: Requested batch size

        Returns:
            Effective batch size to use
        """
        if requested_batch_size is not None:
            return min(requested_batch_size, num_texts)

        # Auto-determine batch size based on model size and available memory
        default_batch_size = self.get_setting('batch_size', self.default_batch_size)

        # Reduce batch size for longer sequences
        if self.max_seq_length > 256:
            default_batch_size = max(1, default_batch_size // 2)

        return min(default_batch_size, num_texts)
