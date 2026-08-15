"""Shared implementation for HuggingFace/SentenceTransformers backed model families."""

import logging
import torch
import numpy as np
from typing import List, Optional, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from transformers import AutoTokenizer, AutoModel

from .base_model import BaseEmbeddingModel


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-masked mean pooling over token embeddings."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


class HFEncoderModel(BaseEmbeddingModel):
    """Base class for encoder families loaded via SentenceTransformers or transformers.

    Subclasses customize behaviour through the class attributes below and by
    overriding `preprocess_text` / `_pool` when a family needs different pooling
    or prefixes.
    """

    # Human readable family name used in log messages
    family_label = "HuggingFace"
    # Batch size used when neither the caller nor the model config specifies one
    default_batch_size = 32
    # Default for settings.trust_remote_code
    default_trust_remote_code = False
    # Whether SentenceTransformers is the only supported backend
    requires_sentence_transformers = False
    # Whether the family accepts an input_type (query/passage) parameter
    uses_input_type = False

    def __init__(self, model_config: Dict[str, Any]):
        """Initialize the model.

        Args:
            model_config: Model configuration dictionary
        """
        super().__init__(model_config)
        self.logger = logging.getLogger(__name__)

        if self.requires_sentence_transformers and not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers library is required for this model family")

        self.use_sentence_transformers = (
            self.get_setting('use_sentence_transformers', True)
            and SENTENCE_TRANSFORMERS_AVAILABLE
        )

    def get_setting(self, name: str, default: Any = None) -> Any:
        """Read a value from the model config `settings` section."""
        return self.model_config.get('settings', {}).get(name, default)

    @property
    def trust_remote_code(self) -> bool:
        """Whether remote code execution is allowed when loading the model."""
        return self.get_setting('trust_remote_code', self.default_trust_remote_code)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Torch dtype requested by the model config."""
        return torch.float16 if self.model_config.get('torch_dtype') == 'float16' else torch.float32

    def load_model(self) -> bool:
        """Load the model (and tokenizer for the transformers backend).

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading {self.family_label} model: {self.model_id}")

            if self.use_sentence_transformers:
                self._load_sentence_transformers()
            else:
                self._load_transformers()

            self.is_loaded = True
            memory_info = self.get_memory_usage()
            self.logger.info(
                f"{self.family_label} model loaded successfully. "
                f"Memory usage: {memory_info['allocated']:.2f}GB"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to load {self.family_label} model {self.model_id}: {str(e)}")
            return False

    def _load_sentence_transformers(self) -> None:
        """Load the model through SentenceTransformers."""
        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            trust_remote_code=self.trust_remote_code
        )

        self.model.eval()
        if self.model_config.get('torch_dtype') == 'float16' and self.device == 'cuda':
            self.model.half()

    def _load_transformers(self) -> None:
        """Load the model through transformers directly for more control."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code
        )
        self.model.to(self.device)
        self.model.eval()

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        try:
            if self.model is not None:
                del self.model
                self.model = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.is_loaded = False
            self.logger.info(f"{self.family_label} model {self.model_id} unloaded")

        except Exception as e:
            self.logger.error(f"Error unloading {self.family_label} model: {str(e)}")

    def encode_texts(
        self,
        texts: List[str],
        input_type: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encode texts into embeddings.

        Args:
            texts: List of texts to encode
            input_type: Type of input ('query' or 'passage'), ignored by families
                that don't support it
            normalize: Whether to normalize embeddings
            batch_size: Batch size for processing

        Returns:
            NumPy array of embeddings
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.uses_input_type:
            self.validate_input_type(input_type)
            processed_texts = [self.preprocess_text(text, input_type) for text in texts]
        else:
            if input_type is not None:
                self.logger.warning(
                    f"{self.family_label} model {self.model_id} ignores input_type parameter"
                )
            processed_texts = [self.preprocess_text(text) for text in texts]

        if batch_size is None:
            batch_size = self.get_setting('batch_size', self.default_batch_size)

        effective_normalize = normalize if normalize is not None else self.normalize_embeddings

        try:
            if self.use_sentence_transformers:
                return self.model.encode(
                    processed_texts,
                    batch_size=batch_size,
                    normalize_embeddings=effective_normalize,
                    convert_to_numpy=True,
                    show_progress_bar=len(processed_texts) > 100
                )

            return self._encode_with_transformers(
                processed_texts,
                batch_size=batch_size,
                normalize=effective_normalize
            )

        except Exception as e:
            self.logger.error(f"Error encoding texts with {self.family_label} model: {str(e)}")
            raise

    def _encode_with_transformers(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode texts using transformers directly.

        Args:
            texts: Preprocessed texts to encode
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings

        Returns:
            NumPy array of embeddings
        """
        if batch_size is None:
            batch_size = self.get_setting('batch_size', self.default_batch_size)

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            inputs = self.tokenizer(
                texts[i:i + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._pool(outputs, inputs['attention_mask'])

                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def _pool(self, outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token embeddings into a single vector per input."""
        return mean_pool(outputs.last_hidden_state, attention_mask)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including family specific details."""
        info = super().get_model_info()
        info['family_specific'] = self.get_family_info()
        return info

    def get_family_info(self) -> Dict[str, Any]:
        """Family specific fields reported by `get_model_info`."""
        return {'use_sentence_transformers': self.use_sentence_transformers}


class PrefixedInputTypeMixin:
    """Adds query/passage prefixing for families that support input_type."""

    uses_input_type = True
    default_query_prefix = ''
    default_passage_prefix = ''

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.query_prefix = self.get_setting('query_prefix', self.default_query_prefix)
        self.passage_prefix = self.get_setting('passage_prefix', self.default_passage_prefix)

    def preprocess_text(self, text: str, input_type: Optional[str] = None) -> str:
        """Prefix the text according to the requested input type.

        Args:
            text: Input text
            input_type: Type of input ('query' or 'passage')

        Returns:
            Preprocessed text with appropriate prefix
        """
        text = text.strip()

        if input_type == 'passage':
            return self.passage_prefix + text

        # Retrieval models default to the query prefix when no type is given
        if input_type == 'query' or self.supports_input_type:
            return self.query_prefix + text

        return text

    def get_family_info(self) -> Dict[str, Any]:
        info = super().get_family_info()
        info.update({
            'query_prefix': self.query_prefix,
            'passage_prefix': self.passage_prefix
        })
        return info
