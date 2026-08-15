"""Unit tests for src/models/base_model.py using a lightweight concrete model."""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from src.models.base_model import BaseEmbeddingModel


class FakeModel(BaseEmbeddingModel):
    """Minimal concrete implementation that never touches a real backend."""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.unload_calls = 0

    def load_model(self) -> bool:
        self.is_loaded = True
        return True

    def unload_model(self) -> None:
        self.unload_calls += 1
        self.is_loaded = False

    def encode_texts(
        self,
        texts: List[str],
        input_type: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        return np.ones((len(texts), self.embedding_dimension), dtype=np.float32)


BASE_CONFIG = {
    "model_id": "intfloat/e5-small-v2",
    "display_name": "E5 Small V2",
    "family": "e5",
    "device": "cpu",
    "embedding_dimension": 8,
    "supports_input_type": True,
    "supported_embedding_types": ["float", "int8", "binary"],
    "supports_dimensions": [4, 8],
    "settings": {"query_prefix": "query: ", "passage_prefix": "passage: "},
}


def make_model(**overrides) -> FakeModel:
    config = {**BASE_CONFIG, **overrides}
    return FakeModel(config)


@pytest.fixture
def model() -> FakeModel:
    return make_model()


class TestInitialization:
    def test_reads_configuration(self, model):
        assert model.model_id == "intfloat/e5-small-v2"
        assert model.model_name == "E5 Small V2"
        assert model.family == "e5"
        assert model.device == "cpu"
        assert model.max_seq_length == 512
        assert model.embedding_dimension == 8
        assert model.normalize_embeddings is True
        assert model.supported_modalities == ["text"]
        assert model.is_loaded is False
        assert model.model is None
        assert model.tokenizer is None

    def test_model_name_defaults_to_model_id(self):
        config = {k: v for k, v in BASE_CONFIG.items() if k != "display_name"}
        assert FakeModel(config).model_name == "intfloat/e5-small-v2"

    def test_requires_mandatory_fields(self):
        with pytest.raises(KeyError):
            FakeModel({"family": "e5", "embedding_dimension": 8})

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseEmbeddingModel(BASE_CONFIG)


class TestPreprocessText:
    def test_strips_whitespace(self, model):
        assert model.preprocess_text("  hello  ") == "hello"

    @pytest.mark.parametrize(
        "input_type, expected", [("query", "query: hi"), ("passage", "passage: hi")]
    )
    def test_adds_prefix_for_input_type(self, model, input_type, expected):
        assert model.preprocess_text("hi", input_type) == expected

    def test_no_prefix_when_model_does_not_support_input_type(self):
        model = make_model(supports_input_type=False)
        assert model.preprocess_text("hi", "query") == "hi"

    def test_no_prefix_when_settings_missing(self):
        model = make_model(settings={})
        assert model.preprocess_text("hi", "query") == "hi"


class TestPostprocessEmbeddings:
    @pytest.fixture
    def vectors(self) -> np.ndarray:
        rng = np.random.default_rng(7)
        return rng.normal(size=(3, 8)).astype(np.float32)

    def test_float_is_default(self, model, vectors):
        result = model.postprocess_embeddings(vectors)
        assert result.dtype == np.float32
        assert np.array_equal(result, vectors)

    def test_reduces_supported_dimensions(self, model, vectors):
        assert np.array_equal(
            model.postprocess_embeddings(vectors, dimensions=4), vectors[:, :4]
        )

    def test_rejects_unsupported_dimensions(self, model, vectors):
        with pytest.raises(ValueError, match="Dimension 2 not supported"):
            model.postprocess_embeddings(vectors, dimensions=2)

    def test_ignores_dimensions_not_smaller_than_input(self, model, vectors):
        assert model.postprocess_embeddings(vectors, dimensions=8).shape == (3, 8)

    def test_int8_conversion(self, model, vectors):
        result = model.postprocess_embeddings(vectors, embedding_type="int8")
        assert result.dtype == np.int8
        assert np.max(np.abs(result)) == 127

    def test_uint8_conversion(self, model, vectors):
        result = model.postprocess_embeddings(vectors, embedding_type="uint8")
        assert result.dtype == np.uint8
        assert np.array_equal(np.min(result, axis=1), np.zeros(len(vectors)))

    @pytest.mark.parametrize(
        "embedding_type, dtype", [("binary", np.int8), ("ubinary", np.uint8)]
    )
    def test_binary_conversion_packs_bits_by_sign(
        self, model, vectors, embedding_type, dtype
    ):
        result = model.postprocess_embeddings(vectors, embedding_type=embedding_type)
        assert result.dtype == dtype
        assert result.shape == (3, 1)
        expected = np.packbits((vectors > 0).astype(np.uint8), axis=1)
        assert np.array_equal(result.astype(np.uint8), expected)


class TestModalitySupport:
    def test_encode_images_rejects_text_only_model(self, model):
        with pytest.raises(NotImplementedError, match="doesn't support image modality"):
            model.encode_images(["data:image/png;base64,AAAA"])

    def test_encode_images_not_implemented_for_capable_model(self):
        model = make_model(supported_modalities=["text", "image"])
        with pytest.raises(NotImplementedError, match="not implemented"):
            model.encode_images(["data:image/png;base64,AAAA"])

    def test_encode_multimodal_rejects_text_only_model(self, model):
        with pytest.raises(NotImplementedError, match="doesn't support multimodal"):
            model.encode_multimodal([{"text": "a"}])

    def test_encode_multimodal_not_implemented_for_capable_model(self):
        model = make_model(supported_modalities=["text", "text_image"])
        with pytest.raises(NotImplementedError, match="not implemented"):
            model.encode_multimodal([{"text": "a"}])


class TestValidation:
    def test_input_type_required_when_supported(self, model):
        with pytest.raises(ValueError, match="requires input_type"):
            model.validate_input_type(None)

    def test_input_type_rejected_when_unsupported(self):
        model = make_model(supports_input_type=False)
        with pytest.raises(ValueError, match="does not support input_type"):
            model.validate_input_type("query")

    def test_invalid_input_type_value(self, model):
        with pytest.raises(ValueError, match="Invalid input_type 'document'"):
            model.validate_input_type("document")

    @pytest.mark.parametrize("input_type", ["query", "passage"])
    def test_valid_input_types(self, model, input_type):
        model.validate_input_type(input_type)

    def test_embedding_type_validation(self, model):
        model.validate_embedding_type("int8")
        with pytest.raises(ValueError, match="not supported by"):
            model.validate_embedding_type("uint8")

    def test_modality_validation(self, model):
        model.validate_modality("text")
        with pytest.raises(ValueError, match="Modality 'image' not supported"):
            model.validate_modality("image")


class TestInfoAndCleanup:
    def test_memory_usage_on_cpu_is_zeroed(self, model):
        assert model.get_memory_usage() == {
            "allocated": 0.0,
            "reserved": 0.0,
            "max_allocated": 0.0,
        }

    def test_model_info_contains_configuration_and_memory(self, model):
        info = model.get_model_info()
        assert info["model_id"] == "intfloat/e5-small-v2"
        assert info["family"] == "e5"
        assert info["is_loaded"] is False
        assert info["device"] == "cpu"
        assert info["embedding_dimension"] == 8
        assert info["supported_embedding_types"] == ["float", "int8", "binary"]
        assert info["memory_usage"] == model.get_memory_usage()

    def test_model_info_tracks_load_state(self, model):
        model.load_model()
        assert model.get_model_info()["is_loaded"] is True

    def test_del_unloads_loaded_model(self, model):
        model.load_model()
        model.__del__()
        assert model.unload_calls == 1
        assert model.is_loaded is False

    def test_del_is_noop_when_not_loaded(self, model):
        model.__del__()
        assert model.unload_calls == 0

    def test_del_swallows_unload_errors(self, model):
        def boom():
            raise RuntimeError("cannot unload")

        model.load_model()
        model.unload_model = boom
        model.__del__()
