"""Unit tests for the E5, GTE, NV-Embed and Sentence Transformers model families.

The heavy backends (`sentence_transformers` / `transformers`) are replaced with
stubs so no weights are downloaded.
"""

from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from src.models import e5_model as e5_module
from src.models import gte_model as gte_module
from src.models import nv_embed_model as nv_module
from src.models import sentence_transformers_model as st_module
from src.models.e5_model import E5Model
from src.models.gte_model import GTEModel
from src.models.nv_embed_model import NVEmbedModel
from src.models.sentence_transformers_model import SentenceTransformersModel

E5_CONFIG: Dict[str, Any] = {
    "model_id": "intfloat/e5-small-v2",
    "family": "e5",
    "device": "cpu",
    "embedding_dimension": 4,
    "max_seq_length": 128,
    "supports_input_type": True,
    "supported_embedding_types": ["float", "int8"],
    "settings": {"query_prefix": "query: ", "passage_prefix": "passage: "},
}

GTE_CONFIG: Dict[str, Any] = {
    "model_id": "thenlper/gte-base",
    "family": "gte",
    "device": "cpu",
    "embedding_dimension": 4,
    "max_seq_length": 128,
    "supports_input_type": False,
}

NV_CONFIG: Dict[str, Any] = {
    "model_id": "nvidia/nv-embed-v1",
    "family": "nv_embed",
    "device": "cpu",
    "embedding_dimension": 16,
    "max_seq_length": 128,
    "supports_input_type": True,
    "supported_embedding_types": ["float", "binary", "ubinary"],
}

ST_CONFIG: Dict[str, Any] = {
    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "family": "sentence_transformers",
    "device": "cpu",
    "embedding_dimension": 4,
}


class StubSentenceTransformer:
    """Records constructor and encode arguments in place of the real class."""

    instances: List["StubSentenceTransformer"] = []
    raise_on_init = False

    def __init__(self, model_id, device=None, trust_remote_code=False):
        if StubSentenceTransformer.raise_on_init:
            raise RuntimeError("weights unavailable")
        self.model_id = model_id
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.eval_calls = 0
        self.half_calls = 0
        self.encode_calls: List[Dict[str, Any]] = []
        StubSentenceTransformer.instances.append(self)

    def eval(self):
        self.eval_calls += 1
        return self

    def half(self):
        self.half_calls += 1
        return self

    def encode(self, texts, **kwargs):
        self.encode_calls.append({"texts": list(texts), **kwargs})
        return np.ones((len(texts), 4), dtype=np.float32)


@pytest.fixture(autouse=True)
def reset_stub():
    StubSentenceTransformer.instances = []
    StubSentenceTransformer.raise_on_init = False
    yield
    StubSentenceTransformer.instances = []
    StubSentenceTransformer.raise_on_init = False


@pytest.fixture
def sentence_transformer_stub(monkeypatch):
    monkeypatch.setattr(e5_module, "SentenceTransformer", StubSentenceTransformer)
    monkeypatch.setattr(gte_module, "SentenceTransformer", StubSentenceTransformer)
    monkeypatch.setattr(nv_module, "SentenceTransformer", StubSentenceTransformer)
    monkeypatch.setattr(st_module, "SentenceTransformer", StubSentenceTransformer)
    return StubSentenceTransformer


@pytest.fixture
def transformers_stubs(monkeypatch):
    """Patch `AutoTokenizer`/`AutoModel` in a module with lightweight stubs."""

    class StubTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return cls()

        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.ones((len(texts), 3), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
            }

    class StubOutput:
        def __init__(self, batch_size):
            self.last_hidden_state = torch.ones((batch_size, 3, 4))

    class StubAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, torch_dtype=None, trust_remote_code=False):
            instance = cls()
            instance.torch_dtype = torch_dtype
            return instance

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            output = StubOutput(inputs["attention_mask"].shape[0])
            if getattr(self, "with_pooler_output", False):
                output.pooler_output = torch.arange(
                    output.last_hidden_state.shape[0] * 4, dtype=torch.float32
                ).reshape(output.last_hidden_state.shape[0], 4)
            return output

    def patch(module):
        monkeypatch.setattr(module, "AutoTokenizer", StubTokenizer)
        monkeypatch.setattr(module, "AutoModel", StubAutoModel)

    return patch


class TestE5Initialization:
    def test_reads_family_settings(self):
        model = E5Model(E5_CONFIG)
        assert model.use_sentence_transformers is True
        assert model.query_prefix == "query: "
        assert model.passage_prefix == "passage: "

    def test_settings_defaults(self):
        model = E5Model({**E5_CONFIG, "settings": {}})
        assert model.query_prefix == "query: "
        assert model.passage_prefix == "passage: "
        assert model.use_sentence_transformers is True

    def test_model_info_includes_family_specific_section(self):
        info = E5Model(E5_CONFIG).get_model_info()
        assert info["family_specific"] == {
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "use_sentence_transformers": True,
        }


class TestE5Preprocessing:
    @pytest.fixture
    def model(self) -> E5Model:
        return E5Model(E5_CONFIG)

    @pytest.mark.parametrize(
        "input_type, expected",
        [("query", "query: hi"), ("passage", "passage: hi"), (None, "query: hi")],
    )
    def test_prefixes(self, model, input_type, expected):
        assert model.preprocess_text("  hi  ", input_type) == expected

    def test_no_prefix_when_input_type_unsupported(self):
        model = E5Model({**E5_CONFIG, "supports_input_type": False})
        assert model.preprocess_text(" hi ") == "hi"


class TestE5SentenceTransformersPath:
    def test_load_configures_the_backend(self, sentence_transformer_stub):
        model = E5Model(E5_CONFIG)
        assert model.load_model() is True
        assert model.is_loaded is True

        backend = sentence_transformer_stub.instances[-1]
        assert backend.model_id == "intfloat/e5-small-v2"
        assert backend.device == "cpu"
        assert backend.trust_remote_code is False
        assert backend.eval_calls == 1
        assert backend.half_calls == 0

    def test_load_failure_is_reported(self, sentence_transformer_stub):
        sentence_transformer_stub.raise_on_init = True
        model = E5Model(E5_CONFIG)
        assert model.load_model() is False
        assert model.is_loaded is False

    def test_encode_requires_a_loaded_model(self):
        with pytest.raises(RuntimeError, match="Model not loaded"):
            E5Model(E5_CONFIG).encode_texts(["hi"], input_type="query")

    def test_encode_applies_prefixes_and_batch_size(self, sentence_transformer_stub):
        model = E5Model(
            {**E5_CONFIG, "settings": {**E5_CONFIG["settings"], "batch_size": 8}}
        )
        model.load_model()
        embeddings = model.encode_texts(["a", "b"], input_type="passage")

        call = sentence_transformer_stub.instances[-1].encode_calls[-1]
        assert call["texts"] == ["passage: a", "passage: b"]
        assert call["batch_size"] == 8
        assert call["normalize_embeddings"] is True
        assert call["convert_to_numpy"] is True
        assert call["show_progress_bar"] is False
        assert embeddings.shape == (2, 4)

    def test_encode_uses_explicit_batch_size(self, sentence_transformer_stub):
        model = E5Model(E5_CONFIG)
        model.load_model()
        model.encode_texts(["a"], input_type="query", batch_size=2)
        assert (
            sentence_transformer_stub.instances[-1].encode_calls[-1]["batch_size"] == 2
        )

    def test_encode_rejects_invalid_input_type(self, sentence_transformer_stub):
        model = E5Model(E5_CONFIG)
        model.load_model()
        with pytest.raises(ValueError, match="Invalid input_type"):
            model.encode_texts(["a"], input_type="document")

    def test_encode_propagates_backend_errors(self, sentence_transformer_stub):
        model = E5Model(E5_CONFIG)
        model.load_model()

        def boom(*args, **kwargs):
            raise RuntimeError("cuda oom")

        model.model.encode = boom
        with pytest.raises(RuntimeError, match="cuda oom"):
            model.encode_texts(["a"], input_type="query")

    def test_unload_releases_backend(self, sentence_transformer_stub):
        model = E5Model(E5_CONFIG)
        model.load_model()
        model.unload_model()
        assert model.model is None
        assert model.tokenizer is None
        assert model.is_loaded is False


class TestE5TransformersPath:
    @pytest.fixture
    def model(self, transformers_stubs) -> E5Model:
        transformers_stubs(e5_module)
        config = {
            **E5_CONFIG,
            "settings": {**E5_CONFIG["settings"], "use_sentence_transformers": False},
        }
        return E5Model(config)

    def test_load_uses_tokenizer_and_automodel(self, model):
        assert model.load_model() is True
        assert model.tokenizer is not None
        assert model.model.device == "cpu"
        assert model.model.torch_dtype is torch.float32

    def test_float16_dtype_is_requested(self, monkeypatch, model):
        model.model_config["torch_dtype"] = "float16"
        model.load_model()
        assert model.model.torch_dtype is torch.float16

    def test_encode_mean_pools_and_normalizes(self, model):
        model.load_model()
        embeddings = model.encode_texts(["a", "b"], input_type="query")
        assert embeddings.shape == (2, 4)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)

    def test_encode_without_normalisation(self, model):
        model.load_model()
        embeddings = model.encode_texts(["a"], input_type="query", normalize=False)
        assert np.allclose(embeddings, np.ones((1, 4)))

    def test_encode_processes_multiple_batches(self, model):
        model.load_model()
        embeddings = model.encode_texts(
            ["a", "b", "c"], input_type="query", batch_size=2
        )
        assert embeddings.shape == (3, 4)


class TestE5EffectiveBatchSize:
    @pytest.fixture
    def model(self) -> E5Model:
        return E5Model(E5_CONFIG)

    def test_requested_batch_size_is_capped_by_input_count(self, model):
        assert model._get_effective_batch_size(3, 8) == 3
        assert model._get_effective_batch_size(20, 8) == 8

    def test_long_sequences_halve_the_default(self):
        model = E5Model({**E5_CONFIG, "max_seq_length": 512})
        assert model._get_effective_batch_size(100) == 16

    def test_short_sequences_keep_the_default(self, model):
        assert model._get_effective_batch_size(100) == 32


class TestGTEModel:
    def test_sentence_transformers_usage_requires_the_library(self, monkeypatch):
        monkeypatch.setattr(gte_module, "SENTENCE_TRANSFORMERS_AVAILABLE", False)
        assert GTEModel(GTE_CONFIG).use_sentence_transformers is False

    def test_settings_can_disable_sentence_transformers(self):
        config = {**GTE_CONFIG, "settings": {"use_sentence_transformers": False}}
        assert GTEModel(config).use_sentence_transformers is False

    def test_preprocess_ignores_input_type(self):
        assert GTEModel(GTE_CONFIG).preprocess_text("  hi  ", "query") == "hi"

    def test_model_info_includes_family_specific_section(self):
        info = GTEModel(GTE_CONFIG).get_model_info()
        assert info["family_specific"] == {
            "use_sentence_transformers": True,
            "sentence_transformers_available": True,
        }

    def test_load_and_unload_with_sentence_transformers(
        self, sentence_transformer_stub
    ):
        model = GTEModel(GTE_CONFIG)
        assert model.load_model() is True
        assert sentence_transformer_stub.instances[-1].eval_calls == 1
        model.unload_model()
        assert model.model is None
        assert model.tokenizer is None
        assert model.is_loaded is False

    def test_load_failure_is_reported(self, sentence_transformer_stub):
        sentence_transformer_stub.raise_on_init = True
        model = GTEModel(GTE_CONFIG)
        assert model.load_model() is False
        assert model.is_loaded is False

    def test_encode_requires_a_loaded_model(self):
        with pytest.raises(RuntimeError, match="Model not loaded"):
            GTEModel(GTE_CONFIG).encode_texts(["hi"])

    def test_encode_defaults_to_batch_size_64(self, sentence_transformer_stub):
        model = GTEModel(GTE_CONFIG)
        model.load_model()
        embeddings = model.encode_texts(["a", "b"], input_type="query")
        call = sentence_transformer_stub.instances[-1].encode_calls[-1]
        assert call["batch_size"] == 64
        assert call["texts"] == ["a", "b"]
        assert embeddings.shape == (2, 4)

    def test_encode_propagates_backend_errors(self, sentence_transformer_stub):
        model = GTEModel(GTE_CONFIG)
        model.load_model()

        def boom(*args, **kwargs):
            raise RuntimeError("encode failed")

        model.model.encode = boom
        with pytest.raises(RuntimeError, match="encode failed"):
            model.encode_texts(["a"])


class TestGTETransformersPath:
    @pytest.fixture
    def model(self, transformers_stubs) -> GTEModel:
        transformers_stubs(gte_module)
        return GTEModel(
            {**GTE_CONFIG, "settings": {"use_sentence_transformers": False}}
        )

    def test_load_uses_tokenizer_and_automodel(self, model):
        assert model.load_model() is True
        assert model.tokenizer is not None
        assert model.model.device == "cpu"
        assert model.model.torch_dtype is torch.float32

    def test_encode_mean_pools_and_normalizes(self, model):
        model.load_model()
        embeddings = model.encode_texts(["a", "b", "c"], batch_size=2)
        assert embeddings.shape == (3, 4)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)

    def test_encode_without_normalisation(self, model):
        model.load_model()
        assert np.allclose(model.encode_texts(["a"], normalize=False), np.ones((1, 4)))


class TestNVEmbedModel:
    def test_default_prefixes(self):
        model = NVEmbedModel(NV_CONFIG)
        assert model.query_prefix.startswith("Represent this sentence for searching")
        assert model.passage_prefix.startswith("Represent this sentence for retrieval")

    def test_settings_override_prefixes(self):
        model = NVEmbedModel(
            {**NV_CONFIG, "settings": {"query_prefix": "q: ", "passage_prefix": "p: "}}
        )
        assert model.preprocess_text(" hi ", "query") == "q: hi"
        assert model.preprocess_text(" hi ", "passage") == "p: hi"

    def test_missing_input_type_defaults_to_query_prefix(self):
        model = NVEmbedModel({**NV_CONFIG, "settings": {"query_prefix": "q: "}})
        assert model.preprocess_text("hi") == "q: hi"

    def test_no_prefix_when_input_type_unsupported(self):
        model = NVEmbedModel({**NV_CONFIG, "supports_input_type": False})
        assert model.preprocess_text(" hi ") == "hi"

    def test_sentence_transformers_usage_requires_the_library(self, monkeypatch):
        monkeypatch.setattr(nv_module, "SENTENCE_TRANSFORMERS_AVAILABLE", False)
        assert NVEmbedModel(NV_CONFIG).use_sentence_transformers is False

    def test_load_defaults_to_trusting_remote_code(self, sentence_transformer_stub):
        model = NVEmbedModel(NV_CONFIG)
        assert model.load_model() is True
        assert sentence_transformer_stub.instances[-1].trust_remote_code is True

    def test_load_failure_is_reported(self, sentence_transformer_stub):
        sentence_transformer_stub.raise_on_init = True
        assert NVEmbedModel(NV_CONFIG).load_model() is False

    def test_unload_releases_backend(self, sentence_transformer_stub):
        model = NVEmbedModel(NV_CONFIG)
        model.load_model()
        model.unload_model()
        assert model.model is None
        assert model.tokenizer is None
        assert model.is_loaded is False

    def test_encode_requires_a_loaded_model(self):
        with pytest.raises(RuntimeError, match="Model not loaded"):
            NVEmbedModel(NV_CONFIG).encode_texts(["hi"], input_type="query")

    def test_encode_applies_prefixes(self, sentence_transformer_stub):
        model = NVEmbedModel({**NV_CONFIG, "settings": {"query_prefix": "q: "}})
        model.load_model()
        model.encode_texts(["a"], input_type="query")
        assert sentence_transformer_stub.instances[-1].encode_calls[-1]["texts"] == [
            "q: a"
        ]

    def test_encode_rejects_invalid_input_type(self, sentence_transformer_stub):
        model = NVEmbedModel(NV_CONFIG)
        model.load_model()
        with pytest.raises(ValueError, match="Invalid input_type"):
            model.encode_texts(["a"], input_type="document")

    def test_encode_propagates_backend_errors(self, sentence_transformer_stub):
        model = NVEmbedModel(NV_CONFIG)
        model.load_model()

        def boom(*args, **kwargs):
            raise RuntimeError("encode failed")

        model.model.encode = boom
        with pytest.raises(RuntimeError, match="encode failed"):
            model.encode_texts(["a"], input_type="query")

    def test_model_info_advertises_advanced_compression(self):
        info = NVEmbedModel(NV_CONFIG).get_model_info()
        assert info["family_specific"]["supports_advanced_compression"] is True
        assert info["family_specific"]["use_sentence_transformers"] is True

    @pytest.mark.parametrize(
        "embedding_type, expected_dtype",
        [("binary", np.int8), ("ubinary", np.uint8)],
    )
    def test_binary_postprocessing_uses_median_thresholds(
        self, embeddings, embedding_type, expected_dtype
    ):
        model = NVEmbedModel(NV_CONFIG)
        packed = model.postprocess_embeddings(embeddings, embedding_type)
        assert packed.dtype == expected_dtype
        assert packed.shape == (embeddings.shape[0], embeddings.shape[1] // 8)

        expected_bits = embeddings > np.median(embeddings, axis=1, keepdims=True)
        assert np.array_equal(
            np.unpackbits(packed.view(np.uint8), axis=1).astype(bool), expected_bits
        )

    def test_float_postprocessing_delegates_to_base_class(self, embeddings):
        model = NVEmbedModel(NV_CONFIG)
        processed = model.postprocess_embeddings(embeddings, "float")
        assert processed.dtype == np.float32
        assert processed.shape == embeddings.shape


class TestNVEmbedTransformersPath:
    @pytest.fixture
    def model(self, transformers_stubs) -> NVEmbedModel:
        transformers_stubs(nv_module)
        config = {
            **NV_CONFIG,
            "embedding_dimension": 4,
            "settings": {"use_sentence_transformers": False, "query_prefix": "q: "},
        }
        return NVEmbedModel(config)

    def test_load_uses_tokenizer_and_automodel(self, model):
        assert model.load_model() is True
        assert model.tokenizer is not None
        assert model.model.device == "cpu"

    def test_encode_mean_pools_when_no_pooler_output(self, model):
        model.load_model()
        embeddings = model.encode_texts(
            ["a", "b", "c"], input_type="query", batch_size=2
        )
        assert embeddings.shape == (3, 4)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)

    def test_encode_prefers_pooler_output(self, model):
        model.load_model()
        model.model.with_pooler_output = True
        embeddings = model.encode_texts(["a"], input_type="query", normalize=False)
        assert np.array_equal(
            embeddings, np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
        )


class TestSentenceTransformersModel:
    def test_requires_the_library(self, monkeypatch):
        monkeypatch.setattr(st_module, "SENTENCE_TRANSFORMERS_AVAILABLE", False)
        with pytest.raises(
            ImportError, match="sentence-transformers library is required"
        ):
            SentenceTransformersModel(ST_CONFIG)

    def test_load_and_unload(self, sentence_transformer_stub):
        model = SentenceTransformersModel(ST_CONFIG)
        assert model.load_model() is True
        assert sentence_transformer_stub.instances[-1].eval_calls == 1
        model.unload_model()
        assert model.model is None
        assert model.is_loaded is False

    def test_load_failure_is_reported(self, sentence_transformer_stub):
        sentence_transformer_stub.raise_on_init = True
        assert SentenceTransformersModel(ST_CONFIG).load_model() is False

    def test_preprocess_only_strips(self, sentence_transformer_stub):
        model = SentenceTransformersModel(ST_CONFIG)
        assert model.preprocess_text("  hi  ", "query") == "hi"

    def test_encode_requires_a_loaded_model(self, sentence_transformer_stub):
        with pytest.raises(RuntimeError, match="Model not loaded"):
            SentenceTransformersModel(ST_CONFIG).encode_texts(["hi"])

    def test_encode_defaults_to_batch_size_128(self, sentence_transformer_stub):
        model = SentenceTransformersModel(ST_CONFIG)
        model.load_model()
        embeddings = model.encode_texts(["a"], input_type="query")
        call = sentence_transformer_stub.instances[-1].encode_calls[-1]
        assert call["batch_size"] == 128
        assert call["texts"] == ["a"]
        assert embeddings.shape == (1, 4)

    def test_encode_propagates_backend_errors(self, sentence_transformer_stub):
        model = SentenceTransformersModel(ST_CONFIG)
        model.load_model()

        def boom(*args, **kwargs):
            raise RuntimeError("encode failed")

        model.model.encode = boom
        with pytest.raises(RuntimeError, match="encode failed"):
            model.encode_texts(["a"])

    def test_model_info_reports_library_version(self, sentence_transformer_stub):
        import sentence_transformers

        info = SentenceTransformersModel(ST_CONFIG).get_model_info()
        assert info["family_specific"]["sentence_transformers_version"] == (
            sentence_transformers.__version__
        )
