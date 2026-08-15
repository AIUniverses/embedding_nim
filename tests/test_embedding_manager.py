"""Unit tests for src/embedding_manager.py with stubbed model plumbing."""

import base64
import io
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from PIL import Image

from src.embedding_manager import (
    BatchRequest,
    EmbeddingManager,
    EmbeddingRequest,
    RequestPriority,
)


class FakeModel:
    """Model stand-in returning deterministic embeddings without any ML runtime."""

    def __init__(self, dimension: int = 8, modalities: Optional[List[str]] = None):
        self.embedding_dimension = dimension
        self.supported_modalities = modalities or ["text", "image", "text_image"]
        self.supported_embedding_types = ["float", "int8", "binary"]
        self.encode_calls: List[Dict[str, Any]] = []

    def validate_embedding_type(self, embedding_type: str) -> None:
        if embedding_type not in self.supported_embedding_types:
            raise ValueError(f"Embedding type '{embedding_type}' not supported")

    def validate_modality(self, modality: str) -> None:
        if modality not in self.supported_modalities:
            raise ValueError(f"Modality '{modality}' not supported")

    def encode_texts(self, texts, input_type=None, normalize=True, batch_size=None):
        self.encode_calls.append(
            {"texts": list(texts), "input_type": input_type, "normalize": normalize}
        )
        return np.stack(
            [
                np.full(self.embedding_dimension, index + 1, dtype=np.float32)
                for index in range(len(texts))
            ]
        )

    def postprocess_embeddings(
        self, embeddings, embedding_type="float", dimensions=None
    ):
        if dimensions:
            embeddings = embeddings[:, :dimensions]
        if embedding_type == "int8":
            return embeddings.astype(np.int8)
        return embeddings.astype(np.float32)


class FakeModelManager:
    """Model manager stand-in that never loads real weights."""

    def __init__(self, model: FakeModel, input_type: Optional[str] = "query"):
        self.model = model
        self.input_type = input_type
        self.ensure_calls: List[str] = []

    def validate_model_request(self, model_name, input_type=None):
        if model_name.startswith("unknown"):
            raise ValueError(f"Model '{model_name}' not found")
        return model_name, input_type or self.input_type

    async def ensure_model_loaded_async(self, model_name):
        self.ensure_calls.append(model_name)
        return True

    def get_current_model(self):
        return self.model


def make_png_data_url(width: int = 8, height: int = 8) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def model_manager(model) -> FakeModelManager:
    return FakeModelManager(model)


@pytest.fixture
def manager(config_manager, model_manager) -> EmbeddingManager:
    return EmbeddingManager(config_manager, model_manager)


class TestDataclasses:
    def test_embedding_request_defaults(self):
        request = EmbeddingRequest(inputs=["a"], model="m")
        assert request.embedding_type == "float"
        assert request.normalize is True
        assert request.priority is RequestPriority.NORMAL
        assert request.input_type is None
        assert request.metadata is None

    def test_batch_request_holds_requests(self):
        request = EmbeddingRequest(inputs=["a"], model="m")
        batch = BatchRequest(
            requests=[request], created_at=1.0, priority=RequestPriority.HIGH
        )
        assert batch.requests == [request]
        assert batch.priority.value == 3


class TestInitialization:
    def test_reads_configuration(self, manager, config_manager):
        assert manager.max_batch_size == config_manager.get_max_batch_size()
        assert manager.enable_dynamic_batching is False
        assert manager.batch_timeout_ms == config_manager.get_batch_timeout_ms()
        assert manager._request_queue is None
        assert manager.cache is not None
        assert manager.strict_mode is False

    def test_caching_disabled_leaves_cache_unset(
        self, config_manager, model_manager, monkeypatch
    ):
        monkeypatch.setattr(config_manager, "is_caching_enabled", lambda: False)
        assert EmbeddingManager(config_manager, model_manager).cache is None

    def test_cache_initialisation_failure_is_tolerated(
        self, config_manager, model_manager, monkeypatch
    ):
        monkeypatch.setattr(config_manager, "get_cache_backend", lambda: "unsupported")
        assert EmbeddingManager(config_manager, model_manager).cache is None

    def test_strict_mode_from_environment(
        self, monkeypatch, config_manager, model_manager
    ):
        monkeypatch.setenv("STRICT_NIM_MODE", "true")
        assert EmbeddingManager(config_manager, model_manager).strict_mode is True

    def test_dummy_metrics_are_usable_when_monitoring_disabled(self, manager):
        manager.request_counter.labels(model="m", status="success").inc()
        manager.request_duration.labels(model="m").observe(0.1)
        manager.batch_size_histogram.observe(1)
        manager.queue_size_gauge.set(3)

    def test_queue_is_created_when_dynamic_batching_enabled(
        self, config_manager, model_manager, monkeypatch
    ):
        monkeypatch.setattr(config_manager, "is_dynamic_batching_enabled", lambda: True)
        manager = EmbeddingManager(config_manager, model_manager)
        assert manager._request_queue is not None
        assert manager.get_stats()["queue_size"] == 0


class TestGenerateEmbeddings:
    async def test_returns_openai_shaped_response(self, manager, model):
        response = await manager.generate_embeddings(["hello world"], "e5-small-v2")

        assert response["object"] == "list"
        assert response["model"] == "e5-small-v2"
        assert len(response["data"]) == 1
        assert response["data"][0]["index"] == 0
        assert response["data"][0]["object"] == "embedding"
        assert len(response["data"][0]["embedding"]) == model.embedding_dimension
        assert response["usage"]["prompt_tokens"] == 2
        assert response["usage"]["total_tokens"] == 2
        assert response["metadata"]["num_embeddings"] == 1
        assert response["metadata"]["embedding_dimension"] == model.embedding_dimension
        assert response["metadata"]["cache"] is False

    async def test_ensures_model_is_loaded_and_forwards_input_type(
        self, manager, model, model_manager
    ):
        await manager.generate_embeddings(["hi"], "e5-small-v2", input_type="passage")
        assert model_manager.ensure_calls == ["e5-small-v2"]
        assert model.encode_calls[0]["input_type"] == "passage"

    async def test_dimensions_and_embedding_type_are_applied(self, manager):
        response = await manager.generate_embeddings(
            ["hi"], "e5-small-v2", embedding_type="int8", dimensions=4
        )
        assert len(response["data"][0]["embedding"]) == 4
        assert all(isinstance(value, int) for value in response["data"][0]["embedding"])

    async def test_normalize_flag_is_forwarded(self, manager, model):
        await manager.generate_embeddings(["hi"], "e5-small-v2", normalize=False)
        assert model.encode_calls[0]["normalize"] is False

    async def test_multiple_inputs_preserve_order(self, manager):
        response = await manager.generate_embeddings(["a", "b", "c"], "e5-small-v2")
        assert [item["index"] for item in response["data"]] == [0, 1, 2]
        assert response["data"][0]["embedding"][0] == 1.0
        assert response["data"][2]["embedding"][0] == 3.0

    async def test_dict_inputs_are_supported(self, manager):
        response = await manager.generate_embeddings(
            [{"text": "hello there"}], "e5-small-v2", modality="text"
        )
        assert response["usage"]["total_tokens"] == 2

    async def test_unsupported_embedding_type_raises(self, manager):
        with pytest.raises(ValueError, match="not supported"):
            await manager.generate_embeddings(
                ["hi"], "e5-small-v2", embedding_type="uint8"
            )

    async def test_unknown_model_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            await manager.generate_embeddings(["hi"], "unknown-model")

    async def test_too_long_input_raises(self, manager, config_manager):
        long_text = "x" * (config_manager.get_max_input_length() + 1)
        with pytest.raises(ValueError, match="Input 0"):
            await manager.generate_embeddings([long_text], "e5-small-v2")

    async def test_strict_mode_drops_metadata(self, manager):
        manager.strict_mode = True
        response = await manager.generate_embeddings(["hi"], "e5-small-v2")
        assert "metadata" not in response

    async def test_statistics_are_tracked(self, manager):
        await manager.generate_embeddings(["hi"], "e5-small-v2")
        assert manager._processing_stats["total_requests"] == 1
        assert manager._processing_stats["cached_requests"] == 0


class TestCaching:
    async def test_second_identical_request_is_served_from_cache(self, manager, model):
        first = await manager.generate_embeddings(["hi"], "e5-small-v2")
        second = await manager.generate_embeddings(["hi"], "e5-small-v2")

        assert len(model.encode_calls) == 1
        assert second["data"] == first["data"]
        assert second["metadata"]["cache"] is True
        assert manager._processing_stats["cached_requests"] == 1

    async def test_cache_key_includes_request_parameters(self, manager, model):
        await manager.generate_embeddings(["hi"], "e5-small-v2")
        await manager.generate_embeddings(["hi"], "e5-small-v2", dimensions=4)
        assert len(model.encode_calls) == 2

    async def test_strict_mode_returns_cached_payload_untouched(self, manager):
        await manager.generate_embeddings(["hi"], "e5-small-v2")
        manager.strict_mode = True
        cached = await manager.generate_embeddings(["hi"], "e5-small-v2")
        assert cached["metadata"]["cache"] is False


class TestPreprocessInputs:
    def test_detects_modality_and_cleans_text(self, manager):
        processed = manager._preprocess_inputs(["  hello   world  "])
        assert processed == [{"modality": "text", "text": "hello world"}]

    def test_explicit_modality_list_is_used_per_input(self, manager):
        processed = manager._preprocess_inputs(["a b", "c d"], modality=["text"])
        assert [item["modality"] for item in processed] == ["text", "text"]

    def test_image_inputs_are_validated_and_annotated(self, manager):
        processed = manager._preprocess_inputs([make_png_data_url()], modality="image")
        assert processed[0]["modality"] == "image"
        assert processed[0]["image_format"] == "png"
        assert processed[0]["image_size"] > 0
        assert isinstance(processed[0]["image_bytes"], bytes)

    def test_unsupported_modality_is_rejected(self, config_manager, monkeypatch):
        manager = EmbeddingManager(
            config_manager, FakeModelManager(FakeModel(modalities=["text"]))
        )
        with pytest.raises(ValueError, match="Input 0"):
            manager._preprocess_inputs([make_png_data_url()], modality="image")

    @pytest.mark.parametrize("strategy", ["head", "tail", "mid"])
    def test_overlong_text_is_rejected_before_truncation_applies(
        self, manager, config_manager, monkeypatch, strategy
    ):
        monkeypatch.setattr(config_manager, "get_max_input_length", lambda: 4)
        monkeypatch.setattr(config_manager, "get_truncate_strategy", lambda: strategy)
        with pytest.raises(ValueError, match="exceeds maximum 4"):
            manager._preprocess_inputs(["abcdefgh"])

    @pytest.mark.parametrize("strategy", ["head", "tail", "mid"])
    def test_text_within_limit_is_untouched_by_truncation(
        self, manager, config_manager, monkeypatch, strategy
    ):
        monkeypatch.setattr(config_manager, "get_truncate_strategy", lambda: strategy)
        assert manager._preprocess_inputs(["abcd"])[0]["text"] == "abcd"

    def test_empty_text_is_rejected(self, manager):
        with pytest.raises(ValueError, match="Input 0"):
            manager._preprocess_inputs([{"text": "   "}], modality="text")


class TestGenerateEmbeddingsInternal:
    async def test_text_only(self, manager, model):
        processed = [
            {"modality": "text", "text": "a"},
            {"modality": "text", "text": "b"},
        ]
        embeddings = await manager._generate_embeddings_internal(
            processed, model, None, True
        )
        assert embeddings.shape == (2, model.embedding_dimension)

    async def test_image_requires_fake_embeddings_flag(self, manager, model):
        processed = [{"modality": "image", "image_bytes": b"bytes"}]
        with pytest.raises(ValueError, match="ENABLE_FAKE_IMAGE_EMBEDDINGS"):
            await manager._generate_embeddings_internal(processed, model, None, True)

    async def test_fake_image_embeddings_are_deterministic_and_normalized(
        self, manager, model, monkeypatch
    ):
        monkeypatch.setenv("ENABLE_FAKE_IMAGE_EMBEDDINGS", "true")
        processed = [{"modality": "image", "image_bytes": b"bytes"}]
        first = await manager._generate_embeddings_internal(
            processed, model, None, True
        )
        second = await manager._generate_embeddings_internal(
            processed, model, None, True
        )
        assert first.shape == (1, model.embedding_dimension)
        assert np.array_equal(first, second)
        assert np.linalg.norm(first[0]) == pytest.approx(1.0, abs=1e-5)

    async def test_fake_image_embeddings_skip_normalisation_when_requested(
        self, manager, model, monkeypatch
    ):
        monkeypatch.setenv("ENABLE_FAKE_IMAGE_EMBEDDINGS", "true")
        processed = [{"modality": "image", "image_bytes": b"bytes"}]
        unnormalized = await manager._generate_embeddings_internal(
            processed, model, None, False
        )
        assert np.linalg.norm(unnormalized[0]) != pytest.approx(1.0, abs=1e-3)

    async def test_multimodal_input_is_average_fusion(
        self, manager, model, monkeypatch
    ):
        monkeypatch.setenv("ENABLE_FAKE_IMAGE_EMBEDDINGS", "true")
        processed = [{"modality": "text_image", "text": "a", "image_bytes": b"bytes"}]
        fused = await manager._generate_embeddings_internal(
            processed, model, None, True
        )

        text_only = model.encode_texts(["a"])
        image_only = await manager._generate_embeddings_internal(
            [{"modality": "image", "image_bytes": b"bytes"}], model, None, True
        )
        assert np.allclose(fused[0], (text_only[0] + image_only[0]) / 2.0)

    async def test_input_without_usable_content_raises(self, manager, model):
        with pytest.raises(
            ValueError, match="Cannot produce embedding for input index 0"
        ):
            await manager._generate_embeddings_internal(
                [{"modality": "text"}], model, None, True
            )


class TestGenerateWithBatching:
    async def test_splits_texts_into_batches_of_max_size(self, manager, model):
        manager.max_batch_size = 2
        embeddings = await manager._generate_with_batching(
            ["a", "b", "c"], model, None, True
        )
        assert embeddings.shape == (3, model.embedding_dimension)
        assert [call["texts"] for call in model.encode_calls] == [["a", "b"], ["c"]]

    async def test_used_when_dynamic_batching_enabled(self, manager, model):
        manager.enable_dynamic_batching = True
        manager.max_batch_size = 2
        processed = [{"modality": "text", "text": text} for text in ["a", "b", "c"]]
        await manager._generate_embeddings_internal(processed, model, None, True)
        assert len(model.encode_calls) == 2


class TestDynamicBatchProcessor:
    @pytest.fixture
    def batching_manager(self, config_manager, model_manager, monkeypatch):
        monkeypatch.setattr(config_manager, "is_dynamic_batching_enabled", lambda: True)
        return EmbeddingManager(config_manager, model_manager)

    async def test_single_text_requests_are_batched_together(
        self, batching_manager, model
    ):
        import asyncio

        responses = await asyncio.gather(
            batching_manager.generate_embeddings(["a"], "e5-small-v2"),
            batching_manager.generate_embeddings(["b"], "e5-small-v2"),
        )
        await batching_manager.stop_batch_processor()

        assert all(len(response["data"]) == 1 for response in responses)
        assert [response["data"][0]["index"] for response in responses] == [0, 1]
        assert len(model.encode_calls) == 1
        assert model.encode_calls[0]["texts"] == ["a", "b"]

    async def test_batch_processor_lifecycle(self, batching_manager):
        await batching_manager.start_batch_processor()
        assert batching_manager.get_stats()["batch_processor_running"] is True
        await batching_manager.stop_batch_processor()
        assert batching_manager.get_stats()["batch_processor_running"] is False

    async def test_stop_without_start_is_a_noop(self, batching_manager):
        await batching_manager.stop_batch_processor()
        assert batching_manager._batch_processor_task is None

    async def test_batch_failures_propagate_to_callers(self, batching_manager, model):
        def boom(*args, **kwargs):
            raise RuntimeError("encode failed")

        model.encode_texts = boom
        with pytest.raises(RuntimeError, match="encode failed"):
            await batching_manager.generate_embeddings(["a"], "e5-small-v2")
        await batching_manager.stop_batch_processor()

    async def test_empty_batch_is_ignored(self, batching_manager):
        await batching_manager._process_request_batch([])


class TestStats:
    def test_stats_report_batching_configuration(self, manager, config_manager):
        assert manager.get_stats() == {
            "max_batch_size": config_manager.get_max_batch_size(),
            "enable_dynamic_batching": False,
            "batch_timeout_ms": config_manager.get_batch_timeout_ms(),
            "batch_processor_running": False,
            "queue_size": 0,
        }
