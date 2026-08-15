"""Unit tests for src/model_manager.py with stubbed model classes."""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from src.model_manager import ModelManager
from src.models.base_model import BaseEmbeddingModel


class StubModel(BaseEmbeddingModel):
    """Model implementation that records lifecycle calls instead of loading weights."""

    load_succeeds = True
    unload_raises = False

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.load_calls = 0
        self.unload_calls = 0

    def load_model(self) -> bool:
        self.load_calls += 1
        self.is_loaded = self.load_succeeds
        return self.load_succeeds

    def unload_model(self) -> None:
        self.unload_calls += 1
        if self.unload_raises:
            raise RuntimeError("unload failed")
        self.is_loaded = False

    def encode_texts(
        self,
        texts: List[str],
        input_type: Optional[str] = None,
        normalize: bool = True,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        return np.zeros((len(texts), self.embedding_dimension), dtype=np.float32)


class FailingStubModel(StubModel):
    load_succeeds = False


@pytest.fixture
def manager(config_manager, monkeypatch) -> ModelManager:
    manager = ModelManager(config_manager)
    monkeypatch.setattr(
        manager, "MODEL_FAMILIES", {"e5": StubModel, "gte": StubModel}, raising=False
    )
    return manager


class TestInitialization:
    def test_starts_without_a_loaded_model(self, manager):
        assert manager.get_current_model() is None
        assert manager.get_current_model_name() is None
        assert manager.is_model_loaded() is False
        assert manager.get_loaded_models() == []
        assert manager.device in {"cpu", "cuda"}

    def test_model_families_are_exposed_as_a_copy(self, config_manager):
        manager = ModelManager(config_manager)
        families = manager.list_model_families()
        families.pop("e5")
        assert "e5" in manager.MODEL_FAMILIES
        assert set(manager.MODEL_FAMILIES) == {
            "e5",
            "gte",
            "sentence_transformers",
            "nv_embed",
        }


class TestLoading:
    async def test_loads_model_and_tracks_state(self, manager):
        assert await manager.load_model_async("e5-small-v2") is True
        assert manager.get_current_model_name() == "e5-small-v2"
        assert isinstance(manager.get_current_model(), StubModel)
        assert manager.is_model_loaded("e5-small-v2") is True
        assert manager.get_loaded_models() == ["e5-small-v2"]

    async def test_strips_input_type_suffix(self, manager):
        assert await manager.load_model_async("e5-small-v2-query") is True
        assert manager.get_current_model_name() == "e5-small-v2"

    async def test_reloading_same_model_is_a_noop(self, manager):
        await manager.load_model_async("e5-small-v2")
        model = manager.get_current_model()
        assert await manager.load_model_async("e5-small-v2") is True
        assert manager.get_current_model() is model
        assert model.load_calls == 1

    async def test_switching_models_unloads_previous(self, manager):
        await manager.load_model_async("e5-small-v2")
        first = manager.get_current_model()
        assert await manager.load_model_async("gte-base") is True
        assert first.unload_calls == 1
        assert first.is_loaded is False
        assert manager.get_current_model_name() == "gte-base"

    async def test_unknown_model_returns_false(self, manager):
        assert await manager.load_model_async("does-not-exist") is False
        assert manager.get_current_model() is None

    async def test_unsupported_family_returns_false(self, manager, monkeypatch):
        monkeypatch.setattr(
            manager, "MODEL_FAMILIES", {"gte": StubModel}, raising=False
        )
        assert await manager.load_model_async("e5-small-v2") is False
        assert manager.get_current_model_name() is None

    async def test_failed_load_clears_state(self, manager, monkeypatch):
        monkeypatch.setattr(
            manager, "MODEL_FAMILIES", {"e5": FailingStubModel}, raising=False
        )
        assert await manager.load_model_async("e5-small-v2") is False
        assert manager.get_current_model() is None
        assert manager.get_current_model_name() is None

    async def test_ensure_model_loaded_async_loads_once(self, manager):
        assert await manager.ensure_model_loaded_async("e5-small-v2-passage") is True
        model = manager.get_current_model()
        assert await manager.ensure_model_loaded_async("e5-small-v2") is True
        assert model.load_calls == 1


class TestUnloading:
    async def test_unload_clears_state(self, manager):
        await manager.load_model_async("e5-small-v2")
        model = manager.get_current_model()
        manager.unload_model()
        assert model.unload_calls == 1
        assert manager.get_current_model() is None
        assert manager.get_current_model_name() is None

    def test_unload_without_model_is_a_noop(self, manager):
        manager.unload_model()
        assert manager.get_current_model() is None

    async def test_unload_errors_still_clear_state(self, manager, monkeypatch):
        await manager.load_model_async("e5-small-v2")
        monkeypatch.setattr(manager.get_current_model(), "unload_raises", True)
        manager.unload_model()
        assert manager.get_current_model() is None

    async def test_del_unloads_current_model(self, manager):
        await manager.load_model_async("e5-small-v2")
        model = manager.get_current_model()
        manager.__del__()
        assert model.unload_calls == 1


class TestIsModelLoaded:
    async def test_returns_false_for_other_models(self, manager):
        await manager.load_model_async("e5-small-v2")
        assert manager.is_model_loaded("gte-base") is False
        assert manager.is_model_loaded("e5-small-v2-query") is True

    async def test_returns_false_when_model_reports_unloaded(self, manager):
        await manager.load_model_async("e5-small-v2")
        manager.get_current_model().is_loaded = False
        assert manager.is_model_loaded() is False


class TestGetModelInfo:
    def test_reports_missing_current_model(self, manager):
        info = manager.get_model_info()
        assert info["status"] == "No model loaded"
        assert info["available_models"] == ["e5-small-v2", "gte-base"]

    async def test_current_model_info_comes_from_the_model(self, manager):
        await manager.load_model_async("e5-small-v2")
        assert manager.get_model_info() == manager.get_current_model().get_model_info()

    def test_named_model_info_is_read_from_config(self, manager):
        info = manager.get_model_info("e5-small-v2")
        assert info["model_id"] == "intfloat/e5-small-v2"
        assert info["display_name"] == "E5 Small V2"
        assert info["family"] == "e5"
        assert info["is_loaded"] is False
        assert info["max_seq_length"] == 512
        assert info["embedding_dimension"] == 384
        assert info["supports_input_type"] is True
        assert "memory_usage" not in info

    def test_named_model_info_defaults(self, manager):
        info = manager.get_model_info("gte-base")
        assert info["display_name"] == "gte-base"
        assert info["max_seq_length"] == 512
        assert info["supports_input_type"] is False
        assert info["supported_embedding_types"] == ["float"]

    async def test_named_model_info_includes_memory_when_loaded(self, manager):
        await manager.load_model_async("e5-small-v2")
        info = manager.get_model_info("e5-small-v2-query")
        assert info["is_loaded"] is True
        assert info["memory_usage"] == manager.get_current_model().get_memory_usage()

    def test_unknown_model_reports_error(self, manager):
        info = manager.get_model_info("nope")
        assert info["error"] == "Model nope not found"
        assert info["available_models"] == ["e5-small-v2", "gte-base"]


class TestValidateModelRequest:
    def test_uses_explicit_input_type(self, manager):
        assert manager.validate_model_request("e5-small-v2", "query") == (
            "e5-small-v2",
            "query",
        )

    def test_uses_suffix_input_type(self, manager):
        assert manager.validate_model_request("e5-small-v2-passage") == (
            "e5-small-v2",
            "passage",
        )

    def test_explicit_input_type_wins_over_suffix(self, manager):
        assert manager.validate_model_request("e5-small-v2-passage", "query") == (
            "e5-small-v2",
            "query",
        )

    def test_requires_input_type_for_supporting_models(self, manager):
        with pytest.raises(ValueError, match="requires input_type"):
            manager.validate_model_request("e5-small-v2")

    def test_rejects_invalid_input_type(self, manager):
        with pytest.raises(ValueError, match="Invalid input_type 'document'"):
            manager.validate_model_request("e5-small-v2", "document")

    def test_ignores_input_type_for_models_without_support(self, manager):
        assert manager.validate_model_request("gte-base", "query") == ("gte-base", None)

    def test_rejects_unknown_model(self, manager):
        with pytest.raises(ValueError, match="Model 'nope' not found"):
            manager.validate_model_request("nope")


class TestAvailableModelsAndGpuInfo:
    def test_available_models_delegate_to_config(self, manager, config_manager):
        assert manager.get_available_models() == config_manager.get_available_models()

    def test_gpu_memory_info_has_expected_keys(self, manager):
        info = manager.get_gpu_memory_info()
        assert set(info) == {"total", "allocated", "reserved", "free"}
        assert all(isinstance(value, float) for value in info.values())
