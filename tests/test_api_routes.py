"""Unit tests for the API routers with stubbed service components."""

from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import embeddings, health, models


class StubModelManager:
    """Model manager stand-in recording calls instead of loading weights."""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.loaded: Optional[str] = None
        self.load_succeeds = True
        self.unload_calls = 0

    async def load_model_async(self, model_name: str) -> bool:
        if self.load_succeeds:
            self.loaded = model_name
        return self.load_succeeds

    def unload_model(self) -> None:
        self.unload_calls += 1
        self.loaded = None

    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        if self.loaded is None:
            return False
        return model_name is None or self.loaded == model_name

    def get_current_model_name(self) -> Optional[str]:
        return self.loaded

    def get_loaded_models(self) -> List[str]:
        return [self.loaded] if self.loaded else []

    def get_gpu_memory_info(self) -> Dict[str, float]:
        return {"total": 0.0, "allocated": 0.0, "reserved": 0.0, "free": 0.0}

    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        if model_name is None:
            return {"status": "No model loaded", "available_models": []}
        if model_name not in self.config_manager.get_available_models():
            return {"error": f"Model {model_name} not found", "available_models": []}
        info = {
            "model_id": f"org/{model_name}",
            "display_name": self.config_manager.get_available_models()[model_name],
            "family": self.config_manager.get_model_family(model_name),
            "is_loaded": self.is_model_loaded(model_name),
            "max_seq_length": self.config_manager.get_max_seq_length(model_name),
            "embedding_dimension": self.config_manager.get_embedding_dimension(
                model_name
            ),
            "supports_input_type": self.config_manager.supports_input_type(model_name),
            "supported_embedding_types": self.config_manager.get_supported_embedding_types(
                model_name
            ),
            "supported_modalities": self.config_manager.get_supported_modalities(
                model_name
            ),
        }
        if info["is_loaded"]:
            info["memory_usage"] = {"allocated": 0.0}
        return info


class StubEmbeddingManager:
    """Embedding manager stand-in returning canned responses."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.exception: Optional[Exception] = None
        self.cache = None
        self._processing_stats = {"total_requests": 0, "cached_requests": 0}

    async def generate_embeddings(self, inputs, model, **kwargs):
        self.calls.append({"inputs": inputs, "model": model, **kwargs})
        if self.exception:
            raise self.exception
        return {
            "object": "list",
            "data": [
                {"index": index, "embedding": [0.1, 0.2], "object": "embedding"}
                for index in range(len(inputs))
            ],
            "model": model,
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            "metadata": {"cache": False},
        }

    def get_stats(self) -> Dict[str, Any]:
        return {"enable_dynamic_batching": True}


@pytest.fixture
def model_manager(config_manager) -> StubModelManager:
    return StubModelManager(config_manager)


@pytest.fixture
def embedding_manager() -> StubEmbeddingManager:
    return StubEmbeddingManager()


@pytest.fixture
def client(config_manager, model_manager, embedding_manager) -> TestClient:
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(embeddings.router)
    app.state.config_manager = config_manager
    app.state.model_manager = model_manager
    app.state.embedding_manager = embedding_manager
    app.state.service_ready = True
    return TestClient(app)


@pytest.fixture
def bare_client() -> TestClient:
    """Client for an app whose service components were never initialised."""
    app = FastAPI()
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(embeddings.router)
    return TestClient(app, raise_server_exceptions=False)


class TestHealthRoutes:
    def test_simple_liveness(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {
            "object": "health-response",
            "message": "Service is alive.",
            "status": "alive",
        }

    def test_liveness(self, client):
        assert client.get("/v1/health/live").json()["status"] == "alive"

    def test_readiness_when_ready(self, client):
        body = client.get("/v1/health/ready").json()
        assert body["status"] == "ready"
        assert body["message"] == "Service is ready."

    def test_readiness_without_embedding_manager(self, bare_client):
        body = bare_client.get("/v1/health/ready").json()
        assert body["status"] == "not_ready"
        assert "embedding manager" in body["message"]

    def test_readiness_without_model_manager(self, client, embedding_manager):
        client.app.state.model_manager = None
        body = client.get("/v1/health/ready").json()
        assert body["status"] == "not_ready"
        assert "model manager" in body["message"]

    def test_readiness_while_initialising(self, client):
        client.app.state.service_ready = False
        body = client.get("/v1/health/ready").json()
        assert body["status"] == "not_ready"
        assert "initializing" in body["message"]

    def test_startup_when_ready(self, client):
        assert client.get("/v1/health/startup").json()["status"] == "ready"

    def test_startup_without_managers(self, bare_client):
        body = bare_client.get("/v1/health/startup").json()
        assert body["status"] == "starting"

    def test_startup_while_loading_default_model(self, client):
        client.app.state.service_ready = False
        body = client.get("/v1/health/startup").json()
        assert body["status"] == "starting"
        assert "loading default model" in body["message"]

    def test_basic_health_summarises_service_state(self, client, model_manager):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert "2 models configured" in body["message"]
        assert "no model loaded" in body["message"]
        assert "dynamic batching enabled" in body["message"]

    def test_basic_health_reports_loaded_model(self, client, model_manager):
        model_manager.loaded = "e5-small-v2"
        assert "model 'e5-small-v2' loaded" in client.get("/health").json()["message"]

    def test_basic_health_handles_errors(self, client, model_manager):
        def boom():
            raise RuntimeError("gpu probe failed")

        model_manager.get_gpu_memory_info = boom
        body = client.get("/health").json()
        assert body["status"] == "unhealthy"
        assert "gpu probe failed" in body["message"]


class TestModelRoutes:
    def test_list_models(self, client):
        body = client.get("/v1/models").json()
        assert body["object"] == "list"
        assert [item["id"] for item in body["data"]] == ["e5-small-v2", "gte-base"]
        assert body["data"][0]["owned_by"] == "organization-owner"

    def test_list_models_without_config_manager(self, bare_client):
        response = bare_client.get("/v1/models")
        assert response.status_code == 500

    def test_get_model_info(self, client):
        body = client.get("/v1/models/e5-small-v2").json()
        assert body["id"] == "e5-small-v2"
        assert body["details"]["family"] == "e5"
        assert body["details"]["embedding_dimension"] == 384
        assert body["details"]["is_loaded"] is False

    def test_get_model_info_strips_suffix(self, client):
        body = client.get("/v1/models/e5-small-v2-query").json()
        assert body["id"] == "e5-small-v2-query"
        assert body["details"]["family"] == "e5"

    def test_get_model_info_unknown_model(self, client):
        response = client.get("/v1/models/nope")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_load_model(self, client, model_manager):
        response = client.post("/v1/models/e5-small-v2/load")
        assert response.status_code == 200
        assert response.json()["status"] == "loaded"
        assert model_manager.loaded == "e5-small-v2"

    def test_load_unknown_model(self, client):
        response = client.post("/v1/models/nope/load")
        assert response.status_code == 404

    def test_load_failure_is_reported(self, client, model_manager):
        model_manager.load_succeeds = False
        response = client.post("/v1/models/e5-small-v2/load")
        assert response.status_code == 500
        assert "Failed to load model" in response.json()["detail"]

    def test_unload_loaded_model(self, client, model_manager):
        model_manager.loaded = "e5-small-v2"
        body = client.post("/v1/models/e5-small-v2/unload").json()
        assert body["status"] == "unloaded"
        assert model_manager.unload_calls == 1

    def test_unload_when_not_loaded(self, client, model_manager):
        body = client.post("/v1/models/e5-small-v2/unload").json()
        assert body["status"] == "not_loaded"
        assert model_manager.unload_calls == 0

    def test_model_status_when_not_loaded(self, client):
        body = client.get("/v1/models/e5-small-v2/status").json()
        assert body == {
            "model_id": "e5-small-v2",
            "base_model_name": "e5-small-v2",
            "is_loaded": False,
            "is_current": False,
            "current_model": None,
        }

    def test_model_status_when_loaded(self, client, model_manager):
        model_manager.loaded = "e5-small-v2"
        body = client.get("/v1/models/e5-small-v2-passage/status").json()
        assert body["is_loaded"] is True
        assert body["is_current"] is True
        assert body["memory_usage"] == {"allocated": 0.0}

    def test_model_status_without_model_manager(self, bare_client):
        assert bare_client.get("/v1/models/e5-small-v2/status").status_code == 500


class TestEmbeddingRoutes:
    def test_create_embeddings(self, client, embedding_manager):
        response = client.post(
            "/v1/embeddings",
            json={"input": "hello", "model": "e5-small-v2", "input_type": "query"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "e5-small-v2"
        assert len(body["data"]) == 1
        assert embedding_manager.calls[0]["inputs"] == ["hello"]
        assert embedding_manager.calls[0]["input_type"] == "query"

    def test_alternative_endpoint(self, client):
        response = client.post(
            "/embeddings",
            json={"input": ["a", "b"], "model": "e5-small-v2-passage"},
        )
        assert response.status_code == 200
        assert len(response.json()["data"]) == 2

    def test_model_suffix_provides_input_type(self, client, embedding_manager):
        client.post(
            "/v1/embeddings", json={"input": "hi", "model": "e5-small-v2-passage"}
        )
        assert embedding_manager.calls[0]["input_type"] == "passage"

    def test_missing_input_type_is_rejected(self, client):
        response = client.post(
            "/v1/embeddings", json={"input": "hi", "model": "e5-small-v2"}
        )
        assert response.status_code == 400
        assert "requires input_type" in response.json()["detail"]

    def test_unknown_model_is_rejected(self, client):
        response = client.post("/v1/embeddings", json={"input": "hi", "model": "nope"})
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]

    def test_unsupported_embedding_type_is_rejected(self, client):
        response = client.post(
            "/v1/embeddings",
            json={
                "input": "hi",
                "model": "e5-small-v2-query",
                "embedding_type": "uint8",
            },
        )
        assert response.status_code == 400
        assert "not supported by model" in response.json()["detail"]

    def test_unsupported_dimensions_are_rejected(self, client):
        response = client.post(
            "/v1/embeddings",
            json={"input": "hi", "model": "e5-small-v2-query", "dimensions": 999},
        )
        assert response.status_code == 400
        assert "Dimensions 999 not supported" in response.json()["detail"]

    def test_unsupported_modality_is_rejected(self, client):
        response = client.post(
            "/v1/embeddings",
            json={"input": "hi", "model": "e5-small-v2-query", "modality": "image"},
        )
        assert response.status_code == 400
        assert "Modality 'image' not supported" in response.json()["detail"]

    def test_dimensions_cannot_be_combined_with_compression(self, client):
        response = client.post(
            "/v1/embeddings",
            json={
                "input": "hi",
                "model": "e5-small-v2-query",
                "dimensions": 128,
                "embedding_type": "int8",
            },
        )
        assert response.status_code == 400
        assert "cannot be used with compressed" in response.json()["detail"]

    def test_batch_size_limit(self, client, config_manager):
        inputs = ["text"] * (config_manager.get_max_batch_size() + 1)
        response = client.post(
            "/v1/embeddings", json={"input": inputs, "model": "e5-small-v2-query"}
        )
        assert response.status_code == 400
        assert "Batch size" in response.json()["detail"]

    def test_input_length_limit(self, client, config_manager):
        long_text = "x" * (config_manager.get_max_input_length() + 1)
        response = client.post(
            "/v1/embeddings", json={"input": long_text, "model": "e5-small-v2-query"}
        )
        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"]

    def test_invalid_payload_is_rejected_by_schema(self, client):
        assert (
            client.post("/v1/embeddings", json={"input": "", "model": "m"}).status_code
            == 422
        )

    def test_value_error_maps_to_400(self, client, embedding_manager):
        embedding_manager.exception = ValueError("bad input")
        response = client.post(
            "/v1/embeddings", json={"input": "hi", "model": "e5-small-v2-query"}
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "bad input"

    def test_runtime_error_maps_to_500(self, client, embedding_manager):
        embedding_manager.exception = RuntimeError("model exploded")
        response = client.post(
            "/v1/embeddings", json={"input": "hi", "model": "e5-small-v2-query"}
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "model exploded"

    def test_unexpected_error_maps_to_500(self, client, embedding_manager):
        embedding_manager.exception = KeyError("boom")
        response = client.post(
            "/v1/embeddings", json={"input": "hi", "model": "e5-small-v2-query"}
        )
        assert response.status_code == 500
        assert "Internal error" in response.json()["detail"]

    def test_uninitialised_service_returns_500(self, bare_client):
        response = bare_client.post(
            "/v1/embeddings", json={"input": "hi", "model": "e5-small-v2-query"}
        )
        assert response.status_code == 500


class TestBatchEmbeddingRoute:
    def test_processes_each_request(self, client, embedding_manager):
        response = client.post(
            "/v1/embeddings/batch",
            json=[
                {"input": "a", "model": "e5-small-v2-query"},
                {"input": ["b", "c"], "model": "e5-small-v2-passage"},
            ],
        )
        assert response.status_code == 200
        body = response.json()
        assert [len(item["data"]) for item in body] == [1, 2]
        assert len(embedding_manager.calls) == 2

    def test_rejects_oversized_batches(self, client, config_manager):
        payload = [
            {"input": "a", "model": "e5-small-v2-query"}
            for _ in range(config_manager.get_max_batch_size() + 1)
        ]
        response = client.post("/v1/embeddings/batch", json=payload)
        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"]

    def test_per_request_failures_do_not_fail_the_batch(
        self, client, embedding_manager
    ):
        embedding_manager.exception = ValueError("bad input")
        response = client.post(
            "/v1/embeddings/batch", json=[{"input": "a", "model": "e5-small-v2-query"}]
        )
        assert response.status_code == 200
        assert response.json()[0]["object"] == "error"
        assert response.json()[0]["data"] == []

    def test_uninitialised_service_returns_500(self, bare_client):
        response = bare_client.post(
            "/v1/embeddings/batch", json=[{"input": "a", "model": "e5-small-v2-query"}]
        )
        assert response.status_code == 500


class TestEmbeddingMetricsRoute:
    def test_returns_processing_statistics(self, client, embedding_manager):
        body = client.get("/v1/embeddings/metrics").json()
        assert body["service_status"] == "healthy"
        assert body["cache_statistics"] == {}
        assert body["processing_statistics"] == embedding_manager._processing_stats

    def test_includes_cache_statistics_when_available(self, client, embedding_manager):
        class StubCache:
            def get_stats(self):
                return {"hit_count": 3}

        embedding_manager.cache = StubCache()
        assert client.get("/v1/embeddings/metrics").json()["cache_statistics"] == {
            "hit_count": 3
        }

    def test_uninitialised_service_returns_500(self, bare_client):
        assert bare_client.get("/v1/embeddings/metrics").status_code == 500


class TestEmbeddingModelsRoute:
    def test_lists_detailed_model_information(self, client):
        body = client.get("/v1/embeddings/models").json()
        assert body["object"] == "list"
        assert body["total"] == 2
        first = body["data"][0]
        assert first["id"] == "e5-small-v2"
        assert first["family"] == "e5"
        assert first["supports_dimensions"] == [128, 256, 384]

    def test_skips_models_that_fail_to_report(self, client, model_manager):
        original = model_manager.get_model_info

        def flaky(model_name=None):
            if model_name == "gte-base":
                raise RuntimeError("info unavailable")
            return original(model_name)

        model_manager.get_model_info = flaky
        body = client.get("/v1/embeddings/models").json()
        assert body["total"] == 1
        assert body["data"][0]["id"] == "e5-small-v2"

    def test_uninitialised_service_returns_500(self, bare_client):
        assert bare_client.get("/v1/embeddings/models").status_code == 500


class TestStreamingEmbeddings:
    def test_streams_embeddings_for_valid_requests(self, client, embedding_manager):
        with client.websocket_connect("/v1/embeddings/stream") as websocket:
            websocket.send_json({"input": "hello", "model": "e5-small-v2-query"})
            assert websocket.receive_json()["status"] == "processing"
            result = websocket.receive_json()
            assert result["status"] == "completed"
            assert len(result["data"]) == 1
            assert "request_id" in result

    def test_missing_fields_are_reported(self, client):
        with client.websocket_connect("/v1/embeddings/stream") as websocket:
            websocket.send_json({"input": "hello"})
            error = websocket.receive_json()
            assert "Missing required fields" in error["error"]

    def test_generation_errors_are_reported(self, client, embedding_manager):
        embedding_manager.exception = ValueError("bad input")
        with client.websocket_connect("/v1/embeddings/stream") as websocket:
            websocket.send_json({"input": "hello", "model": "e5-small-v2-query"})
            assert websocket.receive_json()["status"] == "processing"
            error = websocket.receive_json()
            assert error["status"] == "error"
            assert error["error"] == "bad input"

    def test_uninitialised_service_is_reported(self, bare_client):
        with bare_client.websocket_connect("/v1/embeddings/stream") as websocket:
            assert websocket.receive_json() == {
                "error": "Service not properly initialized"
            }
