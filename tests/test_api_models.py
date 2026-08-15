"""Unit tests for src/api/models.py."""

import pytest
from pydantic import ValidationError

from src.api.models import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    create_error_response,
    validate_batch_size,
    validate_input_length,
    validate_model_name,
)


class TestEmbeddingRequest:
    def test_defaults(self):
        request = EmbeddingRequest(input="hello", model="e5-small-v2")
        assert request.embedding_type == "float"
        assert request.normalize is True
        assert request.input_type is None
        assert request.modality is None
        assert request.dimensions is None

    def test_accepts_list_input_and_modality_list(self):
        request = EmbeddingRequest(
            input=["a", "b"],
            model="m",
            modality=["text", "image"],
            input_type="query",
            dimensions=128,
        )
        assert request.input == ["a", "b"]
        assert request.modality == ["text", "image"]

    @pytest.mark.parametrize("value", ["", "   "])
    def test_rejects_blank_string_input(self, value):
        with pytest.raises(ValidationError, match="Input text cannot be empty"):
            EmbeddingRequest(input=value, model="m")

    def test_rejects_empty_list_input(self):
        with pytest.raises(ValidationError, match="Input list cannot be empty"):
            EmbeddingRequest(input=[], model="m")

    def test_rejects_blank_list_item(self):
        with pytest.raises(ValidationError, match=r"Input\[1\] cannot be empty"):
            EmbeddingRequest(input=["ok", " "], model="m")

    def test_rejects_non_string_list_item(self):
        with pytest.raises(ValidationError):
            EmbeddingRequest(input=["ok", {"nested": 1}], model="m")

    def test_rejects_unknown_modality(self):
        with pytest.raises(ValidationError, match="Invalid modality 'audio'"):
            EmbeddingRequest(input="a", model="m", modality="audio")

    def test_rejects_unknown_modality_in_list(self):
        with pytest.raises(ValidationError, match="Invalid modality 'audio'"):
            EmbeddingRequest(input="a", model="m", modality=["text", "audio"])

    def test_rejects_unknown_input_type(self):
        with pytest.raises(ValidationError):
            EmbeddingRequest(input="a", model="m", input_type="document")

    def test_rejects_unknown_embedding_type(self):
        with pytest.raises(ValidationError):
            EmbeddingRequest(input="a", model="m", embedding_type="float4")

    @pytest.mark.parametrize("dimensions", [0, -8])
    def test_rejects_non_positive_dimensions(self, dimensions):
        with pytest.raises(ValidationError):
            EmbeddingRequest(input="a", model="m", dimensions=dimensions)

    def test_requires_model(self):
        with pytest.raises(ValidationError):
            EmbeddingRequest(input="a")


class TestResponseModels:
    def test_embedding_response_defaults(self):
        response = EmbeddingResponse(
            data=[EmbeddingObject(index=0, embedding=[0.1, 0.2])],
            model="m",
            usage=EmbeddingUsage(prompt_tokens=3, total_tokens=3),
        )
        assert response.object == "list"
        assert response.data[0].object == "embedding"

    def test_embedding_object_accepts_integer_vectors(self):
        assert EmbeddingObject(index=1, embedding=[1, -2, 3]).embedding == [1, -2, 3]

    def test_models_response_defaults(self):
        response = ModelsResponse(data=[ModelInfo(id="m")])
        assert response.object == "list"
        assert response.data[0].object == "model"
        assert response.data[0].created == 0
        assert response.data[0].owned_by == "organization-owner"

    def test_health_response_defaults(self):
        health = HealthResponse(message="ok")
        assert health.object == "health-response"
        assert health.status is None

    def test_error_detail_optional_fields(self):
        detail = ErrorDetail(message="bad", type="invalid_request_error")
        assert detail.param is None
        assert detail.code is None

    def test_error_response_requires_error(self):
        with pytest.raises(ValidationError):
            ErrorResponse()


class TestValidateModelName:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("e5-large-v2", ("e5-large-v2", None)),
            ("e5-large-v2-query", ("e5-large-v2", "query")),
            ("e5-large-v2-passage", ("e5-large-v2", "passage")),
        ],
    )
    def test_parses_input_type_suffix(self, name, expected):
        assert validate_model_name(name) == expected

    @pytest.mark.parametrize("name", ["", None, 123])
    def test_rejects_invalid_names(self, name):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_model_name(name)


class TestValidateInputLength:
    def test_accepts_within_limit(self):
        validate_input_length(["ab", "cd"], max_length=4)

    def test_counts_string_input(self):
        validate_input_length("abcd", max_length=4)

    def test_rejects_over_limit(self):
        with pytest.raises(ValueError, match="exceeds maximum 3"):
            validate_input_length(["ab", "cd"], max_length=3)


class TestValidateBatchSize:
    def test_string_counts_as_one(self):
        validate_batch_size("a", max_batch_size=1)

    def test_accepts_within_limit(self):
        validate_batch_size(["a", "b"], max_batch_size=2)

    def test_rejects_over_limit(self):
        with pytest.raises(ValueError, match="Batch size 3 exceeds maximum 2"):
            validate_batch_size(["a", "b", "c"], max_batch_size=2)


class TestCreateErrorResponse:
    def test_defaults(self):
        response = create_error_response("boom")
        assert isinstance(response, ErrorResponse)
        assert response.error == {
            "message": "boom",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }

    def test_all_fields(self):
        response = create_error_response(
            "boom", error_type="model_error", param="model", code="404"
        )
        assert response.error["type"] == "model_error"
        assert response.error["param"] == "model"
        assert response.error["code"] == "404"
