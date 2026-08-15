"""Unit tests for src/utils/postprocessing.py."""

import numpy as np
import pytest

from src.utils.postprocessing import EmbeddingPostprocessor


@pytest.fixture
def processor() -> EmbeddingPostprocessor:
    return EmbeddingPostprocessor()


class TestNormalize:
    def test_normalize_gives_unit_norms(self, processor, embeddings):
        normalized = processor.normalize_embeddings(embeddings)
        assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-6)

    def test_normalize_leaves_zero_vectors_untouched(self, processor):
        zeros = np.zeros((2, 3), dtype=np.float32)
        assert np.array_equal(processor.normalize_embeddings(zeros), zeros)


class TestReduceDimensions:
    def test_returns_input_when_target_not_smaller(self, processor, embeddings):
        result = processor.reduce_dimensions(embeddings, embeddings.shape[1])
        assert result is embeddings

    def test_truncate_keeps_leading_dimensions(self, processor, embeddings):
        result = processor.reduce_dimensions(embeddings, 4)
        assert result.shape == (4, 4)
        assert np.array_equal(result, embeddings[:, :4])

    def test_pca_reduces_to_target_dimension(self, processor, embeddings):
        result = processor.reduce_dimensions(embeddings, 2, method="pca")
        assert result.shape == (4, 2)

    def test_random_projection_reduces_to_target_dimension(self, processor, embeddings):
        result = processor.reduce_dimensions(embeddings, 3, method="random")
        assert result.shape == (4, 3)

    def test_rejects_unknown_method(self, processor, embeddings):
        with pytest.raises(ValueError, match="Unknown dimension reduction method"):
            processor.reduce_dimensions(embeddings, 2, method="magic")

    def test_pca_falls_back_to_truncation_without_sklearn(
        self, processor, embeddings, monkeypatch
    ):
        monkeypatch.setitem(__import__("sys").modules, "sklearn.decomposition", None)
        result = processor._pca_reduce(embeddings, 5)
        assert np.array_equal(result, embeddings[:, :5])

    def test_random_projection_falls_back_to_truncation_without_sklearn(
        self, processor, embeddings, monkeypatch
    ):
        monkeypatch.setitem(
            __import__("sys").modules, "sklearn.random_projection", None
        )
        result = processor._random_projection(embeddings, 5)
        assert np.array_equal(result, embeddings[:, :5])


class TestQuantize:
    def test_int8_uses_full_range_and_dtype(self, processor, embeddings):
        quantized = processor.quantize_embeddings(embeddings, "int8")
        assert quantized.dtype == np.int8
        assert np.max(np.abs(quantized)) == 127

    def test_int8_handles_zero_vectors(self, processor):
        quantized = processor.quantize_embeddings(
            np.zeros((2, 4), dtype=np.float32), "int8"
        )
        assert np.array_equal(quantized, np.zeros((2, 4), dtype=np.int8))

    def test_uint8_maps_min_and_max_to_range_bounds(self, processor, embeddings):
        quantized = processor.quantize_embeddings(embeddings, "uint8")
        assert quantized.dtype == np.uint8
        assert np.array_equal(np.min(quantized, axis=1), np.zeros(len(embeddings)))
        assert np.array_equal(np.max(quantized, axis=1), np.full(len(embeddings), 255))

    def test_uint8_handles_constant_vectors(self, processor):
        constant = np.full((2, 4), 0.5, dtype=np.float32)
        assert np.array_equal(
            processor.quantize_embeddings(constant, "uint8"),
            np.zeros((2, 4), dtype=np.uint8),
        )

    @pytest.mark.parametrize(
        "kind, dtype", [("binary", np.int8), ("ubinary", np.uint8)]
    )
    def test_binary_packs_bits(self, processor, embeddings, kind, dtype):
        quantized = processor.quantize_embeddings(embeddings, kind)
        assert quantized.dtype == dtype
        assert quantized.shape == (4, 2)

    def test_float_keeps_values_as_float32(self, processor, embeddings):
        quantized = processor.quantize_embeddings(
            embeddings.astype(np.float64), "float"
        )
        assert quantized.dtype == np.float32
        assert np.allclose(quantized, embeddings)

    def test_rejects_unknown_type(self, processor, embeddings):
        with pytest.raises(ValueError, match="Unknown quantization type"):
            processor.quantize_embeddings(embeddings, "float4")


class TestSimilarities:
    def test_cosine_self_similarity_has_unit_diagonal(self, processor, embeddings):
        similarities = processor.compute_similarities(embeddings)
        assert similarities.shape == (4, 4)
        assert np.allclose(np.diag(similarities), 1.0, atol=1e-6)

    def test_cosine_handles_zero_vectors(self, processor):
        vectors = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        similarities = processor.compute_similarities(vectors)
        assert similarities[0, 0] == pytest.approx(0.0)
        assert similarities[1, 1] == pytest.approx(1.0)

    def test_dot_product_metric(self, processor):
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        assert processor.compute_similarities(a, b, metric="dot")[
            0, 0
        ] == pytest.approx(11.0)

    def test_euclidean_metric_returns_negative_distance(self, processor):
        a = np.array([[0.0, 0.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        assert processor.compute_similarities(a, b, metric="euclidean")[
            0, 0
        ] == pytest.approx(-5.0)

    def test_rejects_unknown_metric(self, processor, embeddings):
        with pytest.raises(ValueError, match="Unknown similarity metric"):
            processor.compute_similarities(embeddings, metric="jaccard")


class TestAggregate:
    def test_single_embedding_is_returned_unchanged(self, processor, embeddings):
        assert processor.aggregate_embeddings([embeddings]) is embeddings

    def test_rejects_empty_list(self, processor):
        with pytest.raises(ValueError, match="Empty embeddings list"):
            processor.aggregate_embeddings([])

    @pytest.mark.parametrize(
        "method, expected",
        [("mean", [2.0, 3.0]), ("max", [3.0, 4.0]), ("sum", [4.0, 6.0])],
    )
    def test_aggregation_methods(self, processor, method, expected):
        parts = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        assert np.allclose(
            processor.aggregate_embeddings(parts, method=method), expected
        )

    def test_rejects_unknown_method(self, processor):
        parts = [np.array([1.0]), np.array([2.0])]
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            processor.aggregate_embeddings(parts, method="median")


class TestFilterEmbeddings:
    def test_filters_by_min_and_max_length(self, processor):
        vectors = np.arange(8, dtype=np.float32).reshape(4, 2)
        texts = ["hi", "hello there", "medium text", "  this one is far too long  "]
        filtered, kept = processor.filter_embeddings(
            vectors, texts, min_length=5, max_length=11
        )
        assert kept == ["hello there", "medium text"]
        assert np.array_equal(filtered, vectors[1:3])

    def test_removes_duplicates_keeping_first(self, processor):
        vectors = np.arange(6, dtype=np.float32).reshape(3, 2)
        texts = ["duplicate", "duplicate", "unique text"]
        filtered, kept = processor.filter_embeddings(
            vectors, texts, min_length=1, remove_duplicates=True
        )
        assert kept == ["duplicate", "unique text"]
        assert np.array_equal(filtered, vectors[[0, 2]])

    def test_rejects_length_mismatch(self, processor):
        with pytest.raises(ValueError, match="same length"):
            processor.filter_embeddings(np.zeros((2, 2)), ["only one"])


class TestValidateEmbeddings:
    def test_reports_valid_embeddings(self, processor):
        vectors = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        results = processor.validate_embeddings(vectors)
        assert results["is_valid"] is True
        assert results["issues"] == []
        assert results["shape"] == (2, 2)
        assert results["mean_norm"] == pytest.approx(5.0)

    def test_flags_nan_and_inf(self, processor):
        vectors = np.array([[np.nan, 1.0], [np.inf, 1.0]], dtype=np.float32)
        results = processor.validate_embeddings(vectors)
        assert results["is_valid"] is False
        assert "Contains NaN values" in results["issues"]
        assert "Contains infinite values" in results["issues"]

    def test_flags_small_norms(self, processor):
        results = processor.validate_embeddings(np.full((2, 2), 1e-4, dtype=np.float32))
        assert "Very small embedding norms" in results["issues"]

    def test_flags_high_norm_variance(self, processor):
        vectors = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
        assert (
            "High variance in embedding norms"
            in processor.validate_embeddings(vectors)["issues"]
        )
