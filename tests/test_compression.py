"""Unit tests for src/utils/compression.py."""

import numpy as np
import pytest

from src.utils.compression import EmbeddingCompressor


@pytest.fixture
def compressor() -> EmbeddingCompressor:
    return EmbeddingCompressor()


class TestRoundTrip:
    @pytest.mark.parametrize("compression_type", ["int8", "uint8", "float16"])
    def test_lossy_round_trip_stays_close_to_original(
        self, compressor, embeddings, compression_type
    ):
        compressed = compressor.compress_embeddings(embeddings, compression_type)
        restored = compressor.decompress_embeddings(compressed)
        assert restored.shape == embeddings.shape
        assert compressed["original_shape"] == embeddings.shape
        assert np.allclose(restored, embeddings, atol=0.05)

    @pytest.mark.parametrize(
        "compression_type, dtype", [("binary", np.int8), ("ubinary", np.uint8)]
    )
    def test_binary_round_trip_preserves_sign_relative_to_threshold(
        self, compressor, embeddings, compression_type, dtype
    ):
        compressed = compressor.compress_embeddings(embeddings, compression_type)
        assert compressed["compression_type"] == compression_type
        assert compressed["data"].dtype == dtype
        restored = compressor.decompress_embeddings(compressed)
        thresholds = compressed["thresholds"].reshape(-1, 1)
        assert restored.shape == embeddings.shape
        assert np.array_equal(restored > thresholds, embeddings > thresholds)

    def test_int8_scale_factors_match_row_maxima(self, compressor, embeddings):
        compressed = compressor.compress_embeddings(embeddings, "int8")
        assert np.allclose(
            compressed["scale_factors"], np.max(np.abs(embeddings), axis=1)
        )
        assert compressed["compression_ratio"] > 3.0

    def test_int8_handles_zero_rows(self, compressor):
        zeros = np.zeros((2, 8), dtype=np.float32)
        compressed = compressor.compress_embeddings(zeros, "int8")
        assert np.array_equal(compressed["scale_factors"], np.ones(2))
        assert np.array_equal(compressor.decompress_embeddings(compressed), zeros)

    def test_uint8_handles_constant_rows(self, compressor):
        constant = np.full((2, 8), 3.5, dtype=np.float32)
        compressed = compressor.compress_embeddings(constant, "uint8")
        assert np.allclose(compressor.decompress_embeddings(compressed), constant)

    def test_float16_reports_ratio_of_two(self, compressor, embeddings):
        compressed = compressor.compress_embeddings(embeddings, "float16")
        assert compressed["data"].dtype == np.float16
        assert compressed["compression_ratio"] == pytest.approx(2.0)

    @pytest.mark.parametrize("method", ["median", "mean", "zero"])
    def test_binary_threshold_methods(self, compressor, embeddings, method):
        compressed = compressor.compress_embeddings(
            embeddings, "binary", threshold_method=method
        )
        assert compressed["threshold_method"] == method
        if method == "zero":
            assert np.array_equal(compressed["thresholds"], np.zeros(len(embeddings)))

    def test_rejects_unknown_compression_type(self, compressor, embeddings):
        with pytest.raises(ValueError, match="Unknown compression type"):
            compressor.compress_embeddings(embeddings, "int4")

    def test_rejects_unknown_decompression_type(self, compressor):
        with pytest.raises(ValueError, match="Unknown compression type"):
            compressor.decompress_embeddings({"compression_type": "int4"})


class TestAnalyzeCompressionQuality:
    def test_float16_preserves_direction_almost_exactly(self, compressor, embeddings):
        compressed = compressor.compress_embeddings(embeddings, "float16")
        metrics = compressor.analyze_compression_quality(embeddings, compressed)
        assert metrics["mean_cosine_similarity"] > 0.999
        assert metrics["mse"] < 1e-4
        assert metrics["mae"] < 1e-2
        assert metrics["compression_ratio"] == pytest.approx(2.0)

    def test_binary_loses_more_quality_than_int8(self, compressor, embeddings):
        int8 = compressor.analyze_compression_quality(
            embeddings, compressor.compress_embeddings(embeddings, "int8")
        )
        binary = compressor.analyze_compression_quality(
            embeddings, compressor.compress_embeddings(embeddings, "binary")
        )
        assert binary["mean_cosine_similarity"] < int8["mean_cosine_similarity"]
        assert set(binary) >= {"min_cosine_similarity", "std_cosine_similarity"}

    def test_handles_zero_embeddings(self, compressor):
        zeros = np.zeros((2, 4), dtype=np.float32)
        metrics = compressor.analyze_compression_quality(
            zeros, compressor.compress_embeddings(zeros, "float16")
        )
        assert metrics["mse"] == 0.0
        assert metrics["mean_cosine_similarity"] == 0.0


class TestRecommendCompression:
    def test_recommends_highest_quality_by_default(self, compressor, embeddings):
        assert compressor.recommend_compression(embeddings) == "float16"

    def test_target_ratio_selects_more_aggressive_method(self, compressor, embeddings):
        recommended = compressor.recommend_compression(embeddings, target_ratio=3.5)
        assert recommended in {"int8", "uint8"}

    def test_unreachable_target_ratio_returns_highest_ratio_method(
        self, compressor, embeddings
    ):
        recommended = compressor.recommend_compression(embeddings, target_ratio=1000.0)
        assert recommended in {"float16", "int8", "uint8", "binary"}

    def test_strict_quality_threshold_returns_best_available(
        self, compressor, embeddings
    ):
        assert (
            compressor.recommend_compression(embeddings, quality_threshold=1.5)
            == "float16"
        )

    def test_falls_back_to_float_when_all_methods_fail(self, compressor, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("nope")

        monkeypatch.setattr(compressor, "compress_embeddings", boom)
        assert (
            compressor.recommend_compression(np.zeros((2, 4), dtype=np.float32))
            == "float"
        )


class TestCompressionInfo:
    @pytest.mark.parametrize(
        "compression_type", ["float", "float16", "int8", "uint8", "binary", "ubinary"]
    )
    def test_known_types_expose_metadata(self, compressor, compression_type):
        info = compressor.get_compression_info(compression_type)
        assert set(info) == {"description", "typical_ratio", "quality_loss", "use_case"}

    def test_unknown_type_returns_placeholder(self, compressor):
        assert compressor.get_compression_info("int4") == {
            "description": "Unknown compression type"
        }
