"""Unit tests for src/config_manager.py."""

import pytest
import yaml

from src.config_manager import (
    BatchingConfig,
    CachingConfig,
    ConfigManager,
    DeploymentConfig,
)


class TestDataclassDefaults:
    def test_deployment_defaults(self):
        config = DeploymentConfig()
        assert config.mode == "simple"
        assert config.enable_redis is False
        assert config.redis_url == "redis://localhost:6379"
        assert config.prometheus_port == 9090
        assert config.grafana_port == 3000
        assert config.max_workers == 1

    def test_caching_defaults(self):
        config = CachingConfig()
        assert config.enabled is True
        assert config.backend == "memory"
        assert config.ttl_seconds == 3600
        assert config.max_size == 1000
        assert config.compression is False

    def test_batching_defaults(self):
        config = BatchingConfig()
        assert config.enabled is True
        assert config.max_batch_size == 32
        assert config.timeout_ms == 10
        assert config.adaptive_sizing is True
        assert config.priority_levels == 3


class TestConfigLoading:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ConfigManager(str(tmp_path / "absent.yaml"))

    def test_non_mapping_config_raises(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a dictionary"):
            ConfigManager(str(path))

    def test_missing_models_section_raises(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump({"service": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="must contain 'models' section"):
            ConfigManager(str(path))

    def test_invalid_yaml_raises(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text("models: [unclosed\n", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            ConfigManager(str(path))

    def test_reload_picks_up_changes(self, tmp_path, config_dict):
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
        manager = ConfigManager(str(path))
        assert "new-model" not in manager.get_available_models()

        config_dict["models"]["new-model"] = {"model_id": "org/new", "family": "gte"}
        path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
        manager.reload_config()
        assert "new-model" in manager.get_available_models()


class TestEnvironmentConfig:
    def test_defaults_without_environment_overrides(self, config_manager):
        assert config_manager.get_deployment_mode() == "simple"
        assert config_manager.is_redis_enabled() is False
        assert config_manager.is_monitoring_enabled() is False
        assert config_manager.is_vector_db_enabled() is False
        assert config_manager.get_redis_url() == "redis://localhost:6379"
        assert config_manager.is_caching_enabled() is True
        assert config_manager.get_cache_backend() == "memory"
        assert config_manager.get_cache_ttl() == 3600

    def test_environment_overrides_are_applied(self, monkeypatch, config_file):
        monkeypatch.setenv("DEPLOYMENT_MODE", "full")
        monkeypatch.setenv("ENABLE_REDIS", "TRUE")
        monkeypatch.setenv("ENABLE_MONITORING", "true")
        monkeypatch.setenv("ENABLE_VECTOR_DB", "true")
        monkeypatch.setenv("REDIS_URL", "redis://cache:6380")
        monkeypatch.setenv("PROMETHEUS_PORT", "9191")
        monkeypatch.setenv("GRAFANA_PORT", "3001")
        monkeypatch.setenv("WORKERS", "4")
        monkeypatch.setenv("CACHE_TTL", "60")
        monkeypatch.setenv("CACHE_MAX_SIZE", "10")
        monkeypatch.setenv("CACHE_COMPRESSION", "true")
        monkeypatch.setenv("ENABLE_CACHING", "false")
        monkeypatch.setenv("ENABLE_BATCHING", "false")
        monkeypatch.setenv("MAX_BATCH_SIZE", "8")
        monkeypatch.setenv("BATCH_TIMEOUT_MS", "25")
        monkeypatch.setenv("ADAPTIVE_BATCHING", "false")
        monkeypatch.setenv("PRIORITY_LEVELS", "5")

        manager = ConfigManager(config_file)

        assert manager.get_deployment_mode() == "full"
        assert manager.is_redis_enabled() is True
        assert manager.is_monitoring_enabled() is True
        assert manager.is_vector_db_enabled() is True
        assert manager.get_redis_url() == "redis://cache:6380"
        assert manager.deployment.prometheus_port == 9191
        assert manager.deployment.grafana_port == 3001
        assert manager.deployment.max_workers == 4
        assert manager.is_caching_enabled() is False
        assert manager.get_cache_ttl() == 60
        assert manager.caching.max_size == 10
        assert manager.caching.compression is True
        assert manager.batching.enabled is False
        assert manager.batching.max_batch_size == 8
        assert manager.batching.timeout_ms == 25
        assert manager.batching.adaptive_sizing is False
        assert manager.batching.priority_levels == 5

    def test_redis_enabled_switches_default_cache_backend(
        self, monkeypatch, config_file
    ):
        monkeypatch.setenv("ENABLE_REDIS", "true")
        assert ConfigManager(config_file).get_cache_backend() == "redis"

    def test_explicit_cache_backend_wins(self, monkeypatch, config_file):
        monkeypatch.setenv("ENABLE_REDIS", "true")
        monkeypatch.setenv("CACHE_BACKEND", "disk")
        assert ConfigManager(config_file).get_cache_backend() == "disk"


class TestModelLookups:
    def test_available_models_fall_back_to_name(self, config_manager):
        assert config_manager.get_available_models() == {
            "e5-small-v2": "E5 Small V2",
            "gte-base": "gte-base",
        }

    def test_get_model_config_returns_a_copy(self, config_manager):
        config = config_manager.get_model_config("e5-small-v2")
        config["family"] = "mutated"
        assert config_manager.get_model_family("e5-small-v2") == "e5"

    def test_unknown_model_raises_with_available_list(self, config_manager):
        with pytest.raises(KeyError, match="not found"):
            config_manager.get_model_config("nope")

    def test_model_attribute_accessors(self, config_manager):
        assert config_manager.supports_input_type("e5-small-v2") is True
        assert config_manager.get_supported_embedding_types("e5-small-v2") == [
            "float",
            "int8",
            "binary",
        ]
        assert config_manager.get_supported_dimensions("e5-small-v2") == [128, 256, 384]
        assert config_manager.get_supported_modalities("gte-base") == [
            "text",
            "image",
            "text_image",
        ]
        assert config_manager.get_embedding_dimension("e5-small-v2") == 384
        assert config_manager.get_max_seq_length("e5-small-v2") == 512
        assert config_manager.get_model_settings("e5-small-v2") == {
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
        }

    def test_accessor_defaults_for_sparse_model_config(self, config_manager):
        assert config_manager.supports_input_type("gte-base") is False
        assert config_manager.get_supported_embedding_types("gte-base") == ["float"]
        assert config_manager.get_supported_dimensions("gte-base") == []
        assert config_manager.get_max_seq_length("gte-base") == 512
        assert config_manager.get_model_settings("gte-base") == {}

    def test_missing_family_reported_as_unknown(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump({"models": {"m": {}}}), encoding="utf-8")
        manager = ConfigManager(str(path))
        assert manager.get_model_family("m") == "unknown"
        assert manager.get_embedding_dimension("m") == 768
        assert manager.get_supported_modalities("m") == ["text"]


class TestServiceConfig:
    def test_reads_service_section(self, config_manager):
        assert config_manager.get_default_model() == "e5-small-v2"
        assert config_manager.get_max_batch_size() == 4
        assert config_manager.get_max_input_length() == 64
        assert config_manager.is_dynamic_batching_enabled() is False
        assert config_manager.get_batch_timeout_ms() == 5
        assert config_manager.get_truncate_strategy() == "none"
        assert config_manager.should_return_metadata() is True

    def test_service_defaults_when_section_absent(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump({"models": {"m": {}}}), encoding="utf-8")
        manager = ConfigManager(str(path))
        assert manager.get_service_config() == {}
        assert manager.get_default_model() == "e5-large-v2"
        assert manager.get_max_batch_size() == 64
        assert manager.get_max_input_length() == 8192
        assert manager.is_dynamic_batching_enabled() is True
        assert manager.get_batch_timeout_ms() == 100
        assert manager.get_truncate_strategy() == "none"
        assert manager.should_return_metadata() is True


class TestParseModelName:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("e5-small-v2", ("e5-small-v2", None)),
            ("e5-small-v2-query", ("e5-small-v2", "query")),
            ("e5-small-v2-passage", ("e5-small-v2", "passage")),
        ],
    )
    def test_suffix_parsing(self, config_manager, name, expected):
        assert config_manager.parse_model_name(name) == expected


class TestValidateModelConfig:
    def test_valid_model_has_no_issues(self, config_manager):
        assert config_manager.validate_model_config("e5-small-v2") == []

    def test_unknown_model_is_reported(self, config_manager):
        assert config_manager.validate_model_config("nope") == [
            "Model 'nope' not found in configuration"
        ]

    def test_reports_missing_required_fields(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump({"models": {"m": {}}}), encoding="utf-8")
        issues = ConfigManager(str(path)).validate_model_config("m")
        assert issues == [
            "Missing required field 'model_id' for model 'm'",
            "Missing required field 'family' for model 'm'",
            "Missing required field 'embedding_dimension' for model 'm'",
        ]

    def test_reports_invalid_family_types_and_modalities(self, tmp_path):
        config = {
            "models": {
                "m": {
                    "model_id": "org/m",
                    "family": "unsupported",
                    "embedding_dimension": 128,
                    "supported_embedding_types": ["float", "float4"],
                    "supported_modalities": ["text", "audio"],
                }
            }
        }
        path = tmp_path / "models.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        issues = ConfigManager(str(path)).validate_model_config("m")
        assert any("Invalid family 'unsupported'" in issue for issue in issues)
        assert any("Invalid embedding type 'float4'" in issue for issue in issues)
        assert any("Invalid modality 'audio'" in issue for issue in issues)

    def test_repository_config_is_valid(self, repo_config_manager):
        for model_name in repo_config_manager.get_available_models():
            assert repo_config_manager.validate_model_config(model_name) == []


class TestConfigSummary:
    def test_summary_reflects_config_and_environment(self, config_manager):
        summary = config_manager.get_config_summary()
        assert summary["deployment_mode"] == "simple"
        assert summary["total_models"] == 2
        assert summary["available_models"] == ["e5-small-v2", "gte-base"]
        assert summary["caching_enabled"] is True
        assert summary["caching_backend"] == "memory"
        assert summary["batching_enabled"] is True
        assert summary["max_batch_size"] == 32
        assert summary["redis_enabled"] is False
        assert summary["monitoring_enabled"] is False
        assert summary["vector_db_enabled"] is False
