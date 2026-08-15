"""Shared fixtures for the unit test suite."""

import os
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from src.config_manager import ConfigManager

CONFIG_ENV_VARS = [
    "DEPLOYMENT_MODE",
    "ENABLE_REDIS",
    "ENABLE_MONITORING",
    "ENABLE_VECTOR_DB",
    "REDIS_URL",
    "PROMETHEUS_PORT",
    "GRAFANA_PORT",
    "WORKERS",
    "ENABLE_CACHING",
    "CACHE_BACKEND",
    "CACHE_TTL",
    "CACHE_MAX_SIZE",
    "CACHE_COMPRESSION",
    "ENABLE_BATCHING",
    "MAX_BATCH_SIZE",
    "BATCH_TIMEOUT_MS",
    "ADAPTIVE_BATCHING",
    "PRIORITY_LEVELS",
    "STRICT_NIM_MODE",
    "ENABLE_FAKE_IMAGE_EMBEDDINGS",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove configuration environment variables so tests see defaults."""
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config_dict() -> Dict[str, Any]:
    """Minimal but realistic service configuration."""
    return {
        "models": {
            "e5-small-v2": {
                "model_id": "intfloat/e5-small-v2",
                "display_name": "E5 Small V2",
                "family": "e5",
                "max_seq_length": 512,
                "embedding_dimension": 384,
                "supports_input_type": True,
                "supports_dimensions": [128, 256, 384],
                "supported_embedding_types": ["float", "int8", "binary"],
                "supported_modalities": ["text"],
                "settings": {
                    "query_prefix": "query: ",
                    "passage_prefix": "passage: ",
                },
            },
            "gte-base": {
                "model_id": "thenlper/gte-base",
                "family": "gte",
                "embedding_dimension": 768,
                "supported_modalities": ["text", "image", "text_image"],
            },
        },
        "service": {
            "default_model": "e5-small-v2",
            "max_input_length": 64,
            "max_batch_size": 4,
            "enable_dynamic_batching": False,
            "batch_timeout_ms": 5,
            "truncate": "none",
            "return_metadata": True,
        },
    }


@pytest.fixture
def config_file(tmp_path, config_dict) -> str:
    """Write the configuration to a temporary YAML file."""
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    return str(path)


@pytest.fixture
def config_manager(config_file) -> ConfigManager:
    return ConfigManager(config_file)


@pytest.fixture
def repo_config_manager() -> ConfigManager:
    """Config manager backed by the configuration shipped in the repository."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return ConfigManager(os.path.join(repo_root, "config", "models.yaml"))


@pytest.fixture
def embeddings() -> np.ndarray:
    """Deterministic float32 embeddings with mixed signs and magnitudes."""
    rng = np.random.default_rng(1234)
    return rng.normal(size=(4, 16)).astype(np.float32)
