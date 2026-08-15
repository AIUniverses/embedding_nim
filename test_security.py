"""Unit tests for API security configuration helpers."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent


def _load_module(name: str, relative_path: str):
    """Load a single module by path, without importing the whole src package.

    The src package pulls in torch and the model stack, which these tests do not
    need.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


security = _load_module('embedding_security', 'src/api/security.py')
caching = _load_module('embedding_caching', 'src/utils/caching.py')

docs_enabled = security.docs_enabled
get_allowed_hosts = security.get_allowed_hosts
get_api_key = security.get_api_key
get_cors_config = security.get_cors_config
is_authorized = security.is_authorized
deserialize_value = caching.deserialize_value
serialize_value = caching.serialize_value


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove security environment variables before each test."""
    for name in (
        'EMBEDDING_API_KEY',
        'ALLOW_UNAUTHENTICATED',
        'CORS_ORIGINS',
        'ALLOWED_HOSTS',
        'ENABLE_DOCS',
    ):
        monkeypatch.delenv(name, raising=False)


def test_missing_api_key_fails_closed():
    with pytest.raises(RuntimeError):
        get_api_key()


def test_short_api_key_rejected(monkeypatch):
    monkeypatch.setenv('EMBEDDING_API_KEY', 'too-short')
    with pytest.raises(RuntimeError):
        get_api_key()


def test_unauthenticated_opt_in(monkeypatch):
    monkeypatch.setenv('ALLOW_UNAUTHENTICATED', 'true')
    assert get_api_key() is None


def test_api_key_returned(monkeypatch):
    monkeypatch.setenv('EMBEDDING_API_KEY', 'a' * 32)
    assert get_api_key() == 'a' * 32


@pytest.mark.parametrize(
    'header,expected',
    [
        ('Bearer ' + 'a' * 32, True),
        ('bearer ' + 'a' * 32, True),
        ('Bearer ' + 'b' * 32, False),
        ('Basic ' + 'a' * 32, False),
        (None, False),
    ],
)
def test_is_authorized(header, expected):
    assert is_authorized(header, 'a' * 32) is expected


def test_is_authorized_without_api_key():
    assert is_authorized(None, None) is True


def test_wildcard_cors_disables_credentials(monkeypatch):
    monkeypatch.setenv('CORS_ORIGINS', '*')
    cors = get_cors_config()
    assert cors.origins == ['*']
    assert cors.allow_credentials is False


def test_cors_defaults_to_no_origins():
    cors = get_cors_config()
    assert cors.origins == []
    assert cors.allow_credentials is False


def test_explicit_cors_origins(monkeypatch):
    monkeypatch.setenv('CORS_ORIGINS', 'https://a.example, https://b.example')
    cors = get_cors_config()
    assert cors.origins == ['https://a.example', 'https://b.example']
    assert cors.allow_credentials is True


def test_allowed_hosts(monkeypatch):
    assert get_allowed_hosts() == ['*']
    monkeypatch.setenv('ALLOWED_HOSTS', 'api.example,localhost')
    assert get_allowed_hosts() == ['api.example', 'localhost']


def test_docs_disabled_by_default(monkeypatch):
    assert docs_enabled() is False
    monkeypatch.setenv('ENABLE_DOCS', 'true')
    assert docs_enabled() is True


def test_cache_serialization_round_trip():
    value = {'data': [{'index': 0, 'embedding': [0.1, 0.2]}], 'usage': {'total_tokens': 3}}
    assert deserialize_value(serialize_value(value)) == value


def test_cache_deserialization_rejects_pickle():
    import pickle

    with pytest.raises(ValueError):
        deserialize_value(pickle.dumps({'a': 1}))
