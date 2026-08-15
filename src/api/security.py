"""Security configuration helpers for the embedding API."""

import hmac
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CORSConfig:
    """Resolved CORS settings."""
    origins: List[str]
    allow_credentials: bool


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _split_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def get_api_key() -> Optional[str]:
    """Return the configured API key.

    Authentication is mandatory unless explicitly disabled with
    ALLOW_UNAUTHENTICATED=true, so a missing key fails startup instead of
    silently serving an open endpoint.
    """
    api_key = os.getenv('EMBEDDING_API_KEY', '').strip()
    if api_key:
        if len(api_key) < 16:
            raise RuntimeError("EMBEDDING_API_KEY must be at least 16 characters long")
        return api_key

    if _env_flag('ALLOW_UNAUTHENTICATED'):
        logger.warning(
            "Authentication is disabled (ALLOW_UNAUTHENTICATED=true). "
            "Never use this outside a trusted local environment."
        )
        return None

    raise RuntimeError(
        "EMBEDDING_API_KEY is not set. Set it to enable API authentication, or set "
        "ALLOW_UNAUTHENTICATED=true to intentionally run the service without authentication."
    )


def is_authorized(authorization_header: Optional[str], api_key: Optional[str]) -> bool:
    """Check a bearer token against the configured API key in constant time."""
    if api_key is None:
        return True
    if not authorization_header or not authorization_header.lower().startswith('bearer '):
        return False
    token = authorization_header.split(' ', 1)[1].strip()
    return hmac.compare_digest(token, api_key)


def get_cors_config() -> CORSConfig:
    """Resolve CORS settings from CORS_ORIGINS.

    Credentialed requests are never combined with a wildcard origin, which would
    let any site read authenticated responses.
    """
    raw_origins = os.getenv('CORS_ORIGINS', '').strip()

    if not raw_origins or raw_origins == '*':
        if raw_origins == '*':
            logger.warning(
                "CORS_ORIGINS='*' allows any origin; credentialed cross-origin "
                "requests are rejected. Set an explicit origin list for production."
            )
            return CORSConfig(origins=['*'], allow_credentials=False)
        return CORSConfig(origins=[], allow_credentials=False)

    return CORSConfig(origins=_split_list(raw_origins), allow_credentials=True)


def get_allowed_hosts() -> List[str]:
    """Resolve Host header allow-list from ALLOWED_HOSTS."""
    raw_hosts = os.getenv('ALLOWED_HOSTS', '*').strip() or '*'
    if raw_hosts == '*':
        logger.warning(
            "ALLOWED_HOSTS='*' disables Host header validation; set an explicit "
            "host list for production deployments."
        )
        return ['*']
    return _split_list(raw_hosts)


def docs_enabled() -> bool:
    """Whether the interactive API docs and OpenAPI schema are served."""
    return _env_flag('ENABLE_DOCS', default=False)
