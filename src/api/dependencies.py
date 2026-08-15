"""Shared helpers for API route handlers."""

from typing import Any, Optional
from fastapi import HTTPException, Request, WebSocket

from .models import validate_model_name


def get_app_component(
    scope: Any,
    name: str,
    required: bool = True,
    detail: Optional[str] = None
) -> Any:
    """Fetch a manager from `app.state`.

    Args:
        scope: Request or WebSocket carrying the app
        name: Attribute name on `app.state`
        required: Raise instead of returning None when missing
        detail: Error detail used when the component is missing

    Returns:
        The component, or None when it is missing and not required

    Raises:
        HTTPException: 500 when a required component is unavailable
    """
    component = getattr(scope.app.state, name, None)

    if component is None and required:
        raise HTTPException(
            status_code=500,
            detail=detail or f"{name.replace('_', ' ').title()} not available"
        )

    return component


def get_managers(scope: Request, *names: str) -> tuple:
    """Fetch several managers from `app.state` at once.

    Args:
        scope: Request or WebSocket carrying the app
        names: Attribute names on `app.state`

    Returns:
        Tuple of components in the requested order

    Raises:
        HTTPException: 500 when any component is unavailable
    """
    components = tuple(
        getattr(scope.app.state, name, None) for name in names
    )

    if any(component is None for component in components):
        raise HTTPException(
            status_code=500,
            detail="Service not properly initialized"
        )

    return components


def get_model_context(request: Request, model_id: str) -> tuple:
    """Fetch the managers and base model name needed by model endpoints.

    Args:
        request: Incoming request
        model_id: Model name as provided in the path (may carry a suffix)

    Returns:
        Tuple of (model_manager, config_manager, base_model_name)

    Raises:
        HTTPException: 500 when a manager is unavailable
    """
    model_manager = get_app_component(
        request, 'model_manager', detail="Model manager not available"
    )
    config_manager = get_app_component(
        request, 'config_manager', detail="Configuration manager not available"
    )
    base_model_name, _ = config_manager.parse_model_name(model_id)

    return model_manager, config_manager, base_model_name


def get_websocket_managers(websocket: WebSocket, *names: str) -> Optional[tuple]:
    """Fetch managers for a WebSocket handler without raising.

    Returns:
        Tuple of components, or None when any of them is unavailable
    """
    components = tuple(
        getattr(websocket.app.state, name, None) for name in names
    )

    if any(component is None for component in components):
        return None

    return components


def resolve_input_type(model_name: str, request_input_type: Optional[str]) -> tuple:
    """Resolve the base model name and effective input type for a request.

    An explicit `input_type` takes precedence over the model name suffix.

    Args:
        model_name: Requested model name (may carry a -query/-passage suffix)
        request_input_type: input_type from the request body

    Returns:
        Tuple of (base_model_name, effective_input_type)
    """
    base_model_name, suffix_input_type = validate_model_name(model_name)
    return base_model_name, request_input_type or suffix_input_type
