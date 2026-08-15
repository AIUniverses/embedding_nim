"""Model name parsing shared by the API layer and the configuration manager."""

from typing import Optional

INPUT_TYPE_SUFFIXES = {
    '-query': 'query',
    '-passage': 'passage'
}


def parse_model_name(model_name: str) -> tuple[str, Optional[str]]:
    """Split an input_type suffix off a model name.

    Args:
        model_name: Model name (potentially with -query or -passage suffix)

    Returns:
        Tuple of (base_model_name, input_type)

    Examples:
        "e5-large-v2" -> ("e5-large-v2", None)
        "e5-large-v2-query" -> ("e5-large-v2", "query")
        "e5-large-v2-passage" -> ("e5-large-v2", "passage")
    """
    for suffix, input_type in INPUT_TYPE_SUFFIXES.items():
        if model_name.endswith(suffix):
            return model_name[:-len(suffix)], input_type

    return model_name, None
