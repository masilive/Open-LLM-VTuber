"""Environment-aware API key resolution for LLM providers."""

import os
from collections.abc import Iterable

from loguru import logger

_PLACEHOLDER_VALUES = {
    "",
    "your api key here",
    "your open ai api key",
    "your openai api key",
    "your anthropic api key",
    "your claude api key",
    "your windsurf api key",
    "your gemini api key",
    "api key",
    "api_key",
    "changeme",
}


def resolve_api_key(configured_key: str | None, env_names: Iterable[str]) -> str | None:
    """Resolve an API key from environment variables when configuration is a placeholder.

    Explicit non-placeholder configuration always wins. Environment variables are
    inspected in the supplied priority order.

    Args:
        configured_key: API key loaded from YAML configuration.
        env_names: Environment variable names to inspect.

    Returns:
        The configured key or the first non-empty environment value. If neither is
        available, the original configured value is returned unchanged.
    """
    normalized = (configured_key or "").strip()
    unresolved = normalized.startswith("${") and normalized.endswith("}")
    placeholder = normalized.lower() in _PLACEHOLDER_VALUES or unresolved
    if normalized and not placeholder:
        return configured_key
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value:
            logger.info(f"Using API key from environment variable {env_name}")
            return env_value
    return configured_key
