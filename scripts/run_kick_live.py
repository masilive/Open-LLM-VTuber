"""Run the Kick live chat bridge."""

import asyncio
import os
import sys

from loguru import logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.open_llm_vtuber.config_manager.utils import read_yaml, validate_config
from src.open_llm_vtuber.live.kick_live import KickLivePlatform


def _configured_or_env(configured: str, env_name: str) -> str:
    """Return a configured secret or its environment fallback."""
    if configured and not configured.startswith("${"):
        return configured
    return os.getenv(env_name, "")


async def main() -> None:
    """Load configuration and run the Kick Live bridge."""
    config = validate_config(read_yaml(os.path.join(project_root, "conf.yaml")))
    kick_config = config.live_config.kick_live
    client_id = _configured_or_env(kick_config.client_id, "KICK_CLIENT_ID")
    client_secret = _configured_or_env(kick_config.client_secret, "KICK_CLIENT_SECRET")
    if not kick_config.channel_slugs:
        logger.error("No Kick channel slugs configured")
        return
    if not client_id or not client_secret:
        logger.error(
            "Kick client credentials are missing; set KICK_CLIENT_ID and "
            "KICK_CLIENT_SECRET or configure them"
        )
        return
    platform = KickLivePlatform(
        channel_slugs=kick_config.channel_slugs,
        client_id=client_id,
        client_secret=client_secret,
        webhook_host=kick_config.webhook_host,
        webhook_port=kick_config.webhook_port,
        webhook_path=kick_config.webhook_path,
        include_author=kick_config.include_author,
        verify_signatures=kick_config.verify_signatures,
    )
    await platform.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Kick Live platform")
