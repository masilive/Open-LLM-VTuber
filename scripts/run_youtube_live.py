"""Run the YouTube Live chat bridge."""

import asyncio
import os
import sys

from loguru import logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.open_llm_vtuber.config_manager.utils import read_yaml, validate_config
from src.open_llm_vtuber.live.youtube_live import YouTubeLivePlatform


async def main() -> None:
    """Load configuration and run the YouTube Live bridge."""
    config = validate_config(read_yaml(os.path.join(project_root, "conf.yaml")))
    youtube_config = config.live_config.youtube_live
    api_key = youtube_config.api_key
    if not api_key or api_key.startswith("${"):
        api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not youtube_config.video_ids:
        logger.error("No YouTube video IDs configured")
        return
    if not api_key:
        logger.error(
            "No YouTube API key configured; set YOUTUBE_API_KEY or live_config.youtube_live.api_key"
        )
        return
    platform = YouTubeLivePlatform(
        video_ids=youtube_config.video_ids,
        api_key=api_key,
        poll_interval_seconds=youtube_config.poll_interval_seconds,
        include_author=youtube_config.include_author,
    )
    await platform.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down YouTube Live platform")
