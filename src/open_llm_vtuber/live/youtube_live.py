"""YouTube Live chat integration using the official YouTube Data API."""

import asyncio
from collections import deque
from typing import Any

import aiohttp
from loguru import logger

from .proxy_platform import ProxyLivePlatform

YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"


def extract_live_chat_id(video_payload: dict[str, Any]) -> str | None:
    """Extract the active live chat ID from a YouTube videos.list response.

    Args:
        video_payload: Decoded YouTube API response.

    Returns:
        The active live chat ID, or None when the video is not live.
    """
    items = video_payload.get("items") or []
    if not items:
        return None
    return items[0].get("liveStreamingDetails", {}).get("activeLiveChatId")


def extract_chat_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Extract normalized chat messages from a liveChatMessages.list response.

    Args:
        payload: Decoded YouTube API response.

    Returns:
        Normalized messages containing id, author, and text.
    """
    messages = []
    for item in payload.get("items") or []:
        text = item.get("snippet", {}).get("displayMessage", "").strip()
        author = item.get("authorDetails", {}).get("displayName", "YouTube user")
        if text:
            messages.append({"id": item.get("id", ""), "author": author, "text": text})
    return messages


class YouTubeLivePlatform(ProxyLivePlatform):
    """Poll YouTube live chat and forward messages to Open-LLM-VTuber."""

    def __init__(
        self,
        video_ids: list[str],
        api_key: str,
        poll_interval_seconds: float = 5.0,
        include_author: bool = True,
        proxy_url: str = "ws://localhost:12393/proxy-ws",
    ) -> None:
        """Initialize the YouTube integration.

        Args:
            video_ids: YouTube live video IDs to monitor.
            api_key: YouTube Data API key.
            poll_interval_seconds: Minimum delay between chat polling requests.
            include_author: Prefix forwarded text with the author's display name.
            proxy_url: Open-LLM-VTuber proxy WebSocket URL.

        Raises:
            ValueError: If no video IDs or API key are configured.
        """
        super().__init__(proxy_url)
        if not video_ids:
            raise ValueError("At least one YouTube video ID is required")
        if not api_key:
            raise ValueError("A YouTube Data API key is required")
        self._video_ids = video_ids
        self._api_key = api_key
        self._poll_interval_seconds = poll_interval_seconds
        self._include_author = include_author
        self._seen_message_ids: set[str] = set()
        self._seen_order: deque[str] = deque(maxlen=1000)

    async def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """Perform a YouTube API GET request."""
        session = self._ensure_session()
        async with session.get(
            f"{YOUTUBE_API_BASE_URL}/{path}",
            params={**params, "key": self._api_key},
            timeout=aiohttp.ClientTimeout(total=20),
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"YouTube API {response.status}: {body[:300]}")
            return await response.json()

    async def _resolve_live_chat_id(self, video_id: str) -> str | None:
        """Resolve the active live chat ID for a video."""
        payload = await self._get_json(
            "videos", {"part": "liveStreamingDetails", "id": video_id}
        )
        return extract_live_chat_id(payload)

    def _remember_message(self, message_id: str) -> bool:
        """Return True when a message ID has not been seen recently."""
        if message_id and message_id in self._seen_message_ids:
            return False
        if len(self._seen_order) == self._seen_order.maxlen:
            self._seen_message_ids.discard(self._seen_order[0])
        if message_id:
            self._seen_order.append(message_id)
            self._seen_message_ids.add(message_id)
        return True

    async def _poll_chat(self, live_chat_id: str) -> None:
        """Poll one live chat until the surrounding task is cancelled."""
        page_token = None
        while self._running:
            params: dict[str, Any] = {
                "liveChatId": live_chat_id,
                "part": "snippet,authorDetails",
                "maxResults": 200,
            }
            if page_token:
                params["pageToken"] = page_token
            payload = await self._get_json("liveChatMessages", params)
            for message in extract_chat_messages(payload):
                if not self._remember_message(message["id"]):
                    continue
                text = (
                    f"{message['author']}: {message['text']}"
                    if self._include_author
                    else message["text"]
                )
                await self._send_to_proxy(text)
            page_token = payload.get("nextPageToken", page_token)
            api_delay_ms = payload.get("pollingIntervalMillis", 5000)
            await asyncio.sleep(
                max(self._poll_interval_seconds, api_delay_ms / 1000, 1.0)
            )

    async def _poll_video_loop(self, video_id: str) -> None:
        """Maintain polling for one video with bounded retry backoff."""
        backoff = 5.0
        while self._running:
            try:
                live_chat_id = await self._resolve_live_chat_id(video_id)
                if not live_chat_id:
                    logger.warning(f"YouTube video {video_id} has no active live chat")
                    await asyncio.sleep(30)
                    continue
                logger.info(f"Monitoring YouTube live chat for video {video_id}")
                await self._poll_chat(live_chat_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"YouTube polling error for {video_id}: {exc}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 120)

    async def run(self) -> None:
        """Run the YouTube polling tasks and proxy receiver."""
        self._running = True
        if not await self.connect():
            await self.disconnect()
            return
        receive_task = asyncio.create_task(self.start_receiving())
        poll_tasks = [
            asyncio.create_task(self._poll_video_loop(video_id))
            for video_id in self._video_ids
        ]
        try:
            await asyncio.wait(
                [receive_task, *poll_tasks], return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            self._running = False
            for task in [receive_task, *poll_tasks]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(receive_task, *poll_tasks, return_exceptions=True)
            await self.disconnect()
