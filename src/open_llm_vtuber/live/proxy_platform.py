"""Shared proxy WebSocket plumbing for live platform integrations."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import aiohttp
from loguru import logger

from .live_interface import LivePlatformInterface


class ProxyLivePlatform(LivePlatformInterface):
    """Base implementation for platforms forwarding chat through the local proxy."""

    def __init__(self, proxy_url: str = "ws://localhost:12393/proxy-ws") -> None:
        """Initialize proxy connection state.

        Args:
            proxy_url: WebSocket URL of the Open-LLM-VTuber proxy endpoint.
        """
        self._proxy_url = proxy_url
        self._session: aiohttp.ClientSession | None = None
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._connected = False
        self._running = False
        self._message_handlers: list[Callable[[dict[str, Any]], None]] = []

    @property
    def is_connected(self) -> bool:
        """Return whether the proxy WebSocket is connected."""
        return (
            self._connected
            and self._websocket is not None
            and not self._websocket.closed
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Create or return the shared aiohttp session.

        Returns:
            The active HTTP client session.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def connect(self, proxy_url: str | None = None) -> bool:
        """Connect to the Open-LLM-VTuber proxy WebSocket.

        Args:
            proxy_url: Optional override for the configured proxy URL.

        Returns:
            True when the connection was established.
        """
        target_url = proxy_url or self._proxy_url
        try:
            self._websocket = await self._ensure_session().ws_connect(
                target_url, heartbeat=20, timeout=10
            )
            self._connected = True
            logger.info(f"Connected to proxy at {target_url}")
            return True
        except Exception as exc:
            logger.error(f"Failed to connect to proxy: {exc}")
            return False

    async def disconnect(self) -> None:
        """Close proxy and HTTP connections."""
        self._running = False
        if self._websocket is not None and not self._websocket.closed:
            await self._websocket.close()
        self._websocket = None
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._connected = False

    async def send_message(self, text: str) -> bool:
        """Send a text input message to the VTuber proxy.

        Args:
            text: Chat text to forward.

        Returns:
            True when the message was sent.
        """
        return await self._send_to_proxy(text)

    async def _send_to_proxy(self, text: str) -> bool:
        """Send normalized text input through the proxy."""
        if not self.is_connected or self._websocket is None:
            logger.error("Cannot send message: not connected to proxy")
            return False
        try:
            await self._websocket.send_str(
                json.dumps({"type": "text-input", "text": text})
            )
            logger.info(f"Sent live chat message to VTuber: {text}")
            return True
        except Exception as exc:
            logger.error(f"Error sending message to proxy: {exc}")
            self._connected = False
            return False

    async def register_message_handler(
        self, handler: Callable[[dict[str, Any]], None]
    ) -> None:
        """Register a proxy response handler.

        Args:
            handler: Callable receiving decoded proxy messages.
        """
        self._message_handlers.append(handler)

    async def start_receiving(self) -> None:
        """Receive proxy responses until the connection closes."""
        if not self.is_connected or self._websocket is None:
            logger.error("Cannot receive proxy messages: not connected")
            return
        try:
            async for frame in self._websocket:
                if frame.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(frame.data)
                    await self.handle_incoming_messages(data)
                elif frame.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"Error receiving proxy messages: {exc}")
        finally:
            self._connected = False

    async def handle_incoming_messages(self, message: dict[str, Any]) -> None:
        """Dispatch a proxy response to registered handlers.

        Args:
            message: Decoded proxy response.
        """
        for handler in self._message_handlers:
            try:
                await asyncio.to_thread(handler, message)
            except Exception as exc:
                logger.error(f"Error in live platform message handler: {exc}")
