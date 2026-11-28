"""Kick live chat integration using Kick's official Events API webhooks."""

import asyncio
import base64
import json
from typing import Any

import aiohttp
from aiohttp import web
from loguru import logger

from .proxy_platform import ProxyLivePlatform

KICK_API_BASE_URL = "https://api.kick.com"
KICK_TOKEN_URL = "https://id.kick.com/oauth/token"


def extract_kick_chat_message(payload: dict[str, Any]) -> dict[str, str] | None:
    """Normalize a Kick chat.message.sent webhook payload.

    Args:
        payload: Decoded webhook JSON body.

    Returns:
        A normalized message with id, author, and text, or None for empty content.
    """
    text = str(payload.get("content") or "").strip()
    if not text:
        return None
    sender = payload.get("sender") or {}
    return {
        "id": str(payload.get("message_id") or ""),
        "author": str(sender.get("username") or "Kick user"),
        "text": text,
    }


def verify_kick_signature(
    public_key_pem: str,
    message_id: str,
    timestamp: str,
    raw_body: bytes,
    signature: str,
) -> bool:
    """Verify a Kick webhook signature with the current Kick public key.

    Args:
        public_key_pem: PEM-encoded Kick public key.
        message_id: Kick-Event-Message-Id header value.
        timestamp: Kick-Event-Message-Timestamp header value.
        raw_body: Exact unparsed request body bytes.
        signature: Base64 Kick-Event-Signature header value.

    Returns:
        True when the signature is valid.
    """
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        public_key = serialization.load_pem_public_key(public_key_pem.encode())
        signed = (
            message_id.encode("utf-8")
            + b"."
            + timestamp.encode("utf-8")
            + b"."
            + raw_body
        )
        public_key.verify(
            base64.b64decode(signature),
            signed,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception as exc:
        logger.warning(f"Kick webhook signature verification failed: {exc}")
        return False


class KickLivePlatform(ProxyLivePlatform):
    """Receive Kick chat webhooks and forward messages to Open-LLM-VTuber."""

    def __init__(
        self,
        channel_slugs: list[str],
        client_id: str,
        client_secret: str,
        webhook_host: str = "0.0.0.0",
        webhook_port: int = 8765,
        webhook_path: str = "/kick/events",
        include_author: bool = True,
        verify_signatures: bool = True,
        proxy_url: str = "ws://localhost:12393/proxy-ws",
    ) -> None:
        """Initialize the Kick integration.

        Args:
            channel_slugs: Kick channel slugs to monitor.
            client_id: Kick developer application client ID.
            client_secret: Kick developer application client secret.
            webhook_host: Local host/interface for the webhook listener.
            webhook_port: Local port for the webhook listener.
            webhook_path: Local webhook path matching the publicly exposed URL configured in Kick.
            include_author: Prefix forwarded text with the sender name.
            verify_signatures: Verify Kick webhook signatures before processing.
            proxy_url: Open-LLM-VTuber proxy WebSocket URL.

        Raises:
            ValueError: If required values are missing.
        """
        super().__init__(proxy_url)
        if not channel_slugs:
            raise ValueError("At least one Kick channel slug is required")
        if not client_id or not client_secret:
            raise ValueError("Kick client_id and client_secret are required")
        self._channel_slugs = channel_slugs
        self._client_id = client_id
        self._client_secret = client_secret
        self._webhook_host = webhook_host
        self._webhook_port = webhook_port
        self._webhook_path = webhook_path
        self._include_author = include_author
        self._verify_signatures = verify_signatures
        self._access_token: str | None = None
        self._public_key_pem: str | None = None
        self._web_runner: web.AppRunner | None = None
        self._seen_message_ids: set[str] = set()

    async def _api_request(
        self, method: str, url: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Perform a Kick API request and return decoded JSON."""
        session = self._ensure_session()
        async with session.request(method, url, **kwargs) as response:
            body = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"Kick API {response.status}: {body[:300]}")
            return json.loads(body) if body else {}

    async def _obtain_app_token(self) -> str:
        """Obtain an app access token through the client credentials grant."""
        payload = await self._api_request(
            "POST",
            KICK_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
            timeout=aiohttp.ClientTimeout(total=20),
        )
        token = payload.get("access_token")
        if not token:
            raise RuntimeError("Kick token response did not contain access_token")
        return str(token)

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers for the current app token."""
        if not self._access_token:
            raise RuntimeError("Kick app access token is not available")
        return {"Authorization": f"Bearer {self._access_token}"}

    async def _resolve_channel_ids(self) -> list[int]:
        """Resolve configured Kick channel slugs to broadcaster user IDs."""
        params = [("slug", slug) for slug in self._channel_slugs]
        payload = await self._api_request(
            "GET",
            f"{KICK_API_BASE_URL}/public/v1/channels",
            params=params,
            headers=self._auth_headers(),
            timeout=aiohttp.ClientTimeout(total=20),
        )
        ids = [int(item["broadcaster_user_id"]) for item in payload.get("data", [])]
        missing = len(self._channel_slugs) - len(ids)
        if missing:
            logger.warning(f"Could not resolve {missing} configured Kick channel(s)")
        return ids

    async def _subscribe_chat_events(self, broadcaster_user_id: int) -> None:
        """Subscribe the app to chat.message.sent for one broadcaster."""
        payload = {
            "broadcaster_user_id": broadcaster_user_id,
            "events": [{"name": "chat.message.sent", "version": 1}],
            "method": "webhook",
        }
        result = await self._api_request(
            "POST",
            f"{KICK_API_BASE_URL}/public/v1/events/subscriptions",
            json=payload,
            headers=self._auth_headers(),
            timeout=aiohttp.ClientTimeout(total=20),
        )
        errors = [
            item.get("error") for item in result.get("data", []) if item.get("error")
        ]
        if errors:
            raise RuntimeError(f"Kick event subscription failed: {errors}")
        logger.info(
            f"Subscribed to Kick chat events for broadcaster {broadcaster_user_id}"
        )

    async def _fetch_public_key(self) -> str:
        """Fetch Kick's current webhook public key."""
        payload = await self._api_request(
            "GET",
            f"{KICK_API_BASE_URL}/public/v1/public-key",
            timeout=aiohttp.ClientTimeout(total=20),
        )
        key = payload.get("data", {}).get("public_key")
        if not key:
            raise RuntimeError(
                "Kick public key response did not contain data.public_key"
            )
        return str(key)

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Handle one Kick webhook POST request."""
        raw_body = await request.read()
        event_type = request.headers.get("Kick-Event-Type", "")
        if event_type != "chat.message.sent":
            return web.Response(status=202)
        if self._verify_signatures:
            if not self._public_key_pem:
                return web.Response(status=503, text="Kick public key unavailable")
            valid = verify_kick_signature(
                self._public_key_pem,
                request.headers.get("Kick-Event-Message-Id", ""),
                request.headers.get("Kick-Event-Message-Timestamp", ""),
                raw_body,
                request.headers.get("Kick-Event-Signature", ""),
            )
            if not valid:
                return web.Response(status=401, text="invalid signature")
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return web.Response(status=400, text="invalid json")
        message = extract_kick_chat_message(payload)
        if message is None:
            return web.Response(status=202)
        if message["id"] and message["id"] in self._seen_message_ids:
            return web.Response(status=200)
        if message["id"]:
            self._seen_message_ids.add(message["id"])
            if len(self._seen_message_ids) > 2000:
                self._seen_message_ids.clear()
                self._seen_message_ids.add(message["id"])
        text = (
            f"{message['author']}: {message['text']}"
            if self._include_author
            else message["text"]
        )
        await self._send_to_proxy(text)
        return web.Response(status=200)

    async def _start_webhook_server(self) -> None:
        """Start the local webhook listener."""
        app = web.Application()
        app.router.add_post(self._webhook_path, self._handle_webhook)
        self._web_runner = web.AppRunner(app)
        await self._web_runner.setup()
        site = web.TCPSite(self._web_runner, self._webhook_host, self._webhook_port)
        await site.start()
        logger.info(
            f"Kick webhook listener running at http://{self._webhook_host}:{self._webhook_port}{self._webhook_path}"
        )
        logger.warning(
            "Kick requires this webhook path to be publicly reachable and configured in the Kick Developer portal"
        )

    async def run(self) -> None:
        """Run webhook reception, subscription setup, and proxy forwarding."""
        self._running = True
        if not await self.connect():
            await self.disconnect()
            return
        receive_task = asyncio.create_task(self.start_receiving())
        try:
            if self._verify_signatures:
                self._public_key_pem = await self._fetch_public_key()
            self._access_token = await self._obtain_app_token()
            channel_ids = await self._resolve_channel_ids()
            if not channel_ids:
                raise RuntimeError("No configured Kick channels could be resolved")
            for broadcaster_id in channel_ids:
                await self._subscribe_chat_events(broadcaster_id)
            await self._start_webhook_server()
            await receive_task
        finally:
            self._running = False
            if not receive_task.done():
                receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)
            if self._web_runner is not None:
                await self._web_runner.cleanup()
            await self.disconnect()
