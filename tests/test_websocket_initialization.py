"""Tests for WebSocket initialization-time message ordering."""

import pytest

from src.open_llm_vtuber.websocket_handler import WebSocketHandler


class FakeWebSocket:
    """Minimal WebSocket test double."""

    def __init__(self) -> None:
        """Initialize sent-message storage."""
        self.text_messages = []
        self.json_messages = []

    async def send_text(self, payload: str) -> None:
        """Record a text payload."""
        self.text_messages.append(payload)

    async def send_json(self, payload: dict) -> None:
        """Record a JSON payload."""
        self.json_messages.append(payload)


@pytest.mark.asyncio
async def test_heartbeat_is_not_buffered_during_initialization() -> None:
    """Heartbeat messages are answered before session initialization completes."""
    handler = WebSocketHandler(default_context_cache=None)
    websocket = FakeWebSocket()
    client_uid = "client-1"
    handler.initializing_clients.add(client_uid)

    await handler._route_message(websocket, client_uid, {"type": "heartbeat"})

    assert websocket.json_messages == [{"type": "heartbeat-ack"}]
    assert handler.pending_client_messages.get(client_uid) is None


@pytest.mark.asyncio
async def test_initialization_messages_flush_in_order() -> None:
    """Context-dependent messages are buffered and replayed in receive order."""
    handler = WebSocketHandler(default_context_cache=None)
    websocket = FakeWebSocket()
    client_uid = "client-1"
    calls = []

    async def record_message(_websocket, _client_uid, data):
        calls.append(data["text"])

    handler._message_handlers["text-input"] = record_message
    handler.initializing_clients.add(client_uid)

    await handler._route_message(
        websocket, client_uid, {"type": "text-input", "text": "first"}
    )
    await handler._route_message(
        websocket, client_uid, {"type": "text-input", "text": "second"}
    )

    assert calls == []
    assert len(handler.pending_client_messages[client_uid]) == 2

    handler.client_contexts[client_uid] = object()
    await handler._flush_pending_messages(websocket, client_uid)

    assert calls == ["first", "second"]
    assert client_uid not in handler.initializing_clients
    assert client_uid not in handler.pending_client_messages


@pytest.mark.asyncio
async def test_messages_arriving_during_flush_remain_ordered() -> None:
    """Messages re-entrant to the flush remain behind already buffered work."""
    handler = WebSocketHandler(default_context_cache=None)
    websocket = FakeWebSocket()
    client_uid = "client-1"
    calls = []

    async def first_handler(_websocket, _client_uid, _data):
        calls.append("first")
        await handler._route_message(websocket, client_uid, {"type": "second-message"})

    async def second_handler(_websocket, _client_uid, _data):
        calls.append("second")

    handler._message_handlers["first-message"] = first_handler
    handler._message_handlers["second-message"] = second_handler
    handler.initializing_clients.add(client_uid)
    await handler._route_message(websocket, client_uid, {"type": "first-message"})
    handler.client_contexts[client_uid] = object()

    await handler._flush_pending_messages(websocket, client_uid)

    assert calls == ["first", "second"]
    assert client_uid not in handler.initializing_clients
    assert client_uid not in handler.pending_client_messages
