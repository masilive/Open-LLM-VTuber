"""Tests for history, grouping, message-response, and payload workflows."""

import asyncio
import json

import pytest

from src.open_llm_vtuber.chat_group import ChatGroupManager
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    delete_history,
    get_history,
    get_history_list,
    store_message,
)
from src.open_llm_vtuber.message_handler import MessageHandler
from src.open_llm_vtuber.utils.stream_audio import prepare_audio_payload


def test_history_create_store_fetch_list_and_delete(tmp_path, monkeypatch) -> None:
    """History workflow persists and removes JSON-backed conversation records."""
    monkeypatch.chdir(tmp_path)
    uid = create_new_history("conf")
    store_message("conf", uid, "human", "hello", "Human")
    store_message("conf", uid, "ai", "hi", "AI")

    history = get_history("conf", uid)
    assert [item["role"] for item in history] == ["human", "ai"]
    assert get_history_list("conf")[0]["uid"] == uid
    assert delete_history("conf", uid) is True
    assert get_history("conf", uid) == []


def test_chat_group_add_remove_and_disconnect_semantics() -> None:
    """Group workflow supports ownership, membership, removal, and disconnect."""
    manager = ChatGroupManager()
    manager.client_group_map.update({"owner": "", "member": ""})

    ok, _ = manager.add_client_to_group("owner", "member")
    assert ok is True
    assert set(manager.get_group_members("owner")) == {"owner", "member"}

    ok, _ = manager.remove_client_from_group("owner", "member")
    assert ok is True
    assert manager.get_group_members("owner") == []

    manager.client_group_map.update({"a": "", "b": ""})
    manager.add_client_to_group("a", "b")
    affected = manager.remove_client("a")
    assert set(affected) == {"a", "b"}
    assert "a" not in manager.client_group_map


@pytest.mark.asyncio
async def test_message_handler_matches_and_cleans_pending_response() -> None:
    """Frontend response matching wakes the waiter and cleanup is safe."""
    handler = MessageHandler()
    waiter = handler.wait_for_response("client", "frontend-playback-complete")
    task = asyncio.create_task(waiter)
    await asyncio.sleep(0)
    handler.handle_message("client", {"type": "frontend-playback-complete"})

    assert (await task)["type"] == "frontend-playback-complete"
    handler.cleanup_client("client")
    assert "client" not in handler._response_events


def test_silent_audio_payload_is_json_serializable() -> None:
    """Silent display workflow produces a complete frontend audio payload."""
    payload = prepare_audio_payload(audio_path=None, display_text={"text": "silent"})

    assert payload["type"] == "audio"
    assert payload["audio"] is None
    assert payload["volumes"] == []
    json.dumps(payload)
