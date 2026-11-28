"""Focused tests for live platform normalization and API key resolution."""

import base64
import json

import pytest

from src.open_llm_vtuber.agent.api_key_resolver import resolve_api_key
from src.open_llm_vtuber.config_manager.live import KickLiveConfig, YouTubeLiveConfig
from src.open_llm_vtuber.config_manager.stateless_llm import StatelessLLMConfigs
from src.open_llm_vtuber.live.kick_live import (
    extract_kick_chat_message,
    verify_kick_signature,
)
from src.open_llm_vtuber.live.youtube_live import (
    extract_chat_messages,
    extract_live_chat_id,
)


def test_youtube_extracts_live_chat_id() -> None:
    """YouTube live chat IDs are extracted from videos.list responses."""
    payload = {
        "items": [
            {"liveStreamingDetails": {"activeLiveChatId": "chat-123"}},
        ]
    }
    assert extract_live_chat_id(payload) == "chat-123"
    assert extract_live_chat_id({"items": []}) is None


def test_youtube_extracts_nonempty_chat_messages() -> None:
    """YouTube messages are normalized and empty messages are omitted."""
    payload = {
        "items": [
            {
                "id": "m1",
                "snippet": {"displayMessage": " hello "},
                "authorDetails": {"displayName": "Alice"},
            },
            {"id": "m2", "snippet": {"displayMessage": ""}},
        ]
    }
    assert extract_chat_messages(payload) == [
        {"id": "m1", "author": "Alice", "text": "hello"}
    ]


def test_kick_extracts_chat_message() -> None:
    """Kick webhook payloads are normalized."""
    payload = {
        "message_id": "m1",
        "sender": {"username": "Bob"},
        "content": "hello kick",
    }
    assert extract_kick_chat_message(payload) == {
        "id": "m1",
        "author": "Bob",
        "text": "hello kick",
    }
    assert extract_kick_chat_message({"content": ""}) is None


def test_kick_signature_verification() -> None:
    """A valid RSA PKCS#1 v1.5/SHA-256 Kick signature verifies."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private_key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    body = json.dumps({"content": "hello"}).encode()
    signed = b"message-id.2026-01-01T00:00:00Z." + body
    signature = base64.b64encode(
        private_key.sign(signed, padding.PKCS1v15(), hashes.SHA256())
    ).decode()

    assert verify_kick_signature(
        public_pem, "message-id", "2026-01-01T00:00:00Z", body, signature
    )
    assert not verify_kick_signature(
        public_pem, "wrong", "2026-01-01T00:00:00Z", body, signature
    )


def test_environment_api_key_resolution(monkeypatch) -> None:
    """Environment keys replace placeholders without overriding explicit keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert resolve_api_key("Your Open AI API key", ["OPENAI_API_KEY"]) == "env-key"
    assert resolve_api_key("configured-key", ["OPENAI_API_KEY"]) == "configured-key"
    assert resolve_api_key("${OPENAI_API_KEY}", ["MISSING_KEY"]) == "${OPENAI_API_KEY}"


def test_live_and_windsurf_config_models_validate() -> None:
    """New live and Windsurf configuration models validate."""
    youtube = YouTubeLiveConfig(video_ids=["video"], api_key="key")
    kick = KickLiveConfig(
        channel_slugs=["channel"], client_id="id", client_secret="secret"
    )
    llms = StatelessLLMConfigs(
        windsurf_llm={
            "base_url": "http://localhost:3003/v1",
            "llm_api_key": "key",
            "model": "model",
        }
    )
    assert youtube.video_ids == ["video"]
    assert kick.channel_slugs == ["channel"]
    assert llms.windsurf_llm is not None
    assert llms.windsurf_llm.base_url == "http://localhost:3003/v1"


def test_factory_resolves_provider_environment_keys(monkeypatch) -> None:
    """LLMFactory resolves provider-specific keys before constructing clients."""
    from src.open_llm_vtuber.agent import stateless_llm_factory

    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "openai-env")
    monkeypatch.setattr(stateless_llm_factory, "OpenAICompatibleLLM", FakeLLM)
    stateless_llm_factory.LLMFactory.create_llm(
        "openai_llm",
        model="gpt-test",
        base_url="https://api.openai.com/v1",
        llm_api_key="Your Open AI API key",
        organization_id=None,
        project_id=None,
        temperature=1.0,
    )
    assert captured["llm_api_key"] == "openai-env"

    captured.clear()
    monkeypatch.setenv("WINDSURF_API_KEY", "windsurf-env")
    stateless_llm_factory.LLMFactory.create_llm(
        "windsurf_llm",
        model="bridge-model",
        base_url="http://localhost:3003/v1",
        llm_api_key="${WINDSURF_API_KEY}",
        organization_id=None,
        project_id=None,
        temperature=1.0,
    )
    assert captured["llm_api_key"] == "windsurf-env"
