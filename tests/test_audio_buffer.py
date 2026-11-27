"""Tests for efficient microphone audio buffering and disconnect cleanup."""

import numpy as np
import pytest

from src.open_llm_vtuber.utils.audio_buffer import AudioBuffer
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


def test_audio_buffer_preserves_order_and_resets() -> None:
    """Buffered chunks are concatenated once in input order."""
    buffer = AudioBuffer()
    buffer.append_float32([0.1, 0.2])
    buffer.append_float32(np.array([0.3], dtype=np.float32))

    result = buffer.consume()

    np.testing.assert_allclose(result, [0.1, 0.2, 0.3], rtol=1e-6)
    assert len(buffer) == 0
    assert buffer.consume().size == 0


def test_audio_buffer_normalizes_int16_pcm() -> None:
    """Raw PCM bytes are converted to ASR-compatible normalized float32."""
    pcm = np.array([-32768, 0, 32767], dtype=np.int16).tobytes()
    buffer = AudioBuffer()

    buffer.append_int16_bytes(pcm)
    result = buffer.consume()

    np.testing.assert_allclose(result, [-1.0, 0.0, 32767 / 32768], rtol=1e-6)
    assert result.dtype == np.float32


def test_audio_buffer_rejects_unaligned_pcm_bytes() -> None:
    """Odd byte counts cannot represent complete 16-bit PCM samples."""
    with pytest.raises(ValueError, match="must be even"):
        AudioBuffer().append_int16_bytes(b"x")


@pytest.mark.asyncio
async def test_websocket_mic_audio_chunks_are_accumulated() -> None:
    """The WebSocket mic path stores chunks without destructive concatenation."""
    handler = WebSocketHandler(default_context_cache=None)
    client_uid = "client-audio"
    handler.received_data_buffers[client_uid] = AudioBuffer()

    await handler._handle_audio_data(None, client_uid, {"audio": [0.1, 0.2]})
    await handler._handle_audio_data(None, client_uid, {"audio": [0.3]})

    np.testing.assert_allclose(
        handler.received_data_buffers[client_uid].consume(), [0.1, 0.2, 0.3]
    )


@pytest.mark.asyncio
async def test_disconnect_closes_service_context_before_dropping_it() -> None:
    """Disconnect cleanup closes MCP/model resources held by the session context."""

    class Context:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    handler = WebSocketHandler(default_context_cache=None)
    context = Context()
    handler.client_contexts["client-close"] = context
    handler.received_data_buffers["client-close"] = AudioBuffer()

    await handler.handle_disconnect("client-close")

    assert context.close_calls == 1
    assert "client-close" not in handler.client_contexts
