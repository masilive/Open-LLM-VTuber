"""Workflow tests for deterministic single-user conversation orchestration."""

import asyncio
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.open_llm_vtuber.agent.output_types import Actions, DisplayText, SentenceOutput
from src.open_llm_vtuber.conversations.single_conversation import (
    process_single_conversation,
)
from src.open_llm_vtuber.message_handler import message_handler


def write_test_wav(path: Path, seconds: float = 0.05, sample_rate: int = 16000) -> None:
    """Write a short deterministic mono PCM WAV file."""
    count = int(seconds * sample_rate)
    samples = (np.sin(np.arange(count) / sample_rate * 440) * 12000).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


class FakeTTS:
    """TTS test double that writes local WAV files without external services."""

    def __init__(self, cache_dir: Path) -> None:
        """Initialize the fake engine with a temporary output directory."""
        self.cache_dir = cache_dir

    async def async_generate_audio(self, text: str, file_name_no_ext: str) -> str:
        """Generate audio with an artificial out-of-order completion delay."""
        if text.startswith("First"):
            await asyncio.sleep(0.02)
        path = self.cache_dir / f"{file_name_no_ext}.wav"
        await asyncio.to_thread(write_test_wav, path)
        return str(path)

    def remove_file(self, filepath: str, verbose: bool = True) -> None:
        """Remove a generated file when present."""
        Path(filepath).unlink(missing_ok=True)


class FakeAgent:
    """Agent test double that emits deterministic sentence outputs."""

    def __init__(self, sentences: list[str]) -> None:
        """Initialize the fake agent."""
        self.sentences = sentences

    def chat(self, batch_input):
        """Return an async stream of sentence outputs."""

        async def stream():
            for sentence in self.sentences:
                yield SentenceOutput(
                    display_text=DisplayText(text=sentence),
                    tts_text=sentence,
                    actions=Actions(),
                )

        return stream()


@pytest.mark.asyncio
async def test_text_conversation_orders_audio_and_completes_once(tmp_path) -> None:
    """A text conversation emits ordered audio before one synthesis-complete signal."""
    sentences = ["First sentence.", "Second sentence.", "Third sentence."]
    context = SimpleNamespace(
        asr_engine=None,
        agent_engine=FakeAgent(sentences),
        live2d_model=None,
        tts_engine=FakeTTS(tmp_path),
        translate_engine=None,
        character_config=SimpleNamespace(
            human_name="Human",
            character_name="TestAI",
            avatar=None,
            conf_uid="test-conf",
        ),
        history_uid=None,
    )
    client_uid = "conversation-test-client"
    messages = []

    async def websocket_send(message: str) -> None:
        payload = json.loads(message)
        messages.append(payload)
        if payload["type"] == "backend-synth-complete":
            audio_count = sum(item["type"] == "audio" for item in messages)
            assert audio_count == len(sentences)
            asyncio.get_running_loop().call_later(
                0,
                message_handler.handle_message,
                client_uid,
                {"type": "frontend-playback-complete"},
            )

    response = await process_single_conversation(
        context=context,
        websocket_send=websocket_send,
        client_uid=client_uid,
        user_input="hello",
        images=None,
        session_emoji="test",
    )

    types = [message["type"] for message in messages]
    audio_texts = [
        message["display_text"]["text"]
        for message in messages
        if message["type"] == "audio"
    ]
    assert response == "".join(sentences)
    assert audio_texts == sentences
    assert types.count("backend-synth-complete") == 1
    assert "force-new-message" in types
    assert {"type": "control", "text": "conversation-chain-end"} in messages
    message_handler.cleanup_client(client_uid)
