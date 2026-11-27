"""Tests for incremental sentence segmentation behavior and caching."""

import pytest

import src.open_llm_vtuber.utils.sentence_divider as sentence_divider_module
from src.open_llm_vtuber.utils.sentence_divider import (
    SentenceDivider,
    segment_text_by_regex,
)


async def collect_sentences(chunks: list[str], method: str = "pysbd") -> list[str]:
    """Collect text emitted by a sentence divider for test chunks."""

    async def stream():
        for chunk in chunks:
            yield chunk

    divider = SentenceDivider(faster_first_response=False, segment_method=method)
    return [sentence.text async for sentence in divider.process_stream(stream())]


@pytest.mark.asyncio
async def test_sentence_divider_detects_language_once_per_stream(monkeypatch) -> None:
    """Incremental segmentation does not repeatedly pay language-detection cost."""
    calls = 0
    real_detect = sentence_divider_module.detect

    def counting_detect(text: str) -> str:
        nonlocal calls
        calls += 1
        return real_detect(text)

    monkeypatch.setattr(sentence_divider_module, "detect", counting_detect)

    sentences = await collect_sentences(
        ["Hello world. ", "This is Dr. Smith speaking. ", "Finally done!"]
    )

    assert calls == 1
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[-1] == "Finally done!"


@pytest.mark.asyncio
async def test_sentence_divider_reset_allows_language_redetection(monkeypatch) -> None:
    """A new stream gets a fresh language decision rather than stale state."""
    calls = 0

    def counting_detect(text: str) -> str:
        nonlocal calls
        calls += 1
        return "en"

    monkeypatch.setattr(sentence_divider_module, "detect", counting_detect)

    await collect_sentences(["One. Two."])
    await collect_sentences(["Three. Four."])

    assert calls == 2


def test_regex_segmentation_returns_complete_sentences_and_remainder() -> None:
    """The regex fallback emits complete punctuation-terminated sentences."""
    sentences, remainder = segment_text_by_regex("First! Second? trailing")

    assert sentences == ["First!", "Second?"]
    assert remainder == "trailing"
