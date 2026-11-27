"""Reproducible benchmarks for latency-sensitive Open-LLM-VTuber hot paths.

The benchmark measures backend orchestration cost with deterministic local inputs. It
does not include model inference, network access, or browser playback time.
"""

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger.remove()


def _median_time(callable_object, repeat: int) -> tuple[float, list[float]]:
    """Run a synchronous callable repeatedly and return median and raw milliseconds.

    Args:
        callable_object: Zero-argument callable to measure.
        repeat: Number of repetitions.

    Returns:
        A tuple of median milliseconds and all measured milliseconds.
    """
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        callable_object()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), samples


def benchmark_audio_ingestion(chunks: int, samples_per_chunk: int, repeat: int) -> dict:
    """Benchmark quadratic legacy audio buffering against the production buffer.

    Args:
        chunks: Number of incoming audio chunks.
        samples_per_chunk: Number of float32 samples in each chunk.
        repeat: Number of repetitions.

    Returns:
        Benchmark measurements in milliseconds.
    """
    source_chunk = np.full(samples_per_chunk, 0.25, dtype=np.float32)

    def legacy_np_append() -> int:
        buffer = np.array([])
        for _ in range(chunks):
            buffer = np.append(buffer, source_chunk)
        return int(buffer.size)

    legacy_median, legacy_samples = _median_time(legacy_np_append, repeat)

    production_median = None
    production_samples = None
    production_samples_written = None
    try:
        from src.open_llm_vtuber.utils.audio_buffer import AudioBuffer

        def production_append() -> int:
            buffer = AudioBuffer()
            for _ in range(chunks):
                buffer.append_float32(source_chunk)
            return int(buffer.consume().size)

        production_median, production_samples = _median_time(production_append, repeat)
        production_samples_written = production_append()
    except ImportError:
        production_samples_written = None

    return {
        "workload": {
            "chunks": chunks,
            "samples_per_chunk": samples_per_chunk,
            "total_samples": chunks * samples_per_chunk,
            "equivalent_seconds_at_16khz": chunks * samples_per_chunk / 16000,
        },
        "legacy_np_append_ms": legacy_median,
        "legacy_np_append_samples_ms": legacy_samples,
        "production_audio_buffer_ms": production_median,
        "production_audio_buffer_samples_ms": production_samples,
        "production_audio_buffer_samples_written": production_samples_written,
        "legacy_result_samples": legacy_np_append(),
    }


async def _run_sentence_divider_once(text: str, method: str) -> list[str]:
    """Run one deterministic sentence-divider workload.

    Args:
        text: Source text to stream.
        method: Segmentation method name.

    Returns:
        Produced sentence texts.
    """
    from src.open_llm_vtuber.utils.sentence_divider import SentenceDivider

    async def token_stream():
        for token in text.split(" "):
            yield token + " "

    divider = SentenceDivider(faster_first_response=False, segment_method=method)
    output = []
    async for sentence in divider.process_stream(token_stream()):
        output.append(sentence.text)
    return output


def benchmark_sentence_divider(repeat: int) -> dict:
    """Benchmark incremental sentence segmentation for a representative response.

    Args:
        repeat: Number of repetitions.

    Returns:
        Benchmark measurements in milliseconds and output counts.
    """
    text = (
        "Hello there. Dr. Smith continues speaking without stopping. "
        "Finally we finish now! "
    ) * 3
    results = {}
    for method in ("pysbd", "regex"):
        samples = []
        output_count = 0
        for _ in range(repeat):
            start = time.perf_counter()
            output = asyncio.run(_run_sentence_divider_once(text, method))
            samples.append((time.perf_counter() - start) * 1000)
            output_count = len(output)
        results[method] = {
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
            "output_count": output_count,
        }
    return {
        "workload": {"characters": len(text), "method_repeat": repeat},
        "results": results,
    }


def write_test_wav(path: Path, seconds: float = 8.0, sample_rate: int = 16000) -> None:
    """Write a deterministic mono 16-bit PCM WAV file.

    Args:
        path: Destination path.
        seconds: Audio duration in seconds.
        sample_rate: Audio sample rate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = int(seconds * sample_rate)
    time_values = np.arange(sample_count, dtype=np.float32) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * 440 * time_values) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


async def _measure_event_loop_lag(operation, tick_seconds: float = 0.005) -> dict:
    """Measure event-loop tick delay while an operation runs.

    Args:
        operation: Awaitable operation to measure.
        tick_seconds: Nominal interval between event-loop ticks.

    Returns:
        Tick latency statistics in milliseconds.
    """
    tick_intervals = []
    stop_ticker = False

    async def ticker() -> None:
        last = time.perf_counter()
        while not stop_ticker:
            await asyncio.sleep(tick_seconds)
            now = time.perf_counter()
            tick_intervals.append((now - last) * 1000)
            last = now

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(tick_seconds * 2)
    start = time.perf_counter()
    await operation
    operation_ms = (time.perf_counter() - start) * 1000
    stop_ticker = True
    await ticker_task

    return {
        "operation_ms": operation_ms,
        "tick_count": len(tick_intervals),
        "max_tick_interval_ms": max(tick_intervals, default=0.0),
        "mean_tick_interval_ms": (
            statistics.mean(tick_intervals) if tick_intervals else 0.0
        ),
    }


def benchmark_audio_payload(repeat: int) -> dict:
    """Benchmark conversion of generated audio into the WebSocket audio payload.

    Args:
        repeat: Number of repetitions.

    Returns:
        Direct conversion cost and event-loop responsiveness measurements.
    """
    from src.open_llm_vtuber.utils.stream_audio import prepare_audio_payload

    try:
        from src.open_llm_vtuber.utils.stream_audio import async_prepare_audio_payload
    except ImportError:
        async_prepare_audio_payload = None

    wav_path = ROOT / "cache" / "benchmark_tone.wav"
    write_test_wav(wav_path, seconds=8.0)

    direct_median, direct_samples = _median_time(
        lambda: prepare_audio_payload(str(wav_path)), repeat
    )

    lag_results = []
    for _ in range(repeat):
        if async_prepare_audio_payload is None:

            async def blocking_operation() -> None:
                prepare_audio_payload(str(wav_path))

            lag_results.append(
                asyncio.run(_measure_event_loop_lag(blocking_operation()))
            )
        else:
            lag_results.append(
                asyncio.run(
                    _measure_event_loop_lag(async_prepare_audio_payload(str(wav_path)))
                )
            )

    return {
        "workload": {"wav_seconds": 8.0, "repeat": repeat},
        "direct_prepare_ms": direct_median,
        "direct_prepare_samples_ms": direct_samples,
        "event_loop_lag": lag_results,
        "uses_async_payload_preparation": async_prepare_audio_payload is not None,
    }


def benchmark_json_detector(repeat: int) -> dict:
    """Benchmark streaming MCP JSON detection with nested objects and strings.

    Args:
        repeat: Number of repetitions.

    Returns:
        Processing cost and number of detected JSON objects.
    """
    from src.open_llm_vtuber.mcpp.json_detector import StreamJSONDetector

    text = (
        'prefix {"tool":"search","arguments":{"query":"braces } { inside",'
        '"nested":{"x":1}}} suffix'
    )
    chunk_size = 7
    chunks = [
        text[index : index + chunk_size] for index in range(0, len(text), chunk_size)
    ]

    def run_once() -> int:
        detector = StreamJSONDetector()
        detected = 0
        for chunk in chunks:
            detected += len(detector.process_chunk(chunk))
        return detected

    median, samples = _median_time(run_once, repeat)
    return {
        "workload": {"characters": len(text), "chunks": len(chunks), "repeat": repeat},
        "median_ms": median,
        "samples_ms": samples,
        "detected_json_count": run_once(),
        "expected_json_count": 1,
    }


class BenchmarkTTS:
    """Small TTS engine that deterministically writes local WAV files."""

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """Generate a short WAV file for benchmark text.

        Args:
            text: Text to synthesize.
            file_name_no_ext: Cache file basename.

        Returns:
            Generated WAV path.
        """
        cache_dir = ROOT / "cache"
        cache_dir.mkdir(exist_ok=True)
        path = cache_dir / f"{file_name_no_ext or 'benchmark'}.wav"
        write_test_wav(path, seconds=0.4)
        return str(path)

    async def async_generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """Asynchronously generate audio through a worker thread.

        Args:
            text: Text to synthesize.
            file_name_no_ext: Cache file basename.

        Returns:
            Generated WAV path.
        """
        return await asyncio.to_thread(self.generate_audio, text, file_name_no_ext)

    def remove_file(self, filepath: str, verbose: bool = True) -> None:
        """Remove a generated file if it exists.

        Args:
            filepath: File to remove.
            verbose: Whether verbose logging is requested.
        """
        try:
            os.remove(filepath)
        except FileNotFoundError:
            pass


class BenchmarkAgent:
    """Agent stub that emits deterministic sentence outputs."""

    def __init__(self, sentences: list[str]):
        """Initialize the agent stub.

        Args:
            sentences: Sentences to emit.
        """
        self.sentences = sentences

    def chat(self, batch_input):
        """Return an async stream of deterministic sentence outputs.

        Args:
            batch_input: Conversation input.

        Returns:
            Async iterator of sentence outputs.
        """
        from src.open_llm_vtuber.agent.output_types import (
            Actions,
            DisplayText,
            SentenceOutput,
        )

        async def stream():
            for sentence in self.sentences:
                yield SentenceOutput(
                    display_text=DisplayText(text=sentence),
                    tts_text=sentence,
                    actions=Actions(),
                )

        return stream()


async def benchmark_conversation_turn_async() -> dict:
    """Measure backend-only latency for a deterministic text conversation turn.

    Returns:
        First-audio and completion timings in milliseconds.
    """
    from src.open_llm_vtuber.conversations.single_conversation import (
        process_single_conversation,
    )
    from src.open_llm_vtuber.message_handler import message_handler

    sentences = ["First sentence.", "Second sentence.", "Third sentence."]
    context = SimpleNamespace(
        asr_engine=None,
        agent_engine=BenchmarkAgent(sentences),
        live2d_model=None,
        tts_engine=BenchmarkTTS(),
        translate_engine=None,
        character_config=SimpleNamespace(
            human_name="Human",
            character_name="BenchmarkAI",
            avatar=None,
            conf_uid="benchmark",
        ),
        history_uid=None,
    )

    start_time = time.perf_counter()
    messages = []

    async def websocket_send(message: str) -> None:
        payload = json.loads(message)
        now_ms = (time.perf_counter() - start_time) * 1000
        payload["_received_ms"] = now_ms
        messages.append(payload)
        if payload.get("type") == "backend-synth-complete":
            asyncio.get_running_loop().call_later(
                0,
                message_handler.handle_message,
                "benchmark-client",
                {"type": "frontend-playback-complete"},
            )

    start_time = time.perf_counter()
    response = await process_single_conversation(
        context=context,
        websocket_send=websocket_send,
        client_uid="benchmark-client",
        user_input="benchmark input",
        images=None,
        session_emoji="benchmark",
    )
    completion_ms = (time.perf_counter() - start_time) * 1000
    audio_messages = [message for message in messages if message.get("type") == "audio"]

    return {
        "sentence_count": len(sentences),
        "audio_message_count": len(audio_messages),
        "first_audio_ms": audio_messages[0]["_received_ms"] if audio_messages else None,
        "completion_ms": completion_ms,
        "response_characters": len(response),
        "backend_only_excludes": [
            "LLM inference",
            "ASR inference",
            "network transport",
            "browser playback duration",
        ],
    }


def benchmark_conversation_turn(repeat: int) -> dict:
    """Repeat the deterministic backend-only conversation benchmark.

    Args:
        repeat: Number of repetitions.

    Returns:
        Median and raw measurements.
    """
    runs = [asyncio.run(benchmark_conversation_turn_async()) for _ in range(repeat)]
    first_audio_values = [
        run["first_audio_ms"] for run in runs if run["first_audio_ms"] is not None
    ]
    return {
        "repeat": repeat,
        "median_first_audio_ms": (
            statistics.median(first_audio_values) if first_audio_values else None
        ),
        "median_completion_ms": statistics.median(run["completion_ms"] for run in runs),
        "runs": runs,
    }


def get_git_revision() -> str:
    """Return the current git revision, or an error marker when unavailable.

    Returns:
        Short git commit hash or `unknown`.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_benchmarks(args: argparse.Namespace) -> dict:
    """Run all selected benchmarks.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Complete benchmark result dictionary.
    """
    return {
        "metadata": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_revision": get_git_revision(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "measurement_scope": "deterministic backend hot paths; external model and network latency excluded",
        },
        "audio_ingestion": benchmark_audio_ingestion(
            chunks=args.audio_chunks,
            samples_per_chunk=args.samples_per_chunk,
            repeat=args.repeat,
        ),
        "sentence_divider": benchmark_sentence_divider(repeat=args.repeat),
        "audio_payload": benchmark_audio_payload(repeat=args.repeat),
        "json_detector": benchmark_json_detector(repeat=args.repeat),
        "conversation_turn": benchmark_conversation_turn(repeat=args.repeat),
    }


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat", type=int, default=5, help="repetitions per benchmark"
    )
    parser.add_argument("--audio-chunks", type=int, default=1000)
    parser.add_argument("--samples-per-chunk", type=int, default=1600)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run benchmarks and write or print JSON results."""
    args = parse_args()
    results = run_benchmarks(args)
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
