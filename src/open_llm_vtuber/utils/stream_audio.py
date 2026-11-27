import asyncio
import base64
from typing import Any

from pydub import AudioSegment
from pydub.utils import make_chunks
from ..agent.output_types import Actions
from ..agent.output_types import DisplayText


def _get_volume_by_chunks(audio: AudioSegment, chunk_length_ms: int) -> list:
    """
    Calculate the normalized volume (RMS) for each chunk of the audio.

    Parameters:
        audio (AudioSegment): The audio segment to process.
        chunk_length_ms (int): The length of each audio chunk in milliseconds.

    Returns:
        list: Normalized volumes for each chunk.
    """
    chunks = make_chunks(audio, chunk_length_ms)
    volumes = [chunk.rms for chunk in chunks]
    max_volume = max(volumes)
    if max_volume == 0:
        raise ValueError("Audio is empty or all zero.")
    return [volume / max_volume for volume in volumes]


def prepare_audio_payload(
    audio_path: str | None,
    chunk_length_ms: int = 20,
    display_text: DisplayText = None,
    actions: Actions = None,
    forwarded: bool = False,
) -> dict[str, Any]:
    """
    Prepares the audio payload for sending to a broadcast endpoint.
    If audio_path is None, returns a payload with audio=None for silent display.

    Parameters:
        audio_path (str | None): The path to the audio file to be processed, or None for silent display
        chunk_length_ms (int): The length of each audio chunk in milliseconds
        display_text (DisplayText, optional): Text to be displayed with the audio
        actions (Actions, optional): Actions associated with the audio

    Returns:
        dict: The audio payload to be sent
    """
    if isinstance(display_text, DisplayText):
        display_text = display_text.to_dict()

    if not audio_path:
        # Return payload for silent display
        return {
            "type": "audio",
            "audio": None,
            "volumes": [],
            "slice_length": chunk_length_ms,
            "display_text": display_text,
            "actions": actions.to_dict() if actions else None,
            "forwarded": forwarded,
        }

    try:
        audio = AudioSegment.from_file(audio_path)
        audio_bytes = audio.export(format="wav").read()
    except Exception as e:
        raise ValueError(
            f"Error loading or converting generated audio file to wav file '{audio_path}': {e}"
        )
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    volumes = _get_volume_by_chunks(audio, chunk_length_ms)

    payload = {
        "type": "audio",
        "audio": audio_base64,
        "volumes": volumes,
        "slice_length": chunk_length_ms,
        "display_text": display_text,
        "actions": actions.to_dict() if actions else None,
        "forwarded": forwarded,
    }

    return payload


async def async_prepare_audio_payload(
    audio_path: str | None,
    chunk_length_ms: int = 20,
    display_text: DisplayText = None,
    actions: Actions = None,
    forwarded: bool = False,
) -> dict[str, Any]:
    """Prepare an audio payload without blocking the event loop.

    Args:
        audio_path: Path to the audio file, or ``None`` for silent display.
        chunk_length_ms: Length of each volume-analysis chunk in milliseconds.
        display_text: Text to display with the audio.
        actions: Actions associated with the audio.
        forwarded: Whether this payload is forwarded to another group member.

    Returns:
        The audio payload to send over the WebSocket.
    """
    if not audio_path:
        return prepare_audio_payload(
            audio_path=None,
            chunk_length_ms=chunk_length_ms,
            display_text=display_text,
            actions=actions,
            forwarded=forwarded,
        )
    return await asyncio.to_thread(
        prepare_audio_payload,
        audio_path,
        chunk_length_ms,
        display_text,
        actions,
        forwarded,
    )


# Example usage:
# payload, duration = prepare_audio_payload("path/to/audio.mp3", display_text="Hello", expression_list=[0,1,2])
