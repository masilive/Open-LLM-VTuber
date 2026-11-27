"""Efficient chunked audio accumulation for realtime microphone input."""

import numpy as np


class AudioBuffer:
    """Accumulate audio chunks without repeatedly copying prior samples.

    The buffer stores chunks independently and concatenates them once when the
    complete audio turn is consumed. Appending therefore has linear total cost
    rather than the quadratic cost of repeated ``numpy.append`` calls.
    """

    def __init__(self) -> None:
        """Initialize an empty audio buffer."""
        self._chunks: list[np.ndarray] = []
        self._length = 0

    def append_float32(self, samples) -> None:
        """Append mono float32 audio samples.

        Args:
            samples: One-dimensional array-like audio samples.
        """
        array = np.array(samples, dtype=np.float32, copy=True).reshape(-1)
        if array.size == 0:
            return
        self._chunks.append(array)
        self._length += int(array.size)

    def append_int16_bytes(self, audio_bytes: bytes) -> None:
        """Append 16-bit PCM bytes normalized to float32 samples.

        Args:
            audio_bytes: Little-endian signed 16-bit PCM audio bytes.

        Raises:
            ValueError: If the byte length is not aligned to 16-bit samples.
        """
        if len(audio_bytes) % 2:
            raise ValueError("16-bit PCM audio byte length must be even")
        int_samples = np.frombuffer(audio_bytes, dtype=np.int16)
        float_samples = int_samples.astype(np.float32) / 32768.0
        self.append_float32(float_samples)

    def consume(self) -> np.ndarray:
        """Return all buffered audio and reset the buffer.

        Returns:
            A float32 NumPy array containing every sample in append order.
        """
        if not self._chunks:
            return np.array([], dtype=np.float32)
        if len(self._chunks) == 1:
            result = self._chunks[0]
        else:
            result = np.concatenate(self._chunks).astype(np.float32, copy=False)
        self.clear()
        return result

    def clear(self) -> None:
        """Discard all buffered audio."""
        self._chunks.clear()
        self._length = 0

    def __len__(self) -> int:
        """Return the number of buffered samples.

        Returns:
            Number of audio samples currently buffered.
        """
        return self._length
