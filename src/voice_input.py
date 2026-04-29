import io
import numpy as np
import scipy.io.wavfile as wavfile


def transcribe_audio(audio_bytes: bytes, whisper_pipeline) -> str:
    """Transcribe WAV audio bytes to text using the provided Whisper pipeline."""
    sr, data = wavfile.read(io.BytesIO(audio_bytes))

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Normalize to float32 in [-1, 1]
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    result = whisper_pipeline({"raw": data, "sampling_rate": sr})
    return result["text"].strip()
