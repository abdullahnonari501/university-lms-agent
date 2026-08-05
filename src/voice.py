"""Speech in and out, both running locally.

Speech-to-text is Whisper via transformers (already installed, so no new
dependency). Text-to-speech is Piper, a small ONNX voice that runs fast on CPU
and leaves the GPU to the LLM.

Nothing here calls out to a network service: student audio and answers stay on
this box, which is the same reason the rest of the stack is self-hosted.

Audio decoding deliberately avoids ffmpeg -- it is absent and sudo is
restricted, so the browser sends 16 kHz mono WAV that Python's stdlib `wave`
module reads directly.
"""

import io
import shutil
import subprocess
import sys
import threading
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS = REPO_ROOT / "models"

WHISPER_MODEL = "openai/whisper-small"
# Whisper has never seen this institution's vocabulary and renders "GIKI" as
# "Chiki". A prompt biases decoding toward the domain's proper nouns -- the
# words students say most and the ones retrieval most needs spelled right.
# Only distinctive terms belong here. Adding the faculty acronyms FES/FEE/FME
# made Whisper hear "fees" as "FES" -- biasing toward a rare word that collides
# with a common one costs more than it gains, and "fees" is the single most
# asked-about word in this corpus.
STT_VOCAB = (
    "GIKI, GIK Institute, Ghulam Ishaq Khan Institute, Topi, Swabi, "
    "FCSE, CGPA, SGPA, semester, prospectus, convocation, hostel, "
    "Rector, Dean, HEC, transcript, scholarship, admission."
)
PIPER_VOICE = MODELS / "en_US-lessac-medium.onnx"

_stt_lock = threading.Lock()
_stt = {"processor": None, "model": None, "device": None}


# ------------------------------------------------------------------ speech in

def stt_ready() -> bool:
    return _stt["model"] is not None


def load_stt() -> None:
    """Load Whisper once, on first use.

    Kept on the GPU only if there is room: qwen already holds ~9 of 12.3 GB, and
    an OOM here would take the chat down with it, so CPU is the safe fallback.
    """
    global _stt
    with _stt_lock:
        if _stt["model"] is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        device = "cpu"
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            if free > 2.5 * 1024**3:
                device = "cuda"

        processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
        model.to(device).eval()
        _stt = {"processor": processor, "model": model, "device": device}
        print(f"  [voice] whisper loaded on {device}", flush=True)


def read_wav(data: bytes) -> tuple[list[float], int]:
    """Decode WAV to mono float samples using only the standard library."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        channels, width, rate = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {width * 8}-bit")

    import array

    samples = array.array("h")
    samples.frombytes(frames)
    if channels > 1:                       # average channels down to mono
        samples = array.array("h", [
            int(sum(samples[i:i + channels]) / channels)
            for i in range(0, len(samples) - channels + 1, channels)
        ])
    return [s / 32768.0 for s in samples], rate


def resample(samples: list[float], src: int, dst: int = 16000) -> list[float]:
    """Linear resample. Whisper wants 16 kHz; browsers commonly record 48 kHz."""
    if src == dst or not samples:
        return samples
    ratio = dst / src
    out_len = int(len(samples) * ratio)
    out = []
    for i in range(out_len):
        pos = i / ratio
        lo = int(pos)
        hi = min(lo + 1, len(samples) - 1)
        frac = pos - lo
        out.append(samples[lo] * (1 - frac) + samples[hi] * frac)
    return out


# Whisper emits "you" for silence -- a hallucination, not a transcription, and
# a deeply confusing one to see in a chat box. Detect the silence instead and
# say so, rather than passing a made-up word off as what the user said.
SILENCE_PEAK = 0.01


def audio_levels(samples: list[float]) -> tuple[float, float]:
    """(peak, rms) so a caller can tell silence from speech."""
    if not samples:
        return 0.0, 0.0
    peak = max(abs(s) for s in samples)
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
    return peak, rms


class SilentAudio(Exception):
    """Raised when the recording contains no audible signal."""

    def __init__(self, peak: float, rms: float, seconds: float):
        super().__init__(f"peak={peak:.4f} rms={rms:.4f} {seconds:.1f}s")
        self.peak, self.rms, self.seconds = peak, rms, seconds


def transcribe(wav_bytes: bytes) -> str:
    load_stt()
    samples, rate = read_wav(wav_bytes)
    seconds = len(samples) / rate if rate else 0.0
    peak, rms = audio_levels(samples)
    print(f"  [voice] audio {seconds:.1f}s peak={peak:.4f} rms={rms:.4f} "
          f"rate={rate}", flush=True)

    if len(samples) < rate * 0.2:          # under 200 ms is a stray click
        raise SilentAudio(peak, rms, seconds)
    if peak < SILENCE_PEAK:
        raise SilentAudio(peak, rms, seconds)
    audio = resample(samples, rate)

    import torch

    processor, model, device = _stt["processor"], _stt["model"], _stt["device"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    features = inputs.input_features.to(device)

    kwargs = {"language": "en", "task": "transcribe", "max_new_tokens": 200}
    try:
        kwargs["prompt_ids"] = processor.get_prompt_ids(STT_VOCAB, return_tensors="pt").to(device)
    except Exception:  # noqa: BLE001 - biasing is a bonus, never a requirement
        pass

    with torch.no_grad():
        ids = model.generate(features, **kwargs)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    # The prompt is echoed back by some versions; drop it if so.
    if text.startswith(STT_VOCAB[:40]):
        text = text[len(STT_VOCAB):].strip()
    return text


# ----------------------------------------------------------------- speech out

def piper_binary() -> str | None:
    for candidate in (MODELS / "piper" / "piper", MODELS / "piper"):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("piper")


def tts_ready() -> bool:
    return bool(piper_binary()) and PIPER_VOICE.exists()


def speak(text: str, max_chars: int = 2000) -> bytes:
    """Render text to WAV bytes with Piper. Returns b'' when unavailable."""
    binary = piper_binary()
    if not binary or not PIPER_VOICE.exists():
        return b""

    # Strip the citation list and mode flag -- reading URLs aloud is useless.
    spoken = text.split("\nSources:")[0].replace("⚠", "").strip()[:max_chars]
    if not spoken:
        return b""

    out = MODELS / f".tts_{threading.get_ident()}.wav"
    try:
        subprocess.run(
            [binary, "--model", str(PIPER_VOICE), "--output_file", str(out)],
            input=spoken.encode("utf-8"),
            capture_output=True, timeout=180, check=True,
        )
        return out.read_bytes() if out.exists() else b""
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"  [voice] piper failed: {exc}", file=sys.stderr, flush=True)
        return b""
    finally:
        out.unlink(missing_ok=True)


if __name__ == "__main__":
    print("stt model :", WHISPER_MODEL)
    print("piper bin :", piper_binary())
    print("voice file:", PIPER_VOICE, "exists" if PIPER_VOICE.exists() else "MISSING")
    print("tts ready :", tts_ready())
