"""
Shared configuration, constants, and global state for the Local Realtime API server.
"""

import asyncio
import base64
import logging
import os

from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Global peer / session registry
# ---------------------------------------------------------------------------
pcs = set()
active_sessions = {}

# Media keyed by session then id. Every image/video — client uploads, inline
# images, and auto-tracked frames — is stored here and served back to the
# client via /api/media/{session_id}/{media_id}; the LLM receives it inline as
# a base64 data URL.
uploaded_media = {}


def media_endpoint(session_id: str, media_id: str) -> str:
    """Return the playback URL for a stored media blob."""
    return f"/api/media/{session_id}/{media_id}"


def store_media(session_id: str, media_id: str, mime: str, data: bytes) -> str:
    """Store a media blob under its session and return its id."""
    uploaded_media.setdefault(session_id, {})[media_id] = {"mime": mime, "data": data}
    return media_id


def store_data_url(session_id: str, media_id: str, data_url: str) -> str:
    """Parse a base64 data URL, store its raw bytes, and return the id."""
    head, _, b64 = data_url.partition(",")
    mime = head.split(":", 1)[1].split(";", 1)[0] if ":" in head else "application/octet-stream"
    uploaded_media.setdefault(session_id, {})[media_id] = {"mime": mime, "data": base64.b64decode(b64)}
    return media_id


def resolve_media_url(session_id: str, url: str) -> str:
    """Expand a stored media id into a base64 data URL (kept for playback)."""
    entry = uploaded_media.get(session_id, {}).get(url)
    if entry:
        # Keep only the bare media type — vLLM's data URL parser splits on the
        # first ';' and requires the remainder to be exactly "base64", so a
        # codec parameter ("video/webm;codecs=vp9") would break it.
        mime = entry["mime"].split(";")[0]
        b64 = base64.b64encode(entry["data"]).decode("ascii")
        return f"data:{mime};base64,{b64}"
    return url

# ---------------------------------------------------------------------------
# Default system prompt (output goes to TTS — no markdown)
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your responses will be spoken aloud using "
    "text-to-speech, so keep them concise and conversational. Do not use markdown "
    "formatting, bullet points, numbered lists, or code blocks. "
    "Speak naturally as you would in a phone conversation."
)

# ---------------------------------------------------------------------------
# Reasoning effort → vLLM kwargs
# ---------------------------------------------------------------------------
REASONING_KWARGS = {
    "none":    {"chat_template_kwargs": {"enable_thinking": False}},
    "minimum": {"thinking_token_budget": 256},
    "low":     {"thinking_token_budget": 512},
    "medium":  {"thinking_token_budget": 1024},
    "high":    {"thinking_token_budget": 2048},
    "xhigh":   {},
}

# ---------------------------------------------------------------------------
# Whisper (STT) — configurable via environment
# ---------------------------------------------------------------------------
# WHISPER_MODEL accepts any value that faster-whisper's model_size_or_path
# parameter accepts: a HuggingFace repo id (e.g. "Systran/faster-whisper-large-v3")
# or a shorthand size ("tiny", "base", "small", "medium", "large-v3", …).
# Models are auto-downloaded from HuggingFace on first use.
#
# WHISPER_COMPUTE defaults to "auto", which picks the best available type
# for the device (float16 on CUDA, int8 on CPU).  Override only if you need
# a specific precision (e.g. "float32", "int8_float16").
_whisper_model = os.environ.get("WHISPER_MODEL", "small")
_whisper_device = os.environ.get("WHISPER_DEVICE", "cuda")
_whisper_compute = os.environ.get("WHISPER_COMPUTE", "auto")
_whisper_instances = max(1, int(os.environ.get("WHISPER_INSTANCES", "1")))

logging.info("Loading %d Faster-Whisper instance(s) (%s) on %s/%s …",
             _whisper_instances, _whisper_model, _whisper_device, _whisper_compute)
# Each instance holds its own copy of the weights, so N instances cost N× VRAM.
# They are handed out through whisper_pool (an asyncio.Queue), which guarantees
# a single instance is never used by two transcriptions at the same time.
whisper_instances = [WhisperModel(_whisper_model, device=_whisper_device,
                                  compute_type=_whisper_compute)
                     for _ in range(_whisper_instances)]
whisper_pool = asyncio.Queue()
for model in whisper_instances:
    whisper_pool.put_nowait(model)

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
MAX_HISTORY_LENGTH = 50

WEBRTC_SAMPLE_RATE      = 24000       # Hz — target output rate
WEBRTC_FRAME_DURATION   = 0.02        # seconds (20 ms per frame)
WHISPER_BEAM_SIZE       = 5           # faster-whisper beam width
SILERO_SAMPLE_RATE      = 16000       # Hz — VAD native rate
SILERO_CHUNK_SIZE       = 512         # samples per VAD chunk (32 ms @ 16 kHz)
SILERO_CHUNK_MS         = 32          # ms per VAD chunk
INT16_TO_FLOAT          = 1.0 / 32768.0
BYTES_PER_SAMPLE        = 2           # 16-bit mono
PRE_SPEECH_BUFFER_CHUNKS = 25         # ~800 ms of look-back audio
SSE_PREFIX_LENGTH       = 6           # len("data: ")

# ---------------------------------------------------------------------------
# External API endpoints
# ---------------------------------------------------------------------------
# Base URLs (defaults from environment). Clients may override these per-session
# by sending llm_base_url / tts_base_url in session.update. API keys are always
# supplied by the client — no global/backend key is used.
LLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:5000/v1")
TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://127.0.0.1:5000/v1")


def llm_chat_api(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def llm_models_api(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def tts_ws_api(base_url: str) -> str:
    ws = base_url.replace("http://", "ws://").replace("https://", "wss://")
    return f"{ws.rstrip('/')}/audio/speech/stream"


def tts_voices_api(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/audio/voices"


# ---------------------------------------------------------------------------
# Monotonic event-ID counter
# ---------------------------------------------------------------------------
_id_counter = 0
