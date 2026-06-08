"""
Shared configuration, constants, and global state for the Local Realtime API server.
"""

import asyncio
import logging
import os

from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Global peer / session registry
# ---------------------------------------------------------------------------
pcs = set()
active_sessions = {}

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

logging.info("Loading Faster-Whisper model (%s) on %s/%s …",
             _whisper_model, _whisper_device, _whisper_compute)
whisper_model = WhisperModel(_whisper_model, device=_whisper_device, compute_type=_whisper_compute)
whisper_queue_lock = asyncio.Semaphore(1)  # serialise GPU access

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
# External API endpoints (configurable via environment)
# ---------------------------------------------------------------------------
_VLLM_BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:5000/v1")
LLM_API        = os.environ.get("LLM_API",        f"{_VLLM_BASE}/chat/completions")
LLM_MODELS_API = os.environ.get("LLM_MODELS_API", f"{_VLLM_BASE}/models")
LLM_API_KEY    = os.environ.get("LLM_API_KEY", "")

_TTS_BASE = os.environ.get("TTS_BASE_URL", "http://127.0.0.1:5000/v1")
TTS_WS_API     = os.environ.get("TTS_WS_API",
                     _TTS_BASE.replace("http://", "ws://") + "/audio/speech/stream")
TTS_LIST_VOICES = os.environ.get("TTS_LIST_VOICES",
                     f"{_TTS_BASE}/audio/voices")
TTS_API_KEY    = os.environ.get("TTS_API_KEY", "")

# ---------------------------------------------------------------------------
# Cached model / voice lists (refreshed periodically in the background)
# ---------------------------------------------------------------------------
AVAILABLE_VOICES = []
DEFAULT_VOICE = ""
AVAILABLE_MODELS = []
DEFAULT_MODEL = ""

# ---------------------------------------------------------------------------
# Monotonic event-ID counter
# ---------------------------------------------------------------------------
_id_counter = 0
