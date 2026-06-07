# RealtimeRTC

**OpenAI Realtime API-compatible voice conversation server over WebRTC.**

Real-time pipeline: microphone → VAD → STT (Whisper) → LLM (vLLM) → TTS → speaker.

## Architecture

```
Browser ──WebRTC──▶ Server (realtimertc)
                      │
                      ├─ SileroVAD        (speech detection)
                      ├─ faster-whisper   (transcription)
                      ├─ vLLM HTTP/SSE    (LLM inference)
                      └─ vLLM WebSocket   (TTS streaming)
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Start vLLM

The LLM and TTS backends must be running at `http://127.0.0.1:5000` (configurable via environment variables).

### 3. Run the server

```bash
cd realtimertc
python main.py --host 0.0.0.0 --port 8081
```

### 4. Open the client

Navigate to `http://localhost:8081` — the WebRTC client UI loads automatically.

## Docker

```bash
# Build (CUDA GPU)
docker build -t realtimertc .

# Build (CPU only)
docker build --build-arg BASE=python:3.11-slim -t realtimertc .

# Run (GPU)
docker run --gpus all -p 8081:8081 \
  -e VLLM_BASE_URL=http://host.docker.internal:5000/v1 \
  -e TTS_BASE_URL=http://host.docker.internal:5000/v1 \
  realtimertc

# Or use docker-compose
docker compose up -d
```

> **Note:** `host.docker.internal` resolves to the host machine.  Use `172.17.0.1` on Linux if unavailable.
> Models are auto-downloaded from HuggingFace on first run and cached in the container.

## Configuration

Copy and edit the example file, then run (vars live only for this command, not permanent):

```bash
cp .env.example .env
# edit .env …
set -a && source .env && set +a && python -m realtimertc.main
```

| Environment Variable | Default |
|---|---|
| `VLLM_BASE_URL` | `http://127.0.0.1:5000/v1` |
| `TTS_BASE_URL` | `http://127.0.0.1:5000/v1` |
| `WHISPER_MODEL` | `small` |
| `WHISPER_DEVICE` | `cuda` |
| `WHISPER_COMPUTE` | `auto` |

## API Compatibility

Implements the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) event protocol over an `oai-events` WebRTC DataChannel.

### Supported Client Events

`session.update` · `input_audio_buffer.commit` · `input_audio_buffer.clear` · `output_audio_buffer.clear` · `conversation.item.create` · `conversation.item.retrieve` · `conversation.item.delete` · `conversation.item.truncate` · `response.create` · `response.cancel`

### Supported Server Events

`session.created` · `session.updated` · `conversation.item.added` · `conversation.item.done` · `conversation.item.created` · `conversation.item.retrieved` · `conversation.item.deleted` · `conversation.item.truncated` · `conversation.item.input_audio_transcription.completed` · `conversation.item.input_audio_transcription.failed` · `response.created` · `response.done` · `response.output_audio_transcript.delta` · `response.output_audio_transcript.done` · `response.output_text.delta` · `response.output_text.done` · `response.output_reasoning.delta` · `response.output_reasoning.done` · `response.function_call_arguments.delta` · `response.function_call_arguments.done` · `input_audio_buffer.speech_started` · `input_audio_buffer.speech_stopped` · `input_audio_buffer.committed` · `input_audio_buffer.cleared` · `output_audio_buffer.cleared` · `error`

## Project Structure

```
realtimertc/
├── main.py         Entry point, HTTP routes, lifecycle
├── config.py       Constants, globals, Whisper init
├── webrtc.py       WebRTC handler + OpenAI event router
├── realtime.py     Core pipeline (VAD, STT, LLM+TTS)
├── audio.py        ByteQueue, audio track, TTS receiver
├── cache.py        Background voice/model polling
├── utils.py        Helpers (ID gen, history trim, cleanup)
└── html/
    └── index.html  WebRTC client UI (Tailwind + vanilla JS)
```

## Features

- **Server VAD** (auto-detect speech) or **Push-to-Talk** modes
- **Barge-in** — interrupts AI playback when user speaks
- **Reasoning tokens** — chain-of-thought display for DeepSeek-R1 / Qwen3
- **Tool calling** — streaming function call arguments
- **Multi-voice** — dynamic TTS voice switching
- **History trimming** — keeps last 50 messages to prevent context overflow

## Requirements

- Python 3.11+
- CUDA-capable GPU (for Whisper)
- vLLM serving a compatible LLM + TTS model
