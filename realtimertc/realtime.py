"""
Core real-time pipeline: VAD → STT → LLM → TTS.
"""

import asyncio
import json
import logging
import traceback
from collections import deque

import av
import aiohttp
import numpy as np
from aiortc.mediastreams import MediaStreamError

from realtimertc.audio import LocalAIAudioTrack, receive_audio_from_tts
from realtimertc import config
from realtimertc.config import (
    INT16_TO_FLOAT,
    LLM_API,
    LLM_API_KEY,
    PRE_SPEECH_BUFFER_CHUNKS,
    REASONING_KWARGS,
    SILERO_CHUNK_MS,
    SILERO_CHUNK_SIZE,
    SILERO_SAMPLE_RATE,
    SSE_PREFIX_LENGTH,
    TTS_API_KEY,
    TTS_WS_API,
    WEBRTC_SAMPLE_RATE,
    WHISPER_BEAM_SIZE,
    whisper_model,
    whisper_queue_lock,
)
from realtimertc.utils import channel_open, generate_id, trim_history


def _llm_headers():
    """Build auth headers for LLM API calls."""
    return {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}


def _tts_headers():
    """Build auth headers for TTS API calls."""
    return {"Authorization": f"Bearer {TTS_API_KEY}"} if TTS_API_KEY else {}


# ---------------------------------------------------------------------------
# trigger_ai_response — stream LLM output through TTS back to the client
# ---------------------------------------------------------------------------
async def trigger_ai_response(session_id: str,
                               audio_track: LocalAIAudioTrack,
                               response_config: dict | None = None):
    session_data = config.active_sessions.get(session_id)
    if not session_data:
        return

    history = session_data["history"]
    session_config = session_data["config"]
    channel = session_data.get("channel")

    req_config = response_config or {}
    modalities = req_config.get("output_modalities",
                                session_config.get("output_modalities", ["audio"]))
    tools = req_config.get("tools", session_config.get("tools", []))
    voice = req_config.get("audio", {}).get("output", {}).get(
        "voice", session_config.get("audio", {}).get("output", {}).get("voice", config.DEFAULT_VOICE))
    model_name = req_config.get("model", session_config.get("model", "Qwen3.5-9B"))
    reasoning_effort = req_config.get("reasoning", {}).get(
        "effort", session_config.get("reasoning", {}).get("effort", "none"))
    reasoning_kwargs = REASONING_KWARGS.get(reasoning_effort, {})

    llm_payload = {"model": model_name, "messages": history, "stream": True,
                   **reasoning_kwargs}
    if tools:
        llm_payload["tools"] = tools

    full_ai_response = ""
    full_reasoning_response = ""
    tool_calls_buffer: dict = {}
    ws = None
    recv_task = None
    is_cancelled = False

    item_id = generate_id("item")
    response_id = generate_id("resp")
    audio_track.active_response_id = response_id

    logging.info("[%s] Streaming AI response. modalities=%s tools=%s reasoning=%s",
                 session_id, modalities, bool(tools), reasoning_effort)

    session = aiohttp.ClientSession()
    try:
        async with session.post(LLM_API, json=llm_payload,
                headers=_llm_headers()) as llm_resp:
            if llm_resp.status != 200:
                error_text = await llm_resp.text()
                logging.error("[%s] LLM API error %s: %s", session_id, llm_resp.status, error_text)
                if channel_open(channel):
                    channel.send(json.dumps({
                        "type": "error", "event_id": generate_id(),
                        "error": {"type": "llm_server_error",
                                  "message": f"LLM API Error ({llm_resp.status}): {error_text}"}}))
                    channel.send(json.dumps({
                        "type": "response.created", "event_id": generate_id("evt"),
                        "response": {"id": response_id, "object": "realtime.response",
                                     "status": "failed", "output_modalities": modalities}}))
                return

            has_init_tts = False

            # --- announce assistant item + response to client ---
            if channel_open(channel):
                channel.send(json.dumps({
                    "type": "conversation.item.added", "event_id": generate_id("evt"),
                    "item": {"id": item_id, "object": "realtime.item",
                             "type": "message", "role": "assistant",
                             "content": [], "status": "in_progress"}}))
                resp_obj = {"id": response_id, "object": "realtime.response",
                            "status": "in_progress", "output_modalities": modalities}
                if "audio" in modalities:
                    resp_obj["audio"] = {"output": {"format": {"type": "pcm", "rate": WEBRTC_SAMPLE_RATE},
                                                    "voice": voice}}
                channel.send(json.dumps({"type": "response.created",
                                         "event_id": generate_id("evt"),
                                         "response": resp_obj}))

            # --- SSE stream parsing ---
            async for line in llm_resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    chunk_data = json.loads(line[SSE_PREFIX_LENGTH:])
                    if not chunk_data.get("choices"):
                        continue
                    delta = chunk_data["choices"][0].get("delta", {})

                    # ---- reasoning ----
                    dr = delta.get("reasoning") or ""
                    if dr:
                        full_reasoning_response += dr
                        if channel_open(channel):
                            channel.send(json.dumps({
                                "type": "response.output_reasoning.delta",
                                "event_id": generate_id(),
                                "response_id": response_id, "delta": dr}))

                    # ---- text content ----
                    dt = delta.get("content") or ""
                    if dt:
                        if not has_init_tts:
                            has_init_tts = True
                            if "audio" in modalities:
                                try:
                                    ws = await session.ws_connect(TTS_WS_API,
                                            headers=_tts_headers())
                                    await ws.send_json({"type": "session.config",
                                                        "voice": voice,
                                                        "response_format": "pcm",
                                                        "stream_audio": True})
                                    recv_task = asyncio.create_task(
                                        receive_audio_from_tts(ws, audio_track,
                                                               response_id, session_id))
                                except Exception as exc:
                                    logging.error("[%s] TTS connect failed: %s", session_id, exc)
                                    ws = None
                                    if channel_open(channel):
                                        channel.send(json.dumps({
                                            "type": "error", "event_id": generate_id(),
                                            "error": {"type": "tts_connection_error",
                                                      "message": f"TTS unavailable. {exc}"}}))

                        full_ai_response += dt

                        if channel_open(channel):
                            ev_type = ("response.output_audio_transcript.delta"
                                       if "audio" in modalities else
                                       "response.output_text.delta")
                            channel.send(json.dumps({"type": ev_type, "event_id": generate_id(),
                                                     "response_id": response_id, "delta": dt}))

                        if ws and not ws.closed:
                            try:
                                await ws.send_json({"type": "input.text", "text": dt})
                            except ConnectionResetError:
                                logging.error("TTS WS connection reset")
                                ws = None

                    # ---- tool calls ----
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.get("id", generate_id("call")),
                                "type": "function",
                                "function": {"name": tc.get("function", {}).get("name", ""),
                                             "arguments": ""}}
                        func = tc.get("function")
                        if func and func.get("arguments"):
                            arg_delta = func["arguments"]
                            tool_calls_buffer[idx]["function"]["arguments"] += arg_delta
                            if channel_open(channel):
                                channel.send(json.dumps({
                                    "type": "response.function_call_arguments.delta",
                                    "event_id": generate_id(),
                                    "response_id": response_id,
                                    "call_id": tool_calls_buffer[idx]["id"],
                                    "delta": arg_delta}))
                except json.JSONDecodeError:
                    pass
                except Exception:
                    logging.error("[%s] LLM chunk error: %s", session_id, traceback.format_exc())

    except asyncio.CancelledError:
        is_cancelled = True
        logging.info("[%s] LLM generation cancelled (barge-in).", session_id)
        content = {"type": "output_audio" if "audio" in modalities else "output_text"}
        if "audio" in modalities:
            content["transcript"] = full_ai_response or ""
        else:
            content["text"] = full_ai_response or ""
        if channel_open(channel):
            channel.send(json.dumps({
                "type": "conversation.item.done", "event_id": generate_id("evt"),
                "item": {"id": item_id, "object": "realtime.item", "type": "message",
                         "role": "assistant", "content": [content], "status": "incomplete"}}))
            channel.send(json.dumps({"type": "response.done", "event_id": generate_id(),
                                     "response": {"id": response_id, "status": "cancelled"}}))

    finally:
        # --- tear down TTS ---
        if ws and not ws.closed:
            if is_cancelled:
                if recv_task and not recv_task.done():
                    recv_task.cancel()
            else:
                try:
                    await ws.send_json({"type": "input.done"})
                except Exception:
                    pass
                if recv_task:
                    await recv_task
            await ws.close()

        # --- tear down HTTP session ---
        if not session.closed:
            await session.close()

        # --- save to history ---
        assistant_msg: dict = {"role": "assistant"}
        if full_ai_response:
            assistant_msg["content"] = full_ai_response
        if full_reasoning_response:
            assistant_msg["reasoning_content"] = full_reasoning_response
        if tool_calls_buffer:
            assistant_msg["tool_calls"] = list(tool_calls_buffer.values())
            if channel_open(channel) and not is_cancelled:
                for tc in assistant_msg["tool_calls"]:
                    channel.send(json.dumps({
                        "type": "response.function_call_arguments.done",
                        "event_id": generate_id(), "response_id": response_id,
                        "call_id": tc["id"], "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]}))

        if full_ai_response or full_reasoning_response or tool_calls_buffer:
            assistant_msg["id"] = item_id
            history.append(assistant_msg)
            trim_history(history)

        # --- finalise ---
        content = {"type": "output_audio" if "audio" in modalities else "output_text"}
        if "audio" in modalities:
            content["transcript"] = full_ai_response or ""
        else:
            content["text"] = full_ai_response or ""

        if channel_open(channel) and not full_ai_response.endswith("cancelled"):
            if full_reasoning_response:
                channel.send(json.dumps({"type": "response.output_reasoning.done",
                                         "event_id": generate_id(),
                                         "response_id": response_id,
                                         "content": full_reasoning_response}))
            if full_ai_response:
                if "audio" in modalities:
                    channel.send(json.dumps({"type": "response.output_audio_transcript.done",
                                             "event_id": generate_id(),
                                             "response_id": response_id,
                                             "transcript": full_ai_response}))
                else:
                    channel.send(json.dumps({"type": "response.output_text.done",
                                             "event_id": generate_id(),
                                             "response_id": response_id,
                                             "text": full_ai_response}))
            channel.send(json.dumps({
                "type": "conversation.item.done", "event_id": generate_id("evt"),
                "item": {"id": item_id, "object": "realtime.item", "type": "message",
                         "role": "assistant", "content": [content], "status": "completed"}}))
            channel.send(json.dumps({"type": "response.done", "event_id": generate_id(),
                                     "response": {"id": response_id, "status": "completed"}}))


# ---------------------------------------------------------------------------
# process_user_audio — transcribe a chunk of user speech, update history
# ---------------------------------------------------------------------------
async def process_user_audio(audio_float32: np.ndarray,
                              audio_track: LocalAIAudioTrack,
                              session_id: str,
                              auto_trigger: bool = True):
    logging.info("[%s] Transcribing …", session_id)

    def _transcribe():
        segments, _ = whisper_model.transcribe(audio_float32, beam_size=WHISPER_BEAM_SIZE)
        return " ".join(s.text for s in segments)

    channel = config.active_sessions[session_id].get("channel")
    item_id = generate_id("item")

    async with whisper_queue_lock:
        user_text = await asyncio.to_thread(_transcribe)
    user_text = user_text.strip()

    if not user_text:
        if channel_open(channel):
            channel.send(json.dumps({
                "type": "conversation.item.input_audio_transcription.failed",
                "event_id": generate_id(), "item_id": item_id,
                "error": {"type": "transcription_error", "code": "no_speech",
                          "message": "No speech detected."}}))
        return

    logging.info("[%s] User said: %s", session_id, user_text)
    hist = config.active_sessions[session_id]["history"]
    hist.append({"role": "user", "content": user_text, "id": item_id})
    trim_history(hist)

    if channel_open(channel):
        channel.send(json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": generate_id(), "item_id": item_id, "transcript": user_text}))
        channel.send(json.dumps({
            "type": "conversation.item.added", "event_id": generate_id(),
            "item": {"id": item_id, "object": "realtime.item", "type": "message",
                     "role": "user",
                     "content": [{"type": "input_audio", "transcript": user_text}]}}))

    if auto_trigger:
        sd = config.active_sessions[session_id]
        old = sd.get("response_task")
        if old and not old.done():
            old.cancel()
        sd["response_task"] = asyncio.create_task(
            trigger_ai_response(session_id, audio_track))


# ---------------------------------------------------------------------------
# process_incoming_audio — VAD loop (runs per WebRTC track)
# ---------------------------------------------------------------------------
async def process_incoming_audio(track, local_audio_track: LocalAIAudioTrack,
                                  session_id: str, session_vad):
    resampler_16k = av.AudioResampler(format="s16", layout="mono",
                                       rate=SILERO_SAMPLE_RATE)
    vad_buffer: list[float] = []
    speech_frames: list[float] = []
    is_speaking = False
    silence_chunks = 0
    chunk_size = SILERO_CHUNK_SIZE
    pre_speech_buffer = deque(maxlen=PRE_SPEECH_BUFFER_CHUNKS)

    while True:
        try:
            session_data = config.active_sessions.get(session_id)
            if not session_data:
                break
            channel = session_data.get("channel")

            # --- client-requested buffer clear ---
            if session_data.get("clear_audio_buffer"):
                vad_buffer.clear()
                speech_frames.clear()
                is_speaking = False
                silence_chunks = 0
                pre_speech_buffer.clear()
                session_data["clear_audio_buffer"] = False

            # --- push-to-talk commit ---
            if session_data.get("commit_audio_buffer"):
                if speech_frames:
                    audio = np.array(speech_frames, dtype=np.float32)
                    session_data["transcription_task"] = asyncio.create_task(
                        process_user_audio(audio, local_audio_track, session_id,
                                           auto_trigger=False))
                vad_buffer.clear()
                speech_frames.clear()
                is_speaking = False
                silence_chunks = 0
                pre_speech_buffer.clear()
                session_data["commit_audio_buffer"] = False
                session_data["commit_consumed_event"].set()

            frame = await track.recv()
            session_config = session_data.get("config", {})
            turn_detection = session_config.get("audio", {}).get("input", {}).get("turn_detection")

            for r_frame in resampler_16k.resample(frame):
                pcm = r_frame.to_ndarray().flatten()
                float_pcm = (pcm.astype(np.float32) * INT16_TO_FLOAT).tolist()

                if not turn_detection or turn_detection.get("type") != "server_vad":
                    speech_frames.extend(float_pcm)
                    continue

                # --- server VAD mode ---
                vad_threshold = turn_detection.get("threshold", 0.5)
                silence_duration_ms = turn_detection.get("silence_duration_ms", 500)
                prefix_padding_ms = turn_detection.get("prefix_padding_ms", 480)
                silence_limit = max(1, int(silence_duration_ms / SILERO_CHUNK_MS))
                padding_limit = max(1, int(prefix_padding_ms / SILERO_CHUNK_MS))

                if pre_speech_buffer.maxlen != padding_limit:
                    pre_speech_buffer = deque(pre_speech_buffer, maxlen=padding_limit)

                vad_buffer.extend(float_pcm)

                while len(vad_buffer) >= chunk_size:
                    chunk = np.array(vad_buffer[:chunk_size], dtype=np.float32)
                    del vad_buffer[:chunk_size]
                    prob = session_vad.process(chunk.tobytes())

                    if prob > vad_threshold:
                        if not is_speaking:
                            is_speaking = True
                            if channel_open(channel):
                                channel.send(json.dumps({
                                    "type": "input_audio_buffer.speech_started",
                                    "event_id": generate_id(),
                                    "audio_start_ms": max(
                                        0, len(speech_frames) * SILERO_CHUNK_MS - prefix_padding_ms)}))

                            # --- barge-in ---
                            active_task = session_data.get("response_task")
                            if active_task and not active_task.done():
                                active_task.cancel()
                                session_data["response_task"] = None
                            local_audio_track.active_response_id = None
                            local_audio_track.queue.clear()
                            for pre in pre_speech_buffer:
                                speech_frames.extend(pre)
                        silence_chunks = 0
                        speech_frames.extend(chunk.tolist())
                    else:
                        if is_speaking:
                            silence_chunks += 1
                            speech_frames.extend(chunk.tolist())
                            if silence_chunks >= silence_limit:
                                if channel_open(channel):
                                    channel.send(json.dumps({
                                        "type": "input_audio_buffer.speech_stopped",
                                        "event_id": generate_id(),
                                        "audio_end_ms": len(speech_frames) * SILERO_CHUNK_MS}))
                                audio = np.array(speech_frames, dtype=np.float32)
                                session_data["transcription_task"] = asyncio.create_task(
                                    process_user_audio(audio, local_audio_track,
                                                       session_id, auto_trigger=True))
                                is_speaking = False
                                speech_frames.clear()
                                pre_speech_buffer.clear()
                        else:
                            pre_speech_buffer.append(chunk.tolist())
        except MediaStreamError:
            logging.info("[%s] Audio track ended (peer disconnected).", session_id)
            break
        except Exception:
            logging.exception("[%s] Audio track error.", session_id)
            break
