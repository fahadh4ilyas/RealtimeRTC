"""
Core real-time pipeline: VAD → STT → LLM → TTS.
"""

import asyncio
import base64
import io
import json
import logging
import time
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
    PRE_SPEECH_BUFFER_CHUNKS,
    REASONING_KWARGS,
    SILERO_CHUNK_MS,
    SILERO_CHUNK_SIZE,
    SILERO_SAMPLE_RATE,
    SSE_PREFIX_LENGTH,
    WEBRTC_SAMPLE_RATE,
    WHISPER_BEAM_SIZE,
    llm_chat_api,
    tts_ws_api,
    whisper_pool,
)
from realtimertc.utils import channel_open, generate_id, trim_history


def _headers(api_key: str) -> dict:
    """Build auth headers from a client-supplied API key."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _encode_mp4(frames, fps):
    """Encode a list of RGB ndarray frames into raw mp4 bytes (or None)."""
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    w -= w % 2
    h -= h % 2
    out = io.BytesIO()
    container = av.open(out, mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    for i, arr in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(arr[:h, :w], format="rgb24")
        frame = frame.reformat(format="yuv420p", width=w, height=h)
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return out.getvalue()


def frame_to_jpeg(arr):
    """Encode a single RGB ndarray frame into a JPEG base64 data URL."""
    return "data:image/jpeg;base64," + base64.b64encode(_encode_jpeg(arr)).decode("ascii")


def _encode_jpeg(arr):
    """Encode a single RGB ndarray frame into raw JPEG bytes."""
    frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
    codec = av.codec.CodecContext.create("mjpeg", "w")
    codec.width = frame.width
    codec.height = frame.height
    codec.pix_fmt = "yuvj420p"
    data = b""
    for packet in codec.encode(frame):
        data += bytes(packet)
    data += bytes(codec.encode(None))
    return data


def _track_rgb_small(frame, max_dim=480):
    """Reformat an av.VideoFrame into a small RGB ndarray (memory bound)."""
    w, h = frame.width, frame.height
    scale = min(1.0, max_dim / max(w, h))
    nw = max(2, int(w * scale))
    nh = max(2, int(h * scale))
    nw -= nw % 2
    nh -= nh % 2
    return frame.reformat(format="rgb24", width=nw, height=nh).to_ndarray()


async def process_incoming_video(track, session_id):
    """Consume the inbound video track, sampling frames into a rolling buffer.

    Frames are only recorded while the session's ``tracking`` flag is set
    (toggled by the client's track.start / track.stop events), and are sampled
    at ``track_sample_interval`` so the buffer spans the configured duration.
    """
    while True:
        try:
            frame = await track.recv()
        except Exception:
            break
        sd = config.active_sessions.get(session_id)
        if not sd:
            break
        if not sd.get("tracking"):
            continue
        now = time.time()
        if now - sd.get("last_sample_time", 0.0) >= sd.get("track_sample_interval", 1 / 30):
            sd["last_sample_time"] = now
            try:
                sd["tracked_frames"].append(_track_rgb_small(frame))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# create_user_item — build the user history item for the current turn
# ---------------------------------------------------------------------------
def _media_thumbnail(data_url: str, max_dim: int = 192):
    """Decode the first video frame of an image/video data URL and return a
    small JPEG thumbnail data URL, or None if decoding fails."""
    try:
        b64 = data_url.split(",", 1)[1]
        container = av.open(io.BytesIO(base64.b64decode(b64)))
        try:
            frame = next(container.decode(video=0))
        finally:
            container.close()
        w, h = frame.width, frame.height
        scale = min(1.0, max_dim / max(w, h))
        nw = max(2, int(w * scale))
        nh = max(2, int(h * scale))
        nw -= nw % 2
        nh -= nh % 2
        arr = frame.reformat(format="rgb24", width=nw, height=nh).to_ndarray()
        return frame_to_jpeg(arr)
    except Exception:
        return None


def _strip_media_id(history):
    """Drop client-facing ``media_id`` fields before sending history to vLLM."""
    messages = []
    for m in history:
        content = m.get("content")
        if isinstance(content, list):
            content = [{k: v for k, v in p.items() if k != "media_id"}
                       for p in content]
            m = dict(m, content=content)
        messages.append(m)
    return messages


async def _to_realtime_parts(content, is_transcription: bool, session_id: str) -> list:
    """Map vLLM chat content to Realtime content parts for the emitted item."""
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    result = []
    for p in parts:
        if p["type"] == "text":
            if is_transcription:
                result.append({"type": "input_audio", "transcript": p["text"]})
            else:
                result.append({"type": "input_text", "text": p["text"]})
        elif p["type"] == "image_url":
            # Every image is stored server-side and served via the endpoint, so
            # the client can fetch the full image directly (no thumbnail).
            result.append({"type": "input_image",
                           "url": config.media_endpoint(session_id, p["media_id"])})
        elif p["type"] == "video_url":
            part = {"type": "input_video",
                    "url": config.media_endpoint(session_id, p["media_id"])}
            # Video still gets a thumbnail as its poster/preview.
            thumb = await asyncio.to_thread(_media_thumbnail, p["video_url"]["url"])
            if thumb:
                part["thumbnail"] = thumb
            result.append(part)
    return result


async def create_user_item(session_id: str, content, item_id: str | None = None,
                           send_item_created: bool = False,
                           previous_item_id: str | None = None,
                           is_transcription: bool = False) -> str:
    """Merge held attachments (pending_media) and auto-tracked frames with the
    incoming content, append the user item to history, trim, and emit
    conversation.item.added carrying the full content (media included). When
    send_item_created is true, emit conversation.item.created first.

    `content` is a plain string for text-only, or a list of vLLM chat parts
    (text / image_url / video_url) for mixed content. When is_transcription is
    true, the emitted text part is `input_audio` rather than `input_text`.
    """
    sd = config.active_sessions[session_id]
    hist = sd["history"]

    media_parts = list(sd.get("pending_media", []))
    sd["pending_media"] = []

    tracked = sd.get("tracked_frames")
    if sd.get("tracking") and tracked and len(tracked) > 0:
        if len(tracked) == 1:
            # A single tracked frame is an image, not a video.
            raw = await asyncio.to_thread(_encode_jpeg, tracked[0])
            media_id = generate_id("media")
            config.store_media(session_id, media_id, "image/jpeg", raw)
            media_parts.append({"type": "image_url",
                                "image_url": {"url": config.resolve_media_url(session_id, media_id)},
                                "media_id": media_id})
        else:
            interval = sd.get("track_sample_interval") or 0
            fps = round(1.0 / interval) if interval > 0 else 1
            raw = await asyncio.to_thread(_encode_mp4, list(tracked), fps)
            media_id = generate_id("media")
            config.store_media(session_id, media_id, "video/mp4", raw)
            media_parts.append({"type": "video_url",
                                "video_url": {"url": config.resolve_media_url(session_id, media_id)},
                                "media_id": media_id})

    if media_parts:
        content = media_parts + (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        )

    item_id = item_id or generate_id("item")
    hist.append({"role": "user", "content": content, "id": item_id})
    trim_history(hist)

    channel = sd.get("channel")
    if channel_open(channel):
        item = {"id": item_id, "object": "realtime.item", "type": "message",
                "role": "user", "content": await _to_realtime_parts(content, is_transcription, session_id)}
        if send_item_created:
            channel.send(json.dumps({
                "type": "conversation.item.created", "event_id": generate_id(),
                "previous_item_id": previous_item_id, "item": item}))
        channel.send(json.dumps({
            "type": "conversation.item.added", "event_id": generate_id(),
            "item": item}))

    return item_id


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
    # create_user_item already merged held media + tracked frames into the
    # user turn. Only the media-only case (attachments with no text) still
    # needs a user item here: add a default prompt. Skip a tool result — a
    # function-call continuation must leave any held media undisturbed.
    last_role = history[-1].get("role") if history else None
    if last_role not in ("tool", "user"):
        has_media = bool(session_data.get("pending_media")) or (
            session_data.get("tracking") and session_data.get("tracked_frames"))
        if has_media:
            await create_user_item(
                session_id,
                "Please describe what you see in the attached media.")

    session_config = session_data["config"]
    channel = session_data.get("channel")

    llm_key = session_data.get("api_key", "")
    tts_key = session_data.get("tts_api_key", "")
    llm_base_url = session_data.get("llm_base_url") or config.LLM_BASE_URL
    tts_base_url = session_data.get("tts_base_url") or config.TTS_BASE_URL
    llm_api = llm_chat_api(llm_base_url)
    tts_ws_url = tts_ws_api(tts_base_url)

    req_config = response_config or {}
    modalities = req_config.get("output_modalities",
                                session_config.get("output_modalities", ["audio"]))
    tools = req_config.get("tools", session_config.get("tools", []))
    voice = req_config.get("audio", {}).get("output", {}).get(
        "voice", session_config.get("audio", {}).get("output", {}).get("voice", ""))
    model_name = req_config.get("model", session_config.get("model", "Qwen3.5-9B"))
    reasoning_effort = req_config.get("reasoning", {}).get(
        "effort", session_config.get("reasoning", {}).get("effort", "none"))
    reasoning_kwargs = REASONING_KWARGS.get(reasoning_effort, {})

    llm_payload = {"model": model_name, "messages": _strip_media_id(history),
                   "stream": True, **reasoning_kwargs}
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
        async with session.post(llm_api, json=llm_payload,
                headers=_headers(llm_key)) as llm_resp:
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
                                    ws = await session.ws_connect(tts_ws_url,
                                            headers=_headers(tts_key))
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
            assistant_msg["reasoning"] = full_reasoning_response
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

    def _transcribe(model):
        segments, _ = model.transcribe(audio_float32, beam_size=WHISPER_BEAM_SIZE)
        return " ".join(s.text for s in segments)

    channel = config.active_sessions[session_id].get("channel")
    item_id = generate_id("item")

    model = await whisper_pool.get()
    try:
        user_text = await asyncio.to_thread(_transcribe, model)
    finally:
        whisper_pool.put_nowait(model)
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

    if channel_open(channel):
        channel.send(json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": generate_id(), "item_id": item_id, "transcript": user_text}))

    await create_user_item(session_id, user_text, item_id=item_id, is_transcription=True)

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
