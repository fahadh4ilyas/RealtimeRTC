"""
WebRTC signalling handler and OpenAI Realtime API event router.
"""

import asyncio
import json
import logging
import traceback
from collections import deque

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)
from aiohttp import web
from silero_vad_lite import SileroVAD

from realtimertc import config
from realtimertc.audio import LocalAIAudioTrack
from realtimertc.realtime import (
    create_user_item,
    process_incoming_audio,
    process_incoming_video,
    trigger_ai_response,
)
from realtimertc.utils import cleanup_session, generate_id, trim_history


# ---------------------------------------------------------------------------
# POST /v1/realtime/calls — WebRTC offer handler
# ---------------------------------------------------------------------------
async def handle_webrtc_offer(request: web.Request) -> web.Response:
    content = await request.text()
    offer = RTCSessionDescription(sdp=content, type="offer")

    ice_config = RTCConfiguration(iceServers=[
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
        RTCIceServer(urls="stun:stun1.l.google.com:19302"),
        RTCIceServer(
            urls=["turn:openrelay.metered.ca:80",
                  "turn:openrelay.metered.ca:80?transport=tcp",
                  "turns:openrelay.metered.ca:443?transport=tcp"],
            username="openrelayproject",
            credential="openrelayproject",
        ),
    ])

    pc = RTCPeerConnection(ice_config)
    config.pcs.add(pc)

    session_id = generate_id("sess")
    session_vad = SileroVAD(config.SILERO_SAMPLE_RATE)

    config.active_sessions[session_id] = {
        "pc": pc,
        "channel": None,
        "vad_model": session_vad,
        "clear_audio_buffer": False,
        "commit_audio_buffer": False,
        "commit_consumed_event": asyncio.Event(),
        "response_task": None,
        "transcription_task": None,
        "api_key": "",
        "tts_api_key": "",
        "llm_base_url": config.LLM_BASE_URL,
        "tts_base_url": config.TTS_BASE_URL,
        "pending_media": [],
        "uploaded_media": {},
        "tracking": False,
        "tracked_frames": deque(maxlen=30),
        "track_sample_interval": 0.1,
        "last_sample_time": 0.0,
        "config": {
            "model": "",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.3,
                        "prefix_padding_ms": 800,
                        "silence_duration_ms": 500,
                    }
                },
                "output": {"voice": ""},
            },
            "instructions": config.DEFAULT_SYSTEM_PROMPT,
            "reasoning": {"effort": "none"},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "description": "Returns the current weekday, month, date, year, and time. For example: 'Mon, Jan 15 2025 14:30:00+0700'. Pass a timezone offset (e.g. +0700, -0500) to get the time for that zone. Omit to use the user's local timezone. Use this tool whenever the user asks about the current time, day, date, or any combination of these.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "timezone": {
                                    "type": "string",
                                    "description": "Timezone offset like +0700 or -0500. Omit to use the user's local timezone.",
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Evaluate a mathematical expression and return the result.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Mathematical expression to evaluate, e.g. '2 + 3 * 4' or 'Math.sqrt(144)'.",
                                }
                            },
                            "required": ["expression"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "convert_units",
                        "description": "Convert a value between units (length, weight, temperature, etc.).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "value": {
                                    "type": "number",
                                    "description": "The numeric value to convert.",
                                },
                                "from_unit": {
                                    "type": "string",
                                    "description": "Source unit, e.g. 'km', 'miles', 'celsius', 'fahrenheit', 'kg', 'lbs', 'liters', 'gallons'.",
                                },
                                "to_unit": {
                                    "type": "string",
                                    "description": "Target unit, e.g. 'km', 'miles', 'celsius', 'fahrenheit', 'kg', 'lbs', 'liters', 'gallons'.",
                                }
                            },
                            "required": ["value", "from_unit", "to_unit"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_random_number",
                        "description": "Generate a random integer between min and max (inclusive).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "min": {
                                    "type": "integer",
                                    "description": "Minimum value (inclusive). Default 1.",
                                },
                                "max": {
                                    "type": "integer",
                                    "description": "Maximum value (inclusive). Default 100.",
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_user_location",
                        "description": "Get the user's current geographic location (requires permission). Returns latitude, longitude, and accuracy.",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "roll_dice",
                        "description": "Roll a number of dice with a given number of sides each.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "count": {
                                    "type": "integer",
                                    "description": "Number of dice to roll. Default 1.",
                                },
                                "sides": {
                                    "type": "integer",
                                    "description": "Number of sides per die. Default 6.",
                                }
                            }
                        }
                    }
                }
            ],
        },
        "history": [{"role": "system", "content": config.DEFAULT_SYSTEM_PROMPT,
                      "id": generate_id("item")}],
    }

    logging.info("[%s] New WebRTC session.", session_id)

    # ------------------------------------------------------------------
    # connection-state cleanup
    # ------------------------------------------------------------------
    @pc.on("connectionstatechange")
    async def _on_cs_change():
        if pc.connectionState in ("failed", "closed"):
            cleanup_session(session_id, f"connection state: {pc.connectionState}")

    @pc.on("iceconnectionstatechange")
    async def _on_ice_change():
        logging.info("[%s] ICE → %s", session_id, pc.iceConnectionState)
        if pc.iceConnectionState in ("failed", "disconnected", "closed"):
            cleanup_session(session_id, f"ICE state: {pc.iceConnectionState}")

    local_audio = LocalAIAudioTrack()
    pc.addTrack(local_audio)

    # ------------------------------------------------------------------
    # oai-events DataChannel
    # ------------------------------------------------------------------
    @pc.on("datachannel")
    def _on_datachannel(channel):
        if channel.label != "oai-events":
            return
        config.active_sessions[session_id]["channel"] = channel

        @channel.on("open")
        def _on_open():
            cfg = config.active_sessions[session_id]["config"]
            channel.send(json.dumps({
                "type": "session.created",
                "event_id": generate_id(),
                "session": {
                    "id": session_id,
                    "object": "realtime.session",
                    "model": cfg["model"],
                    "output_modalities": cfg["output_modalities"],
                    "instructions": cfg["instructions"],
                    "audio": {
                        "input": {"turn_detection": cfg["audio"]["input"]["turn_detection"]},
                        "output": {"voice": cfg["audio"]["output"]["voice"]},
                    },
                    "reasoning": cfg["reasoning"],
                    "tools": cfg["tools"],
                },
            }))

        @channel.on("message")
        def _on_message(raw: str):
            try:
                event = json.loads(raw)
                etype = event.get("type")
                if not etype:
                    raise ValueError("Missing 'type' field.")

                # ======================================================
                # session.update
                # ======================================================
                if etype == "session.update":
                    sp = event.get("session", {})
                    sd = config.active_sessions[session_id]
                    cfg = sd["config"]

                    for key in ("api_key", "tts_api_key", "llm_base_url", "tts_base_url"):
                        if key in sp:
                            sd[key] = sp[key]

                    for key in ("model", "output_modalities", "tools"):
                        if key in sp:
                            cfg[key] = sp[key]

                    if "audio" in sp:
                        ap = sp["audio"]
                        if "input" in ap:
                            cfg["audio"]["input"] = cfg["audio"].get("input", {})
                            td = ap["input"].get("turn_detection")
                            if td is None:
                                cfg["audio"]["input"]["turn_detection"] = None
                            elif isinstance(td, dict):
                                cur = cfg["audio"]["input"].get("turn_detection")
                                if cur is None:
                                    cur = {}
                                    cfg["audio"]["input"]["turn_detection"] = cur
                                for k in ("type", "threshold", "prefix_padding_ms",
                                          "silence_duration_ms"):
                                    if k in td:
                                        cur[k] = td[k]
                        if "output" in ap:
                            out = cfg["audio"].setdefault("output", {})
                            for ok in ("voice",):
                                if ok in ap["output"]:
                                    out[ok] = ap["output"][ok]

                    if "reasoning" in sp:
                        cfg.setdefault("reasoning", {})
                        for rk in ("effort",):
                            if rk in sp["reasoning"]:
                                cfg["reasoning"][rk] = sp["reasoning"][rk]

                    if "instructions" in sp:
                        new_inst = sp["instructions"]
                        cfg["instructions"] = (
                            f"{config.DEFAULT_SYSTEM_PROMPT}\n\nAdditional instruction from user:\n{new_inst}"
                            if new_inst else config.DEFAULT_SYSTEM_PROMPT)
                        for m in config.active_sessions[session_id]["history"]:
                            if m["role"] == "system":
                                m["content"] = cfg["instructions"]
                                break

                    channel.send(json.dumps({"type": "session.updated",
                                             "event_id": generate_id(),
                                             "session": cfg}))

                # ======================================================
                # conversation.item.create
                # ======================================================
                elif etype == "conversation.item.create":
                    item = event.get("item", {})
                    hist = config.active_sessions[session_id]["history"]

                    if item.get("type") == "message" and item.get("role") == "user":
                        # Map Realtime-style content parts to the OpenAI chat format
                        # expected by vLLM: input_text → text, input_image → image_url,
                        # input_video → video_url. A single text-only part stays a
                        # plain string for backward compatibility.
                        content_parts = []
                        for cb in item.get("content", []):
                            cb_type = cb.get("type")
                            if cb_type == "input_text":
                                content_parts.append({"type": "text",
                                                      "text": cb.get("text", "")})
                            elif cb_type == "input_image":
                                data_url = cb.get("url", "")
                                media_id = generate_id("media")
                                config.store_data_url(session_id, media_id, data_url)
                                content_parts.append({"type": "image_url",
                                                      "image_url": {"url": data_url},
                                                      "media_id": media_id})
                            elif cb_type == "input_video":
                                media_id = cb.get("url", "")
                                content_parts.append({
                                    "type": "video_url",
                                    "video_url": {"url": config.resolve_media_url(session_id, media_id)},
                                    "media_id": media_id,
                                })
                        if content_parts:
                            if any(p["type"] == "text" for p in content_parts):
                                # Text present — create the user item (merges held media
                                # and tracked frames asynchronously).
                                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                                    content = content_parts[0]["text"]
                                else:
                                    content = content_parts
                                config.active_sessions[session_id]["user_item_task"] = \
                                    asyncio.create_task(create_user_item(
                                        session_id, content,
                                        item_id=item.get("id") or generate_id("item"),
                                        send_item_created=True,
                                        previous_item_id=event.get("previous_item_id")))
                            else:
                                # Media-only — hold it; merged with the current turn
                                # in trigger_ai_response.
                                config.active_sessions[session_id]["pending_media"].extend(content_parts)

                    elif item.get("type") == "function_call_output":
                        item["id"] = item.get("id") or generate_id("item")
                        hist.append({
                            "role": "tool",
                            "tool_call_id": item.get("call_id"),
                            "content": item.get("output"),
                            "id": item["id"],
                        })
                        trim_history(hist)
                        item["object"] = "realtime.item"

                        channel.send(json.dumps({"type": "conversation.item.created",
                                                 "event_id": generate_id(),
                                                 "previous_item_id": event.get("previous_item_id"),
                                                 "item": item}))
                        channel.send(json.dumps({"type": "conversation.item.added",
                                                 "event_id": generate_id(),
                                                 "item": item}))


                # ======================================================
                # conversation.item.retrieve / delete / truncate
                # ======================================================
                elif etype == "conversation.item.retrieve":
                    iid = event["item_id"]
                    found = next((m for m in config.active_sessions[session_id].get("history", [])
                                  if m.get("id") == iid), None)
                    if found:
                        channel.send(json.dumps({"type": "conversation.item.retrieved",
                                                 "event_id": generate_id(), "item": found}))
                    else:
                        channel.send(json.dumps({
                            "type": "error", "event_id": generate_id(),
                            "error": {"type": "invalid_request_error", "code": "item_not_found",
                                      "message": f"Item '{iid}' not found."}}))

                elif etype == "conversation.item.delete":
                    iid = event["item_id"]
                    hist = config.active_sessions[session_id].get("history", [])
                    for i, m in enumerate(hist):
                        if m.get("id") == iid:
                            del hist[i]
                            channel.send(json.dumps({"type": "conversation.item.deleted",
                                                     "event_id": generate_id(),
                                                     "item_id": iid}))
                            break
                    else:
                        channel.send(json.dumps({
                            "type": "error", "event_id": generate_id(),
                            "error": {"type": "invalid_request_error", "code": "item_not_found",
                                      "message": f"Item '{iid}' not found."}}))

                elif etype == "conversation.item.truncate":
                    iid = event["item_id"]
                    ci = event.get("content_index", 0)
                    aems = event["audio_end_ms"]
                    hist = config.active_sessions[session_id].get("history", [])
                    for m in hist:
                        if m.get("id") == iid and m.get("role") == "assistant":
                            for part in m.get("content", []):
                                if part.get("type") == "output_audio":
                                    part["audio_end_ms"] = aems
                                    part["truncated"] = True
                            channel.send(json.dumps({
                                "type": "conversation.item.truncated",
                                "event_id": generate_id(), "item_id": iid,
                                "content_index": ci, "audio_end_ms": aems}))
                            break
                    else:
                        channel.send(json.dumps({
                            "type": "error", "event_id": generate_id(),
                            "error": {"type": "invalid_request_error", "code": "item_not_found",
                                      "message": f"Item '{iid}' not found."}}))

                # ======================================================
                # audio buffer control
                # ======================================================
                elif etype == "input_audio_buffer.clear":
                    config.active_sessions[session_id]["clear_audio_buffer"] = True
                    channel.send(json.dumps({"type": "input_audio_buffer.cleared",
                                             "event_id": generate_id()}))

                elif etype == "output_audio_buffer.clear":
                    local_audio.active_response_id = None
                    local_audio.queue.clear()
                    channel.send(json.dumps({"type": "output_audio_buffer.cleared",
                                             "event_id": generate_id()}))

                elif etype == "input_audio_buffer.commit":
                    config.active_sessions[session_id]["commit_audio_buffer"] = True
                    config.active_sessions[session_id]["commit_consumed_event"].clear()
                    channel.send(json.dumps({"type": "input_audio_buffer.committed",
                                             "event_id": generate_id()}))

                # ======================================================
                # video track control (auto-track mode)
                # ======================================================
                elif etype == "track.start":
                    sd = config.active_sessions[session_id]
                    sd["tracking"] = True
                    frames_n = max(1, int(event.get("frames", 30)))
                    duration = max(0.1, float(event.get("duration", 3.0)))
                    sd["tracked_frames"] = deque(maxlen=frames_n)
                    sd["track_sample_interval"] = duration / frames_n
                    sd["last_sample_time"] = 0.0

                elif etype == "track.stop":
                    sd = config.active_sessions[session_id]
                    sd["tracking"] = False
                    # Drop any buffered frames so a later response doesn't reuse
                    # a stale "tracked" video after tracking has stopped.
                    sd["tracked_frames"].clear()

                # ======================================================
                # response.create / cancel
                # ======================================================
                elif etype == "response.cancel":
                    t = config.active_sessions[session_id].get("response_task")
                    if t and not t.done():
                        t.cancel()
                        config.active_sessions[session_id]["response_task"] = None

                elif etype == "response.create":
                    old = config.active_sessions[session_id].get("response_task")
                    if old and not old.done():
                        old.cancel()

                    resp_cfg = event.get("response", {})
                    cfg = config.active_sessions[session_id]["config"]

                    async def _queued_response():
                        sd = config.active_sessions[session_id]
                        # Wait for any in-flight audio commit to be consumed by the VAD loop
                        if sd.get("commit_audio_buffer"):
                            await sd["commit_consumed_event"].wait()
                        # Wait for transcription to finish (may be created by VAD loop above)
                        trans = sd.get("transcription_task")
                        if trans and not trans.done():
                            try:
                                await trans
                            except asyncio.CancelledError:
                                pass
                            except Exception as exc:
                                logging.error("[%s] Transcription failed: %s", session_id, exc)
                        # Wait for a pending user item (typed text merging media).
                        user_item = sd.get("user_item_task")
                        if user_item and not user_item.done():
                            try:
                                await user_item
                            except asyncio.CancelledError:
                                pass
                            except Exception as exc:
                                logging.error("[%s] User item failed: %s", session_id, exc)
                        await trigger_ai_response(session_id, local_audio, resp_cfg)

                    config.active_sessions[session_id]["response_task"] = asyncio.create_task(
                        _queued_response())

            except Exception as exc:
                channel.send(json.dumps({
                    "type": "error", "event_id": generate_id(),
                    "error": {"type": "invalid_request_error", "message": str(exc)}}))
                logging.error("[%s] Event error: %s", session_id,
                              traceback.format_exc())

    # ------------------------------------------------------------------
    # inbound media track
    # ------------------------------------------------------------------
    @pc.on("track")
    def _on_track(track):
        if track.kind == "audio":
            asyncio.create_task(
                process_incoming_audio(track, local_audio, session_id, session_vad))
        elif track.kind == "video":
            asyncio.create_task(
                process_incoming_video(track, session_id))

    # ------------------------------------------------------------------
    # SDP handshake
    # ------------------------------------------------------------------
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.Response(content_type="application/sdp", text=pc.localDescription.sdp)
