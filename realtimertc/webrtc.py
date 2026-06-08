"""
WebRTC signalling handler and OpenAI Realtime API event router.
"""

import asyncio
import json
import logging
import traceback

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
from realtimertc.realtime import process_incoming_audio, trigger_ai_response
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
        "config": {
            "model": config.DEFAULT_MODEL,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 480,
                        "silence_duration_ms": 500,
                    }
                },
                "output": {"voice": config.DEFAULT_VOICE},
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
                    cfg = config.active_sessions[session_id]["config"]

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
                                    v = ap["output"][ok]
                                    if out.get(ok) != v:
                                        if ok == "voice" and v not in list(config.AVAILABLE_VOICES):
                                            logging.warning(
                                                "[%s] Unavailable voice '%s'. Keeping '%s'.",
                                                session_id, v,
                                                out.get("voice", config.DEFAULT_VOICE))
                                            continue
                                        out[ok] = v

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
                        for cb in item.get("content", []):
                            if cb.get("type") == "input_text":
                                hist.append({"role": "user", "content": cb["text"],
                                             "id": item.get("id", generate_id("item"))})
                                trim_history(hist)

                    elif item.get("type") == "function_call_output":
                        hist.append({
                            "role": "tool",
                            "tool_call_id": item.get("call_id"),
                            "content": item.get("output"),
                            "id": item.get("id", generate_id("item")),
                        })
                        trim_history(hist)

                    if not item.get("id"):
                        item["id"] = generate_id("item")
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

                    # validate per-response voice override
                    if ("audio" in resp_cfg and "output" in resp_cfg["audio"]
                            and "voice" in resp_cfg["audio"]["output"]):
                        v = resp_cfg["audio"]["output"]["voice"]
                        if v != cfg["audio"]["output"].get("voice", config.DEFAULT_VOICE):
                            if v not in list(config.AVAILABLE_VOICES):
                                logging.warning("[%s] Unavailable voice '%s'.", session_id, v)
                                resp_cfg["audio"]["output"].pop("voice", None)

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

    # ------------------------------------------------------------------
    # SDP handshake
    # ------------------------------------------------------------------
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.Response(content_type="application/sdp", text=pc.localDescription.sdp)
