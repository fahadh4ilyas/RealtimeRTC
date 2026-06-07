"""
Audio infrastructure: byte-level queue, WebRTC MediaStreamTrack, and the
background task that pulls PCM from the TTS WebSocket.
"""

import asyncio
import fractions
import logging
import time

import aiohttp
import av
from aiortc.mediastreams import MediaStreamTrack

from realtimertc.config import (
    BYTES_PER_SAMPLE,
    WEBRTC_FRAME_DURATION,
    WEBRTC_SAMPLE_RATE,
)


# ---------------------------------------------------------------------------
# ByteQueue — thread‑safe(ish) byte buffer with timed reads
# ---------------------------------------------------------------------------
class ByteQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.buffer = bytearray()

    def put_nowait(self, data: bytes):
        self.queue.put_nowait(data)

    async def get_with_timeout(self, n: int, timeout: float) -> bytes | None:
        start_time = time.time()
        while len(self.buffer) < n:
            time_left = timeout - (time.time() - start_time)
            if time_left <= 0:
                return None
            try:
                data = await asyncio.wait_for(self.queue.get(), timeout=time_left)
                self.buffer.extend(data)
            except asyncio.TimeoutError:
                return None
        result = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return bytes(result)

    def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.buffer.clear()


# ---------------------------------------------------------------------------
# LocalAIAudioTrack — outbound WebRTC audio (AI → browser)
# ---------------------------------------------------------------------------
class LocalAIAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = ByteQueue()
        self._timestamp = 0
        self.sample_rate = WEBRTC_SAMPLE_RATE
        self.samples_per_frame = int(self.sample_rate * WEBRTC_FRAME_DURATION)
        self.bytes_per_frame = self.samples_per_frame * BYTES_PER_SAMPLE
        self.active_response_id = None
        self._last_frame_time = None
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=self.sample_rate)

    async def recv(self):
        now = time.time()
        if self._last_frame_time is None:
            self._last_frame_time = now

        target_time = self._last_frame_time + WEBRTC_FRAME_DURATION
        wait_time = max(0.0, target_time - time.time())

        payload = await self.queue.get_with_timeout(self.bytes_per_frame, timeout=wait_time)
        if not payload:
            payload = b"\x00" * self.bytes_per_frame

        now = time.time()
        if now < target_time:
            await asyncio.sleep(target_time - now)
        self._last_frame_time = target_time

        frame = av.AudioFrame(format="s16", layout="mono", samples=self.samples_per_frame)
        frame.sample_rate = self.sample_rate
        frame.planes[0].update(payload)
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        self._timestamp += self.samples_per_frame
        return frame

    def enqueue_pcm(self, pcm_bytes: bytes, sample_rate: int, response_id: str):
        """Queue raw PCM, resampling if needed.  Ignores stale response IDs."""
        if self.active_response_id != response_id:
            return
        if sample_rate != self.sample_rate:
            samples = len(pcm_bytes) // BYTES_PER_SAMPLE
            in_frame = av.AudioFrame(format="s16", layout="mono", samples=samples)
            in_frame.sample_rate = sample_rate
            in_frame.planes[0].update(pcm_bytes)
            resampled = bytearray()
            for out_frame in self.resampler.resample(in_frame):
                resampled.extend(out_frame.to_ndarray().tobytes())
            pcm_bytes = bytes(resampled)
        self.queue.put_nowait(pcm_bytes)


# ---------------------------------------------------------------------------
# Background task: drain audio bytes from the TTS WebSocket
# ---------------------------------------------------------------------------
async def receive_audio_from_tts(ws, audio_track: LocalAIAudioTrack,
                                  response_id: str, session_id: str):
    current_sample_rate = WEBRTC_SAMPLE_RATE
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                audio_track.enqueue_pcm(msg.data, current_sample_rate, response_id)
            elif msg.type == aiohttp.WSMsgType.TEXT:
                data = msg.json()
                if data.get("type") == "audio.start":
                    current_sample_rate = data.get("sample_rate", WEBRTC_SAMPLE_RATE)
                elif data.get("type") == "session.done":
                    break
                elif data.get("type") == "error":
                    logging.warning("[%s] TTS stream error: %s", session_id, data["message"])
            elif msg.type == aiohttp.WSMsgType.CLOSE:
                logging.error("[%s] TTS stream closed: %s", session_id, msg.extra)
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logging.error("[%s] TTS stream transport error: %s", session_id, msg.data)
                break
    except asyncio.CancelledError:
        pass
