"""
Local Realtime API Server — entry point.

Start with:  python main.py [--host 0.0.0.0] [--port 8081]
"""

import logging
logging.basicConfig(level=logging.INFO)

import argparse
import asyncio
import json
import os

import aiohttp_cors
from aiohttp import web

from realtimertc import config
from realtimertc.cache import fetch_models, fetch_voices
from realtimertc.utils import generate_id
from realtimertc.webrtc import handle_webrtc_offer


async def _shutdown_peers(app):
    await asyncio.gather(*(pc.close() for pc in config.pcs))


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
async def upload_media(request: web.Request) -> web.Response:
    """Store uploaded media (video) keyed by session, return an id for later use."""
    session_id = request.query.get("session_id", "")
    data = await request.read()
    mime = request.headers.get("Content-Type", "application/octet-stream")
    media_id = generate_id("media")
    config.store_media(session_id, media_id, mime, data)
    return web.json_response({"id": media_id})


async def get_media(request: web.Request) -> web.Response:
    """Serve an uploaded media blob so the client can play it back."""
    session_id = request.match_info["session_id"]
    media_id = request.match_info["media_id"]
    entry = config.uploaded_media.get(session_id, {}).get(media_id)
    if entry is None:
        return web.Response(status=404, text="Media not found.")
    return web.Response(body=entry["data"], content_type=entry["mime"])


async def get_models(request: web.Request) -> web.Response:
    """Fetch the model list using the client-supplied API key + base URL."""
    body = await request.json()
    api_key = (body or {}).get("api_key", "")
    base_url = (body or {}).get("base_url") or config.LLM_BASE_URL
    models, error, status = await fetch_models(api_key, base_url)
    if error is not None:
        return web.json_response({"error": error},
                                 status=status if status else 502)
    return web.json_response({"models": models})


async def get_voices(request: web.Request) -> web.Response:
    """Fetch the voice list using the client-supplied API key + base URL."""
    body = await request.json()
    api_key = (body or {}).get("api_key", "")
    base_url = (body or {}).get("base_url") or config.TTS_BASE_URL
    voices, error, status = await fetch_voices(api_key, base_url)
    if error is not None:
        return web.json_response({"error": error},
                                 status=status if status else 502)
    return web.json_response({"voices": voices})


async def serve_index(request: web.Request) -> web.Response:
    html_path = os.path.join(os.path.dirname(__file__), "html", "index.html")
    try:
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return web.Response(status=404, text="index.html not found.")

    content = (content
               .replace("'$LLM_BASE_URL$'", json.dumps(config.LLM_BASE_URL))
               .replace("'$TTS_BASE_URL$'", json.dumps(config.TTS_BASE_URL)))
    return web.Response(content_type="text/html", text=content)


async def favicon(request: web.Request) -> web.Response:
    path = os.path.join(os.path.dirname(__file__), "html", "favicon.ico")
    return web.FileResponse(path, headers={"Content-Type": "image/x-icon"})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    app = web.Application(client_max_size=100 * 1024 * 1024)

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True, expose_headers="*", allow_headers="*")})

    app.router.add_get("/", serve_index)
    app.router.add_get("/favicon.ico", favicon)
    app.router.add_post("/api/upload", upload_media)
    app.router.add_get("/api/media/{session_id}/{media_id}", get_media)
    app.router.add_post("/api/voices", get_voices)
    app.router.add_post("/api/models", get_models)
    app.router.add_post("/v1/realtime/calls", handle_webrtc_offer)

    app.on_shutdown.append(_shutdown_peers)

    for route in list(app.router.routes()):
        cors.add(route)

    logging.info("Realtime API server → http://%s:%s", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)
