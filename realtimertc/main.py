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
from realtimertc.cache import background_polling_task
from realtimertc.webrtc import handle_webrtc_offer


# ---------------------------------------------------------------------------
# App lifecycle helpers
# ---------------------------------------------------------------------------
async def _background_worker(app):
    task = asyncio.create_task(background_polling_task(app))
    task.add_done_callback(
        lambda t: logging.error("Background poller crashed: %s", t.exception())
        if not t.cancelled() and t.exception() else None)
    logging.info("Background cache poller started.")
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logging.info("Background cache poller stopped.")


async def _shutdown_peers(app):
    await asyncio.gather(*(pc.close() for pc in config.pcs))


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
async def get_models(request: web.Request) -> web.Response:
    return web.json_response({"models": list(config.AVAILABLE_MODELS)})


async def get_voices(request: web.Request) -> web.Response:
    return web.json_response({"voices": list(config.AVAILABLE_VOICES)})


async def serve_index(request: web.Request) -> web.Response:
    html_path = os.path.join(os.path.dirname(__file__), "html", "index.html")
    try:
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return web.Response(status=404, text="index.html not found.")

    content = (content
               .replace("'$AVAILABLE_VOICES_JSON$'", json.dumps(config.AVAILABLE_VOICES))
               .replace("'$AVAILABLE_MODELS_JSON$'", json.dumps(config.AVAILABLE_MODELS)))
    return web.Response(content_type="text/html", text=content)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    app = web.Application()

    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True, expose_headers="*", allow_headers="*")})

    app.router.add_get("/", serve_index)
    app.router.add_get("/api/voices", get_voices)
    app.router.add_get("/api/models", get_models)
    app.router.add_post("/v1/realtime/calls", handle_webrtc_offer)

    app.cleanup_ctx.append(_background_worker)
    app.on_shutdown.append(_shutdown_peers)

    for route in list(app.router.routes()):
        cors.add(route)

    logging.info("Realtime API server → http://%s:%s", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port)
