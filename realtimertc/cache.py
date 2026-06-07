"""
Background tasks that keep the model / voice caches up to date.
"""

import asyncio
import logging

import aiohttp

from realtimertc import config


def _llm_headers():
    return ({"Authorization": f"Bearer {config.LLM_API_KEY}"}
            if config.LLM_API_KEY else {})


def _tts_headers():
    return ({"Authorization": f"Bearer {config.TTS_API_KEY}"}
            if config.TTS_API_KEY else {})


async def update_model_cache():
    """Fetch the latest loaded models from vLLM."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(config.LLM_MODELS_API,
                                   headers=_llm_headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m.get("id") for m in data.get("data", [])
                              if m.get("id")]
                    if models and models != config.AVAILABLE_MODELS:
                        config.AVAILABLE_MODELS = models
                        if config.DEFAULT_MODEL not in models:
                            config.DEFAULT_MODEL = models[0]
                        logging.info("LLM models updated: %s",
                                     config.AVAILABLE_MODELS)
                else:
                    logging.warning("LLM models API returned %s", resp.status)
    except Exception:
        logging.warning("Model cache update failed", exc_info=True)


async def update_voice_cache():
    """Fetch the latest TTS voices."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(config.TTS_LIST_VOICES,
                                   headers=_tts_headers()) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    voices = data.get("voices", [])
                    if voices and voices != config.AVAILABLE_VOICES:
                        config.AVAILABLE_VOICES = voices
                        if config.DEFAULT_VOICE not in voices:
                            config.DEFAULT_VOICE = voices[0]
                        logging.info("TTS voices updated: %s",
                                     config.AVAILABLE_VOICES)
                else:
                    logging.warning("TTS voices API returned %s", resp.status)
    except Exception:
        logging.warning("Voice cache update failed", exc_info=True)


async def background_polling_task(app):
    """Refresh caches once, then every 60 seconds."""
    try:
        await asyncio.gather(update_voice_cache(), update_model_cache())
    except Exception:
        logging.warning("Initial cache populate failed", exc_info=True)

    while True:
        await asyncio.sleep(60)
        try:
            await asyncio.gather(update_voice_cache(), update_model_cache())
        except Exception:
            logging.warning("Cache refresh failed", exc_info=True)
