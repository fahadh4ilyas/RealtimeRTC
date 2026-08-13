"""
On-demand model / voice fetchers.

Lists are fetched per client-supplied API key and base URL (not from any global
environment key), so each client only sees the models/voices their own key can
access.
"""

import aiohttp

from realtimertc import config


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def fetch_models(api_key: str, base_url: str):
    """Fetch model ids from the LLM base URL.

    Returns (models, error, status): models is a list of ids; error is a message
    string (None on success); status is the HTTP status (0 for transport errors).
    """
    url = config.llm_models_api(base_url)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=_headers(api_key)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m.get("id") for m in data.get("data", [])
                              if m.get("id")]
                    return models, None, 200
                return [], await resp.text(), resp.status
    except Exception as exc:
        return [], str(exc), 0


async def fetch_voices(api_key: str, base_url: str):
    """Fetch voice names from the TTS base URL.

    Returns (voices, error, status) with the same shape as fetch_models.
    """
    url = config.tts_voices_api(base_url)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=_headers(api_key)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("voices", []), None, 200
                return [], await resp.text(), resp.status
    except Exception as exc:
        return [], str(exc), 0
