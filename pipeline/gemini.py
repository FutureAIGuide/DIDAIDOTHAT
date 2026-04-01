"""
pipeline/gemini.py — Thin wrapper around the Google Generative AI SDK.

Keeps all Gemini interaction in one place so every module can call
`call_gemini(prompt)` without worrying about initialisation.
"""
from __future__ import annotations

import logging

import config

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            import google.generativeai as genai  # type: ignore[import]

            genai.configure(api_key=config.GEMINI_API_KEY)
            _client = genai.GenerativeModel(config.GEMINI_MODEL)
        except Exception as exc:
            logger.error("Failed to initialise Gemini client: %s", exc)
            raise
    return _client


def call_gemini(prompt: str) -> str:
    """
    Send a prompt to Gemini and return the text response.
    Raises RuntimeError if the API key is not configured.
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set — cannot call Gemini.")
    try:
        client = _get_client()
        response = client.generate_content(prompt)
        return response.text
    except Exception as exc:
        logger.error("Gemini call failed: %s", exc)
        raise
