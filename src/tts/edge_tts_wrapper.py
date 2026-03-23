"""Small wrapper around the optional edge-tts package."""

from __future__ import annotations

import os
import re

try:
    import edge_tts
except ImportError:  # pragma: no cover - exercised through route fallback
    edge_tts = None


DEFAULT_VOICE = os.getenv("EDGE_TTS_VOICE", "vi-VN-HoaiMyNeural")
DEFAULT_RATE = os.getenv("EDGE_TTS_RATE", "+0%")
_VOICE_CACHE: dict[str, list[dict[str, str]]] = {}
_VI_VN_FALLBACK_VOICES = [
    {
        "short_name": "vi-VN-HoaiMyNeural",
        "display_name": "Microsoft HoaiMy Online (Natural) - Vietnamese (Vietnam)",
        "locale": "vi-VN",
        "gender": "Female",
    },
    {
        "short_name": "vi-VN-NamMinhNeural",
        "display_name": "Microsoft NamMinh Online (Natural) - Vietnamese (Vietnam)",
        "locale": "vi-VN",
        "gender": "Male",
    },
]


class TTSUnavailableError(RuntimeError):
    """Raised when the selected TTS backend is not available."""


def _ensure_edge_tts() -> None:
    if edge_tts is None:
        raise TTSUnavailableError(
            "edge-tts is not installed. Run `pip install edge-tts` first."
        )


def normalize_tts_text(text: str) -> str:
    """Strip UI formatting before sending text to the speech service."""
    cleaned = (text or "").replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"[*_`#>\[\]\(\)]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


async def list_edge_voices(locale: str = "vi-VN") -> list[dict[str, str]]:
    """Return a compact voice list, preferring the requested locale."""
    _ensure_edge_tts()

    try:
        raw_voices = await edge_tts.list_voices()
        filtered = [
            {
                "short_name": voice["ShortName"],
                "display_name": voice.get("FriendlyName") or voice["ShortName"],
                "locale": voice["Locale"],
                "gender": voice.get("Gender", ""),
            }
            for voice in raw_voices
            if not locale or voice.get("Locale") == locale
        ]
        filtered.sort(key=lambda voice: voice["short_name"])
        _VOICE_CACHE[locale] = filtered
        return filtered
    except Exception as err:
        if locale in _VOICE_CACHE:
            return _VOICE_CACHE[locale]
        if locale == "vi-VN":
            return list(_VI_VN_FALLBACK_VOICES)
        raise TTSUnavailableError(f"Edge TTS voice list failed: {err}") from err


async def synthesize_edge_tts_bytes(
    text: str,
    voice: str | None = None,
    rate: str | None = None,
) -> bytes:
    """Generate a full MP3 payload from edge-tts."""
    _ensure_edge_tts()

    normalized_text = normalize_tts_text(text)
    if not normalized_text:
        raise ValueError("No text provided for TTS.")

    try:
        payload = bytearray()
        communicate = edge_tts.Communicate(
            normalized_text,
            voice=voice or DEFAULT_VOICE,
            rate=rate or DEFAULT_RATE,
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                payload.extend(chunk["data"])

        if not payload:
            raise TTSUnavailableError("Edge TTS returned no audio.")
        return bytes(payload)
    except TTSUnavailableError:
        raise
    except Exception as err:
        raise TTSUnavailableError(f"Edge TTS synthesis failed: {err}") from err
