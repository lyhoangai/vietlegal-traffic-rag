"""src/tts/valtec_wrapper.py — Valtec TTS wrapper with graceful fallback."""
import os
import sys
import base64
import io
from dotenv import load_dotenv

load_dotenv()

VALTEC_PATH = os.getenv("VALTEC_TTS_PATH", r"D:\Valtec_AI\valtec-tts")
sys.path.insert(0, VALTEC_PATH)

# Fix namespace collision for "src"
valtec_src = os.path.join(VALTEC_PATH, "src")
if 'src' in sys.modules:
    if hasattr(sys.modules['src'], '__path__'):
        if valtec_src not in sys.modules['src'].__path__:
            sys.modules['src'].__path__.append(valtec_src)

try:
    from valtec_tts import TTS
    import soundfile as sf
    TTS_AVAILABLE = True
except Exception as e:
    TTS_AVAILABLE = False
    print(f"⚠️ Valtec TTS không tìm thấy hoặc lỗi import ({e}) — TTS disabled")


class ValtecTTSWrapper:
    """Wrapper around Valtec TTS with graceful fallback when unavailable."""

    def __init__(self, speaker: str = None):
        self.speaker = speaker or os.getenv("TTS_SPEAKER", "NF")
        self._tts = TTS() if TTS_AVAILABLE else None

    def is_available(self) -> bool:
        """Check if TTS engine is loaded and ready."""
        return TTS_AVAILABLE and self._tts is not None

    def synthesize_b64(self, text: str) -> str | None:
        """Synthesize text to speech and return base64-encoded WAV string.

        Returns None if TTS is not available.
        """
        if not self.is_available():
            return None
        audio, sr = self._tts.synthesize(text, speaker=self.speaker)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return base64.b64encode(buf.getvalue()).decode()

    def synthesize_bytes(self, text: str) -> bytes | None:
        """Synthesize text to speech and return raw WAV bytes.

        Returns None if TTS is not available.
        """
        if not self.is_available():
            return None
        audio, sr = self._tts.synthesize(text, speaker=self.speaker)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()

_shared_tts = None

def get_tts() -> ValtecTTSWrapper:
    """Get the shared ValtecTTSWrapper instance."""
    global _shared_tts
    if _shared_tts is None:
        _shared_tts = ValtecTTSWrapper()
    return _shared_tts
