"""Voice: Twilio speech recognition in, ElevenLabs TTS out.

The turn-based prototype: Twilio <Gather input="speech"> transcribes, the agent runs, ElevenLabs
synthesises, and Twilio <Play>s the audio. Latency budget is about 1.5s per turn, so the agent
must be fast (small model, no ReAct loops on voice).
Stretch: stream with Twilio Media Streams + a realtime speech model for barge-in support.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import httpx


class TTSCache:
    """In-memory LRU of synthesised audio. Production: blob storage + CDN, keyed by hash."""

    def __init__(self, max_items: int = 256) -> None:
        self._data: OrderedDict[str, bytes] = OrderedDict()
        self.max_items = max_items

    def put(self, text: str, audio: bytes) -> str:
        key = hashlib.sha256(text.encode()).hexdigest()[:24]
        self._data[key] = audio
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)
        return key

    def get(self, key: str) -> bytes | None:
        return self._data.get(key)


class ElevenLabsTTS:
    def __init__(self, api_key: str, voice_id: str, model: str = "eleven_flash_v2_5",
                 client: httpx.Client | None = None) -> None:
        self.key, self.voice, self.model = api_key, voice_id, model
        self.http = client or httpx.Client(timeout=10)

    def synthesize(self, text: str) -> bytes:
        r = self.http.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}",
            params={"output_format": "mp3_22050_32"},
            headers={"xi-api-key": self.key, "accept": "audio/mpeg"},
            json={"text": text, "model_id": self.model,
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}})
        r.raise_for_status()
        return r.content
