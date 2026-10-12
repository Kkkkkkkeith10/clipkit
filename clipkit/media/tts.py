"""Narration: text to speech with word timings, checked against its own text.

Ported from the original pipeline's ``EdgeTTSProvider``. The edge-tts service fails in ways
that do not raise, and each check below exists because one of them reached a
published episode:

  * **Truncated audio.** It sometimes returns a fraction of a second for a full
    sentence. Episode 35 went to render with 0.36s of audio for a seventeen-
    second opening, and nothing noticed until a schema floor did.
  * **No word timings.** Full audio, empty boundaries -- which leaves that line
    with no captions in the finished video.
  * **A dropped sentence.** Audio length looks plausible but a short sentence
    is missing. The word timings say exactly what was spoken, so comparing
    them to the text catches what a length check cannot. Five published
    episodes lost a sentence this way.

Every clip is measured against its text and resynthesised until it passes.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from pathlib import Path

SPOKEN = re.compile(r"[㐀-鿿0-9A-Za-z]")
SILENT = set("，。！？、；：「」『』（）,.!?;:() \n\t")


@dataclass(frozen=True)
class Voice:
    name: str = "zh-CN-YunyangNeural"
    rate_percent: int = -28
    pitch_hz: int = -2

    @property
    def rate(self) -> str:
        return f"{self.rate_percent:+d}%"

    @property
    def pitch(self) -> str:
        return f"{self.pitch_hz:+d}Hz"

    @classmethod
    def from_plan(cls, plan: dict) -> "Voice":
        """Read the channel's voice from an episode's visual plan."""
        return cls(name=plan.get("voice", cls.name),
                   rate_percent=int(plan.get("voice_rate_percent", cls.rate_percent)),
                   pitch_hz=int(plan.get("voice_pitch_hz", cls.pitch_hz)))


class EdgeTTS:
    ATTEMPTS = 6
    BYTES_PER_SECOND = 6000   # edge-tts default: 24 kHz, 48 kbit/s mono MP3
    CHARS_PER_SECOND = 4.8    # Mandarin at +0%, scaled by the requested rate
    SHORT = 0.4               # under this share of the estimate is truncated

    def __init__(self, voice: Voice) -> None:
        self.voice = voice

    def expected_seconds(self, text: str) -> float:
        spoken = sum(1 for ch in text if ch not in SILENT)
        speed = 1 + self.voice.rate_percent / 100
        return spoken / (self.CHARS_PER_SECOND * max(speed, 0.3))

    def _stream(self, text: str) -> tuple[bytes, list[dict]]:
        import edge_tts

        async def once():
            speaker = edge_tts.Communicate(text=text, voice=self.voice.name,
                                           rate=self.voice.rate, pitch=self.voice.pitch)
            audio, marks = bytearray(), []
            async for chunk in speaker.stream():
                if chunk["type"] == "audio":
                    audio.extend(chunk["data"])
                elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                    marks.append(chunk)
            return bytes(audio), marks

        expected = self.expected_seconds(text)
        problem = ""
        for attempt in range(self.ATTEMPTS):
            try:
                audio, marks = asyncio.run(once())
                seconds = len(audio) / self.BYTES_PER_SECOND
                said = "".join(SPOKEN.findall("".join(m.get("text", "") for m in marks)))
                want = "".join(SPOKEN.findall(text))
                if expected >= 1.5 and not marks:
                    problem = "audio arrived without word timings"
                elif want and len(said) < len(want) * 0.97:
                    problem = f"spoke {len(said)} of {len(want)} characters"
                elif expected < 1.5 or seconds >= expected * self.SHORT:
                    return audio, marks
                else:
                    problem = f"{seconds:.2f}s of audio for about {expected:.1f}s of text"
            except edge_tts.exceptions.NoAudioReceived:
                problem = "no audio received"
            time.sleep(min(2 ** (attempt + 1), 30))
        raise RuntimeError(f"edge-tts failed {self.ATTEMPTS} times ({problem}): {text[:40]}")

    def speak(self, text: str, audio: Path, subtitles: Path) -> Path:
        """Write an MP3 and a word-timed SRT for one passage.

        The text is recorded beside the audio. A later edit to the script then
        cannot silently reuse this recording's duration for different words.
        """
        import edge_tts

        data, marks = self._stream(text)
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(data)
        maker = edge_tts.SubMaker()
        for chunk in marks:
            maker.feed(chunk)
        subtitles.write_text(maker.get_srt(), encoding="utf-8")
        audio.with_suffix(".txt").write_text(text, encoding="utf-8")
        return audio

    def is_current(self, text: str, audio: Path, subtitles: Path) -> bool:
        """Is there already a recording of exactly this text?"""
        record = audio.with_suffix(".txt")
        return (audio.is_file() and subtitles.is_file() and record.is_file()
                and record.read_text(encoding="utf-8") == text)
