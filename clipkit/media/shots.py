"""The shot list: what is said, when, and for how long.

``shots.json`` is the authoring document for everything visual. It exists
because the original pipeline got the ordering wrong -- image prompts were
written per chapter before anyone knew what each shot would say, then matched
to narration afterwards by keyword, so roughly half the images showed something
the voice was not talking about. The order that works is:

    script -> shots -> TTS -> per-shot prompt -> image -> render

Every shot therefore carries the exact sentence it speaks, how long the voice
really takes to say it, and where it lands in the finished video.

Two fields deserve care:

``timing``
    ``spoken`` means ``seconds`` was measured from a real recording.
    ``estimated`` means it was derived from a character count and will move once
    the audio exists. A cutdown must only trust ``spoken``, because it is
    slicing a video that already exists.

``starts_at``
    ``MM:SS.ss`` from the start of the episode, and it can pass an hour, at
    which point it is still minutes -- ``81:30.00``, not ``01:21:30.00``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


def parse_timecode(value: str) -> float:
    """``MM:SS.ss`` or ``HH:MM:SS.ss`` to seconds."""
    parts = value.strip().split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"Bad timecode {value!r}")
    seconds = float(parts[-1])
    seconds += int(parts[-2]) * 60
    if len(parts) == 3:
        seconds += int(parts[0]) * 3600
    return seconds


def format_timecode(seconds: float) -> str:
    """Inverse of :func:`parse_timecode`, in the minutes-can-exceed-60 form."""
    if seconds < 0:
        raise ValueError("Negative timecode")
    minutes, remainder = divmod(round(seconds, 2), 60)
    return f"{int(minutes):02d}:{remainder:05.2f}"


@dataclass
class Shot:
    id: str
    starts_at: str
    seconds: float
    timing: str
    text: str
    prompt: str | None = None
    scene_id: str = ""
    scene_title: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def start(self) -> float:
        return parse_timecode(self.starts_at)

    @property
    def end(self) -> float:
        return self.start + self.seconds

    @property
    def is_spoken(self) -> bool:
        """Was this duration measured, or guessed from a character count?"""
        return self.timing == "spoken"

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "starts_at": self.starts_at,
            "seconds": self.seconds,
            "timing": self.timing,
            "text": self.text,
        }
        if self.prompt is not None:
            out["prompt"] = self.prompt
        # Keys written by later stages -- card, visual_role, negative -- are
        # carried through rather than dropped, so a round trip is lossless.
        out.update(self.extra)
        return out


@dataclass
class Scene:
    id: str
    title: str
    accent: str
    shots: list[Shot]

    @property
    def seconds(self) -> float:
        return sum(shot.seconds for shot in self.shots)


@dataclass
class ShotDocument:
    project_id: str
    title: str
    visual_style: str
    scenes: list[Scene]
    source: Path | None = None

    # --- loading ------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "ShotDocument":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        scenes: list[Scene] = []
        known = {"id", "starts_at", "seconds", "timing", "text", "prompt"}
        for entry in raw.get("scenes", []):
            shots = [
                Shot(
                    id=shot["id"],
                    starts_at=shot["starts_at"],
                    seconds=float(shot["seconds"]),
                    timing=shot.get("timing", "estimated"),
                    text=shot["text"],
                    prompt=shot.get("prompt"),
                    scene_id=entry["id"],
                    scene_title=entry.get("title", ""),
                    extra={k: v for k, v in shot.items() if k not in known},
                )
                for shot in entry.get("shots", [])
            ]
            scenes.append(Scene(
                id=entry["id"],
                title=entry.get("title", ""),
                accent=entry.get("accent", ""),
                shots=shots,
            ))
        return cls(
            project_id=raw.get("project_id", ""),
            title=raw.get("title", ""),
            visual_style=raw.get("visual_style", ""),
            scenes=scenes,
            source=path,
        )

    def save(self, path: Path | None = None) -> Path:
        target = Path(path) if path else self.source
        if target is None:
            raise ValueError("No path to save to")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "project_id": self.project_id,
            "title": self.title,
            "visual_style": self.visual_style,
            "scenes": [
                {
                    "id": scene.id,
                    "title": scene.title,
                    "accent": scene.accent,
                    "shots": [shot.to_json() for shot in scene.shots],
                }
                for scene in self.scenes
            ],
        }
        # Write beside, then replace, so an interrupted save cannot leave a
        # truncated shots.json where the pipeline expects the authoring document.
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(target)
        return target

    # --- access -------------------------------------------------------------

    @property
    def shots(self) -> list[Shot]:
        return [shot for scene in self.scenes for shot in scene.shots]

    def __iter__(self) -> Iterator[Shot]:
        return iter(self.shots)

    def __len__(self) -> int:
        return len(self.shots)

    def by_id(self, shot_id: str) -> Shot:
        for shot in self.shots:
            if shot.id == shot_id:
                return shot
        raise KeyError(f"No shot {shot_id!r} in {self.project_id}")

    def index_of(self, shot_id: str) -> int:
        for index, shot in enumerate(self.shots):
            if shot.id == shot_id:
                return index
        raise KeyError(f"No shot {shot_id!r} in {self.project_id}")

    @property
    def seconds(self) -> float:
        return sum(shot.seconds for shot in self.shots)

    @property
    def all_spoken(self) -> bool:
        """True when every duration came from a recording.

        The cutdown format refuses to slice a document that is still estimating,
        because the timecodes it would cut on are not the ones in the rendered
        video.
        """
        return bool(self.shots) and all(shot.is_spoken for shot in self.shots)

    def narration(self) -> str:
        """Every spoken word, in order, with nothing between.

        The batch runner asserts this equals the script's own parsed narration;
        it is what catches a shot list that drifted from the words recorded.
        """
        return "".join(shot.text for shot in self.shots)
