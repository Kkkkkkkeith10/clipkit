"""Burned-in subtitles, in the original pipeline's house style, at any frame size.

Ported from ``render_storyboard.py``, which had the geometry compiled in:
1280x720, twelve characters a cue, 96px. All three move to the profile here,
because a short is watched on a phone and read in one glance.

The style itself is kept: dark text on an opaque light box (ASS BorderStyle 3),
bottom-centred, never wrapping. The box is not decoration -- narration plays
over bright kitchens and white bedsheets, and outlined text disappears on both.

Two subtleties worth keeping, both learned the expensive way upstream:

  * A character budget alone is not enough. 「HbA1c」 and 「8000」 are single
    cues by character count and far too wide to fit, so every line is also
    *measured* and the budget tightened until it does.
  * Cue time is apportioned by character count across the split parts, then
    each cue is held for a readable minimum and snapped shut against the next.
    Without the minimum, a three-character part flashes for a tenth of a
    second.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from clipkit.profile import RenderProfile

SRT_PATTERN = re.compile(
    r"\d+\s*\n(\d\d:\d\d:\d\d[,.]\d+)\s*-->\s*"
    r"(\d\d:\d\d:\d\d[,.]\d+)\s*\n(.*?)(?=\n\s*\n|\Z)",
    re.S,
)
PUNCTUATION = "，。！？；：、"
TRAILING = "，。、；：,.;:"
FONT_CANDIDATES = (r"C:\Windows\Fonts\msjhbd.ttc", r"C:\Windows\Fonts\msjh.ttc")


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def seconds(self) -> float:
        return self.end - self.start


def srt_seconds(value: str) -> float:
    hours, minutes, rest = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(rest)


def parse_srt(path: Path) -> list[Cue]:
    text = Path(path).read_text(encoding="utf-8-sig")
    return [
        Cue(srt_seconds(start), srt_seconds(end), " ".join(body.split()).strip())
        for start, end, body in SRT_PATTERN.findall(text)
        if body.strip()
    ]


def word_boundaries(text: str) -> set[int] | None:
    """Offsets where a Chinese word ends, or None if no segmenter is installed.

    Chinese has no spaces, so a character budget alone cuts through words:
    「這不是要您每次腿痛都嚇得睡不著」 split evenly lands between 腿 and 痛 and
    puts half a word on each line. Only segmentation fixes that properly.

    jieba is optional on purpose. It is the standard tool and pure Python, but
    a caption pipeline should not hard-fail without it -- absent, splitting
    falls back to the punctuation-and-balance rule, which is what the original pipeline has
    always done.
    """
    try:
        import jieba
    except ImportError:
        return None
    offsets, position = set(), 0
    for token in jieba.cut(text, cut_all=False):
        position += len(token)
        offsets.add(position)
    return offsets


def split_text(text: str, limit: int) -> list[str]:
    """Break a line into parts of at most ``limit`` characters.

    Preference order for a cut: Chinese punctuation near the balance point, a
    word boundary near it when a segmenter is available, then the balance
    point itself. A Latin word or number is never cut, so 「HbA1c」 and 「8000」
    survive whichever rule applies.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    boundaries = word_boundaries(text)
    consumed = 0
    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        target = min(limit, max(8, len(rest) // ((len(rest) + limit - 1) // limit)))
        candidates = [i + 1 for i, char in enumerate(rest[:limit]) if char in PUNCTUATION]
        if candidates:
            cut = min(candidates, key=lambda x: abs(x - target))
        elif boundaries:
            # Boundaries are absolute offsets in the original string.
            nearby = [offset - consumed for offset in boundaries
                      if 0 < offset - consumed <= limit]
            cut = min(nearby, key=lambda x: abs(x - target)) if nearby else target
        else:
            cut = target
        while (cut > 1 and rest[cut - 1].isascii() and rest[cut - 1].isalnum()
               and rest[cut].isascii() and rest[cut].isalnum()):
            cut -= 1
        parts.append(rest[:cut].strip())
        consumed += cut
        rest = rest[cut:]
        stripped = rest.lstrip()
        consumed += len(rest) - len(stripped)
        rest = stripped
    if rest:
        parts.append(rest)
    return [part for part in parts if part]


def _font(profile: RenderProfile):
    from PIL import ImageFont
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, profile.caption_font_size)
    return ImageFont.load_default(profile.caption_font_size)


def rendered_width(text: str, font, spacing: int = 2) -> int:
    """Width in pixels, including the per-glyph spacing ASS adds."""
    return font.getbbox(text)[2] + spacing * len(text)


def fit(text: str, profile: RenderProfile) -> list[str]:
    """Split until every part fits one line of this frame."""
    font = _font(profile)
    limit = profile.caption_units_per_cue
    parts = split_text(text, limit)
    while limit > 4 and any(
            rendered_width(part, font) > profile.caption_max_width for part in parts):
        limit -= 1
        parts = split_text(text, limit)
    return parts


def normalize(cues: list[Cue], duration: float, profile: RenderProfile) -> list[Cue]:
    """Split every cue to fit, share its time out, and close the gaps."""
    expanded: list[Cue] = []
    for cue in cues:
        parts = fit(cue.text, profile)
        total = sum(len(part) for part in parts) or 1
        used = 0
        for index, part in enumerate(parts):
            start = cue.start + cue.seconds * used / total
            used += len(part)
            end = (cue.end if index == len(parts) - 1
                   else cue.start + cue.seconds * used / total)
            expanded.append(Cue(start, end, part.rstrip(TRAILING)))
    for index, cue in enumerate(expanded):
        following = expanded[index + 1].start if index + 1 < len(expanded) else duration
        cue.end = min(max(cue.end, cue.start + profile.min_cue_seconds), following)
        # A gap under half a second reads as a flicker, not a pause.
        if 0 < following - cue.end <= 0.40:
            cue.end = following
    return [cue for cue in expanded if cue.text and cue.end > cue.start + 0.05]


def shift(cues: list[Cue], offset: float) -> list[Cue]:
    return [Cue(cue.start + offset, cue.end + offset, cue.text) for cue in cues]


def ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def ass_colour(hex_colour: str) -> str:
    """ASS wants &HBBGGRR, the reverse of CSS."""
    value = hex_colour.lstrip("#")
    return f"&H00{value[4:6]}{value[2:4]}{value[0:2]}"


def ass_escape(text: str) -> str:
    return text.replace("{", "（").replace("}", "）").replace("\n", " ")


def write_ass(cues: list[Cue], path: Path, profile: RenderProfile, *,
              extra_styles: list[str] | None = None,
              extra_events: list[str] | None = None) -> Path:
    """Write a subtitle file sized to this profile's frame.

    ``PlayResX/Y`` must match the video, or libass scales every size and
    margin by the ratio and the captions come out wrong.

    ``extra_styles`` and ``extra_events`` let a format put other text on screen
    -- headlines, labels -- in the same file, so everything is burned in one
    pass and shares one clock.
    """
    text = ass_colour(profile.caption_text_colour)
    box = ass_colour(profile.caption_box_colour)
    # Alignment 2 is bottom-centre; MarginV lifts it clear of the frame edge.
    margin_v = max(profile.caption_margin,
                   int(profile.height * profile.caption_safe_area * 0.35))
    style = (
        f"Style: Zh,{profile.caption_font},{profile.caption_font_size},"
        f"{text},{text},{box},&H00000000,-1,0,0,0,100,100,2,0,3,"
        f"{profile.caption_pad},0,2,{profile.caption_margin},"
        f"{profile.caption_margin},{margin_v},1"
    )
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {profile.width}",
        f"PlayResY: {profile.height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        style,
        *(extra_styles or []),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    lines += [
        f"Dialogue: 0,{ass_time(cue.start)},{ass_time(cue.end)},Zh,,0,0,0,,"
        f"{ass_escape(cue.text)}"
        for cue in cues
    ]
    lines += extra_events or []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def for_shots(timings: Path, shot_ids: list[str], durations: list[float],
              profile: RenderProfile) -> list[Cue]:
    """Assemble one timeline from the per-shot SRTs an episode already has.

    Each shot's cues are relative to that shot, so they are shifted by the
    running total rather than by the shot's position in the source episode --
    a cutdown starts at zero.
    """
    out: list[Cue] = []
    offset = 0.0
    for shot_id, seconds in zip(shot_ids, durations):
        source = Path(timings) / f"{shot_id}.srt"
        if source.is_file():
            out += shift(normalize(parse_srt(source), seconds, profile), offset)
        offset += seconds
    return out
