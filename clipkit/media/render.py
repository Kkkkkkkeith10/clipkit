"""ffmpeg building blocks every format renders with.

A still with its narration and a slow move, subtitles burned in, and the one
place ffmpeg is invoked. Moved here from the cut-down format, which the
shortform format was borrowing them from; both now import them from here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from clipkit.profile import RenderProfile


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        tail = (result.stderr or "").strip().splitlines()[-6:]
        raise RuntimeError("ffmpeg failed:\n  " + "\n  ".join(tail))


def motion_filter(motion: str, frames: int, profile: RenderProfile) -> str:
    """Ken Burns expressions for zoompan, at this profile's amplitude.

    zoompan crops at integer source coordinates, so a move is smooth only when
    it travels at least one source pixel per frame. That is what the
    supersampled base is for; the zoom is a straight function of the frame
    number rather than an increment, so its speed does not depend on how long
    the shot happens to be.
    """
    span = max(1, frames - 1)
    centre_x, centre_y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    travel, pan = profile.zoom_travel, profile.pan_zoom
    if motion in ("punch", "punch_up"):
        # Halfway through, cut to a tighter frame of the same picture and keep
        # drifting. One still then reads as two shots: a change of view every
        # few seconds without generating a second image. punch_up anchors the
        # tight frame on the upper third, where a subject's face or the top of
        # a bowl usually is.
        half = max(1, span // 2)
        drift, jump = travel * 0.4, 1.30
        zoom = (f"if(lt(on,{half}),1+{drift}*on/{half},"
                f"{jump}+{drift}*(on-{half})/{half})")
        anchor_y = centre_y if motion == "punch" else "(ih-ih/zoom)*0.30"
        return f"z='{zoom}':x='{centre_x}':y='{anchor_y}'"
    if motion == "pan_up":
        return f"z='{pan}':x='{centre_x}':y='(ih-ih/zoom)*(1-on/{span})'"
    if motion == "pan_down":
        return f"z='{pan}':x='{centre_x}':y='(ih-ih/zoom)*on/{span}'"
    if motion == "slow_zoom_out":
        return f"z='{1 + travel}-{travel}*on/{span}':x='{centre_x}':y='{centre_y}'"
    return f"z='1+{travel}*on/{span}':x='{centre_x}':y='{centre_y}'"


def shot_from_still(image: Path, audio: Path, destination: Path, seconds: float,
                    motion: str, profile: RenderProfile) -> Path:
    """One clip: a generated still, its narration, and a slow move."""
    frames = profile.frames(seconds)
    base_w, base_h = profile.supersample_size
    video = (
        f"scale={base_w}:{base_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={base_w}:{base_h},"
        f"zoompan={motion_filter(motion, frames, profile)}:"
        f"d={frames}:s={profile.width}x{profile.height}:fps={profile.fps},"
        # zoompan downscales its supersampled crop with a bilinear scaler and
        # the move magnifies part of the source; both cost detail.
        f"unsharp=5:5:0.8,format=yuv420p"
    )
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(image),
        "-i", str(audio), "-t", f"{seconds:.3f}", "-vf", video,
        "-r", str(profile.fps), "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-ar", "48000", "-ac", "2", "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
        "-shortest", str(destination),
    ])
    return destination


def burn(video: Path, subtitles: Path, destination: Path) -> Path:
    """Burn an ASS file in. Windows paths need escaping for the filter parser."""
    escaped = str(subtitles.resolve()).replace("\\", "/").replace(":", "\\:")
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
        "-vf", f"subtitles='{escaped}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p", "-c:a", "copy", str(destination),
    ])
    return destination
