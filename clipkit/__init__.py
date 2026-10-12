"""clipkit -- turn a script into a finished video, in whatever shape you need.

The pipeline splits at the script. Everything downstream of it -- cutting into
shots, narrating, generating a still per shot, rendering with motion and
captions and rendering -- barely knows what the video is about,
and that half is this library. Everything upstream is specific to a kind of
video, and lives in a format plus a prompt pack.

    from clipkit import config, profile
    from clipkit.pack import PromptPack
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import config, profile
from .pack import PromptPack, Stage
from .profile import RenderProfile

__all__ = ["config", "profile", "PromptPack", "Stage", "RenderProfile", "__version__"]
