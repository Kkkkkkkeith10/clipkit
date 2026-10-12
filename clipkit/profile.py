"""Everything about a video that is geometry rather than content.

Three constants in the original pipeline decided the shape of every video it could ever
make, and none of them looked like a decision at the time:

    render_storyboard.py:19   WIDTH, HEIGHT, FPS = 1280, 720, 30
    render_storyboard.py:23   MAX_ZH_PER_CUE = 18
    schemas.py:50-58          disclaimer required; duration <= 30; visuals <= 40

They are fine for a 25-minute horizontal health video and wrong for everything
else. A vertical short is not 1280x720, its captions are not 18 characters at
64px, and it has neither a medical disclaimer nor three scenes. This module is
where those stop being constants.

A format owns a profile. Nothing else should be reading pixel dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd


@dataclass(frozen=True)
class RenderProfile:
    """Geometry, caption metrics and schema bounds for one output shape."""

    name: str
    width: int
    height: int
    fps: int = 30

    # --- captions -----------------------------------------------------------
    # Characters per subtitle cue. This is a legibility budget, not a language
    # fact: 18 was chosen for 64px text read at television distance by viewers
    # over 60. A phone held at arm's length wants fewer, larger characters.
    caption_units_per_cue: int = 18
    caption_font_size: int = 64
    caption_font: str = "Microsoft JhengHei"
    # BorderStyle 3 draws an opaque box behind the glyphs in the outline
    # colour, which is what keeps captions readable over a bright kitchen or a
    # white bedsheet. Dark text on a light box, not the reverse.
    caption_text_colour: str = "#111111"
    caption_box_colour: str = "#B9F6A6"
    caption_pad: int = 14
    caption_margin: int = 40
    # A cue shorter than this is merged into its neighbour -- below roughly a
    # second, a subtitle registers as a flicker rather than as words.
    min_cue_seconds: float = 0.90
    # Fraction of frame height kept clear at the bottom for captions. Image
    # prompts are written to keep faces out of it, so changing the shape of the
    # frame without revisiting this is how subjects end up behind text.
    caption_safe_area: float = 0.22

    # --- Ken Burns ----------------------------------------------------------
    # zoompan crops at integer source coordinates, so smoothness is set by how
    # many source pixels a move travels per frame. Below one, frames repeat and
    # then jump. Supersampling buys those pixels back; see media/render.py.
    supersample: int = 3
    oversample: float = 1.1
    pan_zoom: float = 1.22
    zoom_travel: float = 0.16

    # --- schema bounds ------------------------------------------------------
    min_scenes: int = 3
    max_visuals_per_scene: int = 40
    min_shot_seconds: float = 3.0
    max_shot_seconds: float = 30.0
    requires_disclaimer: bool = False

    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{self.name}: dimensions must be positive")
        # libx264 with yuv420p subsamples chroma by two in both directions, so an
        # odd dimension is rejected by the encoder rather than rounded.
        if self.width % 2 or self.height % 2:
            raise ValueError(f"{self.name}: {self.width}x{self.height} must be even for yuv420p")
        if self.fps <= 0:
            raise ValueError(f"{self.name}: fps must be positive")
        if not 0 <= self.caption_safe_area < 1:
            raise ValueError(f"{self.name}: caption_safe_area must be in [0, 1)")
        if self.min_shot_seconds > self.max_shot_seconds:
            raise ValueError(f"{self.name}: min_shot_seconds exceeds max_shot_seconds")
        if self.supersample < 1:
            raise ValueError(f"{self.name}: supersample must be at least 1")

    # --- derived ------------------------------------------------------------

    @property
    def aspect(self) -> tuple[int, int]:
        divisor = gcd(self.width, self.height)
        return self.width // divisor, self.height // divisor

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width

    @property
    def supersample_size(self) -> tuple[int, int]:
        """Source size a Ken Burns move is cropped from.

        Oversampling first, then supersampling, reproduces the original pipeline's 1408x792
        base for a 1280x720 output: the move is composed slightly wider than the
        frame so it has somewhere to travel, and that wider frame is rendered at
        SUPERSAMPLE times scale so it travels smoothly.
        """
        width = int(self.width * self.oversample) * self.supersample
        height = int(self.height * self.oversample) * self.supersample
        return width - width % 2, height - height % 2

    @property
    def caption_max_width(self) -> int:
        """Pixels a caption line may occupy before it runs off the frame.

        Captions never wrap (ASS WrapStyle 2), so a line wider than this does
        not fold -- it runs off both edges.
        """
        return self.width - 2 * (self.caption_margin + self.caption_pad)

    def frames(self, seconds: float) -> int:
        return max(1, round(seconds * self.fps))


# the original pipeline's shape, reproduced exactly. supersample_size is (4224, 2376), which
# is 1408x792 at 3x -- the values render_storyboard.py hardcodes today.
LANDSCAPE_720 = RenderProfile(
    name="landscape_720",
    width=1280,
    height=720,
    tags=("longform", "youtube"),
)

# Shorts. Two deliberate differences beyond the rotation:
#   * Nine characters per cue, not eighteen. A cue is read in one glance on a
#     phone, and the text is proportionally much larger.
#   * supersample 2, not 3. At 1.1 oversample a 3x vertical base is 3564x6336,
#     about 22 megapixels per zoompan frame; 2x holds smoothness at these
#     shorter shot lengths without the memory cost.
VERTICAL_1080 = RenderProfile(
    name="vertical_1080",
    width=1080,
    height=1920,
    # Eleven, not nine: at 96px inside 972px of usable width the frame holds
    # about ten glyphs, and a budget under that splits more often than it has
    # to -- nine broke 「腿痛」 across two cues on the first pass. fit() still
    # tightens it per line when Latin or digits make a line wider than it looks.
    caption_units_per_cue=11,
    caption_font_size=96,
    # Light blue rather than the green the long-form videos use. Same
    # lightness, so the dark text keeps its contrast; LANDSCAPE_720 stays
    # green to match the episodes already published.
    caption_box_colour="#A6DCF6",
    min_cue_seconds=0.70,
    caption_safe_area=0.28,
    supersample=2,
    min_scenes=1,
    max_visuals_per_scene=30,
    min_shot_seconds=1.0,
    max_shot_seconds=15.0,
    tags=("shorts", "youtube", "tiktok"),
)

BUILT_IN: dict[str, RenderProfile] = {p.name: p for p in (LANDSCAPE_720, VERTICAL_1080)}


def get(name: str) -> RenderProfile:
    try:
        return BUILT_IN[name]
    except KeyError:
        raise KeyError(f"No profile {name!r}. Known: {', '.join(sorted(BUILT_IN))}") from None
