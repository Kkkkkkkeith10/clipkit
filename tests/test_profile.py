import unittest

from clipkit.profile import LANDSCAPE_720, VERTICAL_1080, RenderProfile, get


class TestGeometry(unittest.TestCase):
    def test_landscape_reproduces_original_pipeline(self):
        """The 16:9 profile must be the original pipeline's renderer, exactly.

        1408x792 at 3x supersample is what render_storyboard.py hardcodes; if
        this drifts, ported videos stop matching the ones already published.
        """
        self.assertEqual((LANDSCAPE_720.width, LANDSCAPE_720.height), (1280, 720))
        self.assertEqual(LANDSCAPE_720.fps, 30)
        self.assertEqual(LANDSCAPE_720.supersample_size, (1408 * 3, 792 * 3))
        self.assertEqual(LANDSCAPE_720.caption_units_per_cue, 18)

    def test_aspect_ratios(self):
        self.assertEqual(LANDSCAPE_720.aspect, (16, 9))
        self.assertEqual(VERTICAL_1080.aspect, (9, 16))

    def test_orientation(self):
        self.assertFalse(LANDSCAPE_720.is_vertical)
        self.assertTrue(VERTICAL_1080.is_vertical)

    def test_vertical_captions_are_shorter(self):
        """A phone cue is read in one glance; the budget must not be inherited."""
        self.assertLess(VERTICAL_1080.caption_units_per_cue,
                        LANDSCAPE_720.caption_units_per_cue)
        self.assertGreater(VERTICAL_1080.caption_font_size,
                           LANDSCAPE_720.caption_font_size)

    def test_supersample_size_is_even(self):
        """zoompan feeds libx264/yuv420p, which rejects odd dimensions."""
        for candidate in (LANDSCAPE_720, VERTICAL_1080):
            width, height = candidate.supersample_size
            self.assertEqual((width % 2, height % 2), (0, 0), candidate.name)

    def test_frames(self):
        self.assertEqual(LANDSCAPE_720.frames(2.0), 60)
        self.assertEqual(LANDSCAPE_720.frames(0.0), 1, "a shot is never zero frames")


class TestValidation(unittest.TestCase):
    def test_odd_dimensions_rejected(self):
        with self.assertRaisesRegex(ValueError, "even for yuv420p"):
            RenderProfile(name="odd", width=1281, height=720)

    def test_negative_dimensions_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            RenderProfile(name="bad", width=0, height=720)

    def test_safe_area_bounds(self):
        with self.assertRaisesRegex(ValueError, "caption_safe_area"):
            RenderProfile(name="bad", width=1280, height=720, caption_safe_area=1.0)

    def test_shot_bounds_ordered(self):
        with self.assertRaisesRegex(ValueError, "min_shot_seconds"):
            RenderProfile(name="bad", width=1280, height=720,
                          min_shot_seconds=20, max_shot_seconds=10)

    def test_disclaimer_is_not_universal(self):
        """schemas.py made this required; an ad or a short has none."""
        self.assertFalse(VERTICAL_1080.requires_disclaimer)


class TestLookup(unittest.TestCase):
    def test_get_known(self):
        self.assertIs(get("vertical_1080"), VERTICAL_1080)

    def test_get_unknown_lists_options(self):
        with self.assertRaisesRegex(KeyError, "landscape_720"):
            get("nope")


if __name__ == "__main__":
    unittest.main()
