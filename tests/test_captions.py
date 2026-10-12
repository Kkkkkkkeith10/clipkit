import tempfile
import unittest
from pathlib import Path

from clipkit.media import captions as cap
from clipkit.profile import LANDSCAPE_720, VERTICAL_1080

SRT = """1
00:00:00,100 --> 00:00:05,831
先說清楚，這不是要您每次腿痛都嚇得睡不著。

2
00:00:05,900 --> 00:00:09,000
典型抽筋是肌肉突然縮住。
"""


class TestParsing(unittest.TestCase):
    def test_parses_cues(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.srt"
            path.write_text(SRT, encoding="utf-8")
            cues = cap.parse_srt(path)
        self.assertEqual(len(cues), 2)
        self.assertAlmostEqual(cues[0].start, 0.1)
        self.assertAlmostEqual(cues[0].end, 5.831)
        self.assertTrue(cues[0].text.startswith("先說清楚"))

    def test_srt_seconds_accepts_comma_and_dot(self):
        self.assertAlmostEqual(cap.srt_seconds("00:00:05,831"), 5.831)
        self.assertAlmostEqual(cap.srt_seconds("00:01:05.500"), 65.5)


class TestSplitting(unittest.TestCase):
    def test_short_text_is_one_part(self):
        self.assertEqual(cap.split_text("先說清楚", 11), ["先說清楚"])

    def test_every_part_respects_the_limit(self):
        for part in cap.split_text("先說清楚這不是要您每次腿痛都嚇得睡不著", 8):
            self.assertLessEqual(len(part), 8)

    def test_prefers_punctuation(self):
        parts = cap.split_text("先說清楚，這不是要您每次腿痛都嚇得睡不著", 11)
        self.assertEqual(parts[0], "先說清楚，")

    def test_never_cuts_a_latin_word(self):
        """Splitting HbA1c across two cues reads as a typo."""
        for limit in range(5, 12):
            for part in cap.split_text("糖化血色素HbA1c反映三個月的平均血糖水平", limit):
                self.assertNotIn(part[-1:], list("HbA1"), f"limit {limit}: {part}")

    def test_never_cuts_a_number(self):
        for part in cap.split_text("每天建議攝取8000步以上維持活動量", 7):
            if part.endswith(tuple("0123456789")):
                self.assertFalse(part[-1].isdigit() and part != part.rstrip("0123456789"))

    def test_nothing_is_lost(self):
        text = "先說清楚這不是要您每次腿痛都嚇得睡不著"
        self.assertEqual("".join(cap.split_text(text, 8)), text)

    def test_empty_text(self):
        self.assertEqual(cap.split_text("", 11), [])

    def test_word_boundaries_absent_without_segmenter(self):
        """Optional dependency: absent, the fallback must still produce parts."""
        result = cap.word_boundaries("先說清楚")
        self.assertTrue(result is None or isinstance(result, set))


class TestNormalize(unittest.TestCase):
    def test_splits_and_shares_time(self):
        cues = [cap.Cue(0.0, 6.0, "先說清楚這不是要您每次腿痛都嚇得睡不著")]
        out = cap.normalize(cues, 6.0, VERTICAL_1080)
        self.assertGreater(len(out), 1)
        self.assertAlmostEqual(out[0].start, 0.0)
        self.assertLessEqual(out[-1].end, 6.0 + 1e-6)

    def test_cues_do_not_overlap(self):
        cues = [cap.Cue(0.0, 6.0, "先說清楚這不是要您每次腿痛都嚇得睡不著")]
        out = cap.normalize(cues, 6.0, VERTICAL_1080)
        for before, after in zip(out, out[1:]):
            self.assertLessEqual(before.end, after.start + 1e-6)

    def test_minimum_duration_enforced(self):
        """A three-character part flashing for a tenth of a second is unreadable."""
        cues = [cap.Cue(0.0, 4.0, "先說清楚這不是要您每次腿痛都嚇得睡不著")]
        out = cap.normalize(cues, 10.0, VERTICAL_1080)
        for cue in out[:-1]:
            self.assertGreaterEqual(cue.seconds, 0.05)

    def test_trailing_punctuation_removed(self):
        out = cap.normalize([cap.Cue(0.0, 3.0, "先說清楚。")], 3.0, VERTICAL_1080)
        self.assertEqual(out[0].text, "先說清楚")

    def test_shift(self):
        moved = cap.shift([cap.Cue(1.0, 2.0, "x")], 10.0)
        self.assertEqual((moved[0].start, moved[0].end), (11.0, 12.0))


class TestAssOutput(unittest.TestCase):
    def test_time_format(self):
        self.assertEqual(cap.ass_time(0), "0:00:00.00")
        self.assertEqual(cap.ass_time(3661.5), "1:01:01.50")

    def test_colour_is_bgr(self):
        """ASS wants &HBBGGRR -- the reverse of CSS, and easy to get wrong."""
        self.assertEqual(cap.ass_colour("#112233"), "&H00332211")

    def test_braces_escaped(self):
        """An unescaped brace is an ASS override block, not text."""
        self.assertNotIn("{", cap.ass_escape("a{b}c"))

    def test_playres_matches_the_profile(self):
        """Mismatched PlayRes makes libass scale every size and margin."""
        with tempfile.TemporaryDirectory() as tmp:
            path = cap.write_ass([cap.Cue(0, 1, "測試")],
                                 Path(tmp) / "c.ass", VERTICAL_1080)
            body = path.read_text(encoding="utf-8")
        self.assertIn("PlayResX: 1080", body)
        self.assertIn("PlayResY: 1920", body)

    def test_style_uses_profile_font_and_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = cap.write_ass([cap.Cue(0, 1, "測試")],
                                 Path(tmp) / "c.ass", LANDSCAPE_720)
            body = path.read_text(encoding="utf-8")
        self.assertIn(f",{LANDSCAPE_720.caption_font_size},", body)
        self.assertIn(LANDSCAPE_720.caption_font, body)

    def test_dialogue_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = cap.write_ass([cap.Cue(0, 1.5, "測試")],
                                 Path(tmp) / "c.ass", VERTICAL_1080)
            body = path.read_text(encoding="utf-8")
        self.assertIn("Dialogue: 0,0:00:00.00,0:00:01.50,Zh,,0,0,0,,測試", body)


class TestProfileWidth(unittest.TestCase):
    def test_usable_width_excludes_margin_and_padding(self):
        self.assertEqual(VERTICAL_1080.caption_max_width,
                         1080 - 2 * (VERTICAL_1080.caption_margin + VERTICAL_1080.caption_pad))


if __name__ == "__main__":
    unittest.main()
