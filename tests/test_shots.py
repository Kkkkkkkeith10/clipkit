import json
import tempfile
import unittest
from pathlib import Path

from clipkit.media.shots import ShotDocument, format_timecode, parse_timecode

DOCUMENT = {
    "project_id": "demo_v1",
    "title": "示範",
    "visual_style": "photorealistic",
    "scenes": [
        {
            "id": "SCN01",
            "title": "第一章",
            "accent": "#55D6BE",
            "shots": [
                {"id": "SCN01_VIS01", "starts_at": "00:00.00", "seconds": 8.5,
                 "timing": "spoken", "text": "朋友們大家好。", "prompt": "a kitchen"},
                {"id": "SCN01_VIS02", "starts_at": "00:08.50", "seconds": 6.0,
                 "timing": "spoken", "text": "今天談抽筋。", "prompt": "a leg",
                 "visual_role": "elderly_story"},
            ],
        },
        {
            "id": "SCN02",
            "title": "第二章",
            "accent": "#FFB865",
            "shots": [
                {"id": "SCN02_VIS01", "starts_at": "00:14.50", "seconds": 7.25,
                 "timing": "estimated", "text": "所以要補鈣嗎？"},
            ],
        },
    ],
}


def write(root: Path) -> Path:
    path = root / "shots.json"
    path.write_text(json.dumps(DOCUMENT, ensure_ascii=False), encoding="utf-8")
    return path


class TestTimecode(unittest.TestCase):
    def test_round_trip(self):
        for seconds in (0.0, 8.5, 59.99, 128.48):
            self.assertAlmostEqual(parse_timecode(format_timecode(seconds)), seconds, places=2)

    def test_minutes_exceed_sixty(self):
        """Episodes run past an hour and the format stays MM:SS."""
        self.assertEqual(parse_timecode("81:30.00"), 81 * 60 + 30)
        self.assertEqual(format_timecode(81 * 60 + 30), "81:30.00")

    def test_accepts_hh_mm_ss(self):
        self.assertEqual(parse_timecode("01:21:30.00"), 3600 + 21 * 60 + 30)

    def test_rejects_nonsense(self):
        with self.assertRaises(ValueError):
            parse_timecode("930")


class TestLoading(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = write(Path(self._tmp.name))
        self.doc = ShotDocument.load(self.path)

    def test_flattens_shots_in_order(self):
        self.assertEqual([s.id for s in self.doc],
                         ["SCN01_VIS01", "SCN01_VIS02", "SCN02_VIS01"])
        self.assertEqual(len(self.doc), 3)

    def test_scene_context_attached_to_shots(self):
        self.assertEqual(self.doc.by_id("SCN01_VIS02").scene_id, "SCN01")
        self.assertEqual(self.doc.by_id("SCN01_VIS02").scene_title, "第一章")

    def test_timings(self):
        self.assertTrue(self.doc.by_id("SCN01_VIS01").is_spoken)
        self.assertFalse(self.doc.by_id("SCN02_VIS01").is_spoken)
        self.assertFalse(self.doc.all_spoken, "one shot is still estimated")

    def test_start_and_end(self):
        shot = self.doc.by_id("SCN01_VIS02")
        self.assertAlmostEqual(shot.start, 8.5)
        self.assertAlmostEqual(shot.end, 14.5)

    def test_durations(self):
        self.assertAlmostEqual(self.doc.seconds, 21.75)
        self.assertAlmostEqual(self.doc.scenes[0].seconds, 14.5)

    def test_narration_is_every_word_joined(self):
        self.assertEqual(self.doc.narration(), "朋友們大家好。今天談抽筋。所以要補鈣嗎？")

    def test_unknown_shot_names_the_project(self):
        with self.assertRaises(KeyError) as caught:
            self.doc.by_id("SCN09_VIS01")
        self.assertIn("demo_v1", str(caught.exception))


class TestRoundTrip(unittest.TestCase):
    def test_save_preserves_unknown_keys(self):
        """Later stages add card/visual_role/negative; a reload must keep them."""
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp))
            ShotDocument.load(path).save()
            reloaded = json.loads(path.read_text(encoding="utf-8"))
            shot = reloaded["scenes"][0]["shots"][1]
            self.assertEqual(shot["visual_role"], "elderly_story")

    def test_save_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp))
            ShotDocument.load(path).save()
            leftovers = list(Path(tmp).glob("*.tmp"))
            self.assertEqual(leftovers, [], "temp file should be replaced, not left")

    def test_missing_prompt_is_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp))
            ShotDocument.load(path).save()
            reloaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("prompt", reloaded["scenes"][1]["shots"][0])


if __name__ == "__main__":
    unittest.main()
