import tempfile
import unittest
from pathlib import Path

from clipkit.pack import PromptPack, Stage, placeholders_in

MANIFEST = """
name = "demo"
description = "fixture"
system = "system.md"
placeholders = ["topic", "greeting", "allocation", "unit", "spare"]

[[stages]]
name = "brief"
template = "stages/brief.md"
effort = "high"

[[stages]]
name = "review"
template = "stages/review.md"
returns = "json"
"""


def write_pack(root: Path, manifest: str = MANIFEST, review: str | None = None) -> Path:
    path = root / "demo"
    (path / "stages").mkdir(parents=True)
    (path / "pack.toml").write_text(manifest, encoding="utf-8")
    (path / "system.md").write_text("You write about {topic}.", encoding="utf-8")
    (path / "stages" / "brief.md").write_text("Open with {greeting}.", encoding="utf-8")
    (path / "stages" / "review.md").write_text(
        review if review is not None
        else 'Budget {allocation} {unit}. Return {{"fixes": []}}.',
        encoding="utf-8")
    return path


class TestPlaceholderExtraction(unittest.TestCase):
    def test_escaped_braces_are_not_placeholders(self):
        """Prompts carry JSON examples; {{...}} must stay literal.

        A regex over {...} reports "fixes" here as a placeholder, which would
        make every prompt containing a JSON example fail to load.
        """
        found = placeholders_in('Return {{"fixes": []}} for {topic}.')
        self.assertEqual(found, {"topic"})

    def test_attribute_and_index_access(self):
        self.assertEqual(placeholders_in("{plan.title} and {rows[0]}"), {"plan", "rows"})

    def test_no_placeholders(self):
        self.assertEqual(placeholders_in("plain text"), set())


class TestLoading(unittest.TestCase):
    def test_loads_and_exposes_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = PromptPack.load(write_pack(Path(tmp)))
            self.assertEqual(pack.name, "demo")
            self.assertEqual([s.name for s in pack.stages], ["brief", "review"])

    def test_per_stage_effort(self):
        """Research and review deserve high effort; a rewrite usually does not."""
        with tempfile.TemporaryDirectory() as tmp:
            pack = PromptPack.load(write_pack(Path(tmp)))
            self.assertEqual(pack.stage("brief").effort, "high")
            self.assertEqual(pack.stage("review").effort, "medium", "default")

    def test_undeclared_placeholder_fails_to_load(self):
        """The whole point: a bad pack fails now, not three stages later."""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_pack(Path(tmp), review="Budget {allocation} for {mystery}.")
            with self.assertRaises(ValueError) as caught:
                PromptPack.load(path)
            self.assertIn("mystery", str(caught.exception))
            self.assertIn("stage review", str(caught.exception))

    def test_missing_template_file_named_in_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_pack(Path(tmp))
            (path / "stages" / "review.md").unlink()
            with self.assertRaises(FileNotFoundError) as caught:
                PromptPack.load(path)
            self.assertIn("review", str(caught.exception))

    def test_missing_pack_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "pack.toml"):
                PromptPack.load(Path(tmp))

    def test_duplicate_stage_rejected(self):
        manifest = MANIFEST + """
[[stages]]
name = "brief"
template = "stages/brief.md"
"""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "duplicate stage"):
                PromptPack.load(write_pack(Path(tmp), manifest))

    def test_unused_placeholder_is_reported_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = PromptPack.load(write_pack(Path(tmp)))
            self.assertEqual(pack.unused_placeholders, frozenset({"spare"}))


class TestRendering(unittest.TestCase):
    def test_render_fills_and_preserves_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = PromptPack.load(write_pack(Path(tmp)))
            out = pack.render("review", allocation="600, 700", unit="characters")
            self.assertIn("Budget 600, 700 characters.", out)
            self.assertIn('{"fixes": []}', out, "escaped braces must survive")

    def test_missing_value_names_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = PromptPack.load(write_pack(Path(tmp)))
            with self.assertRaises(KeyError) as caught:
                pack.render("review", allocation="600")
            self.assertIn("unit", str(caught.exception))

    def test_extra_values_are_harmless(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = PromptPack.load(write_pack(Path(tmp)))
            out = pack.render("brief", greeting="朋友們大家好", topic="ignored")
            self.assertEqual(out, "Open with 朋友們大家好.")

    def test_unknown_stage_lists_known(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = PromptPack.load(write_pack(Path(tmp)))
            with self.assertRaises(KeyError) as caught:
                pack.render("nope")
            self.assertIn("brief", str(caught.exception))


class TestStage(unittest.TestCase):
    def test_bad_effort_rejected(self):
        with self.assertRaisesRegex(ValueError, "effort"):
            Stage(name="x", template="", effort="maximum")

    def test_bad_returns_rejected(self):
        with self.assertRaisesRegex(ValueError, "returns"):
            Stage(name="x", template="", returns="yaml")


if __name__ == "__main__":
    unittest.main()
