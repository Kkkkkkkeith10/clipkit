import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from clipkit import config


class EnvCase(unittest.TestCase):
    """Each test gets its own root and a clean environment."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        patch = mock.patch.dict(os.environ, {config.ROOT_VAR: str(self.root)}, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self._tmp.cleanup)

    def write_env(self, text: str, name: str = ".env") -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path


class TestRoot(EnvCase):
    def test_root_from_environment(self):
        self.assertEqual(config.root(), self.root.resolve())

    def test_root_defaults_to_cwd(self):
        del os.environ[config.ROOT_VAR]
        self.assertEqual(config.root(), Path.cwd().resolve())


class TestLoadEnv(EnvCase):
    def test_reads_key_values_and_strips_quotes(self):
        self.write_env('OPENAI_API_KEY="abc"\n# comment\nMODEL=gpt-5.6-sol\n')
        env = config.load_env()
        self.assertEqual(env["OPENAI_API_KEY"], "abc")
        self.assertEqual(env["MODEL"], "gpt-5.6-sol")

    def test_comments_ignored(self):
        self.write_env("# NOPE=1\nYES=2\n")
        self.assertNotIn("NOPE", config.load_env())

    def test_bom_does_not_corrupt_first_key(self):
        """Windows editors add a BOM, which silently renames the first key."""
        (self.root / ".env").write_bytes("﻿OPENAI_API_KEY=abc\n".encode("utf-8"))
        self.assertEqual(config.load_env()["OPENAI_API_KEY"], "abc")

    def test_environment_overrides_file(self):
        self.write_env("MODEL=from-file\n")
        os.environ["MODEL"] = "from-environment"
        self.assertEqual(config.load_env()["MODEL"], "from-environment")

    def test_missing_file_is_not_an_error(self):
        self.assertIsInstance(config.load_env(), dict)


class TestChaining(EnvCase):
    def test_chained_file_supplies_base_values(self):
        self.write_env("OPENAI_API_KEY=chained\n", name="other.env")
        self.write_env(f"{config.ENV_FILE}=other.env\n")
        self.assertEqual(config.load_env()["OPENAI_API_KEY"], "chained")

    def test_primary_file_wins_over_chained(self):
        self.write_env("MODEL=chained\n", name="other.env")
        self.write_env(f"{config.ENV_FILE}=other.env\nMODEL=primary\n")
        self.assertEqual(config.load_env()["MODEL"], "primary")

    def test_relative_chain_resolves_beside_primary(self):
        nested = self.root / "creds"
        nested.mkdir()
        (nested / "keys.env").write_text("TOKEN=deep\n", encoding="utf-8")
        self.write_env(f"{config.ENV_FILE}=creds/keys.env\n")
        self.assertEqual(config.load_env()["TOKEN"], "deep")

    def test_broken_chain_says_where_it_looked(self):
        self.write_env(f"{config.ENV_FILE}=missing.env\n")
        with self.assertRaises(FileNotFoundError) as caught:
            config.load_env()
        self.assertIn("missing.env", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
