import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from clipkit import llm

ENV = {"OPENAI_API_KEY": "sk-test"}


class TestClientConfig(unittest.TestCase):
    def test_openai_selected_when_key_present(self):
        client = llm.Client("sys", env=ENV)
        self.assertIn("api.openai.com", client.url)

    def test_azure_selected_without_openai_key(self):
        client = llm.Client("sys", env={"AZURE_OPENAI_ENDPOINT": "https://x.example",
                                        "AZURE_OPENAI_KEY": "k"})
        self.assertTrue(client.url.startswith("https://x.example/openai/v1/"))

    def test_missing_credentials_named(self):
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            llm.Client("sys", env={})

    def test_older_models_refused(self):
        for model in ("gpt-4o", "gpt-3.5-turbo", "gpt-35"):
            with self.assertRaisesRegex(ValueError, "disabled"):
                llm.Client("sys", model=model, env=ENV)

    def test_default_model(self):
        self.assertEqual(llm.Client("sys", env=ENV).model, llm.DEFAULT_MODEL)


class TestRequestBody(unittest.TestCase):
    def test_per_call_effort_overrides_client(self):
        """Research and review deserve high effort; a rewrite does not."""
        client = llm.Client("sys", env=ENV, effort="low")
        self.assertEqual(client.body("p", None, None)["reasoning_effort"], "low")
        self.assertEqual(client.body("p", None, "high")["reasoning_effort"], "high")

    def test_streams(self):
        """The gateway drops idle connections; long reasoning needs a stream."""
        self.assertTrue(llm.Client("sys", env=ENV).body("p", None, None)["stream"])

    def test_cache_hints_dropped_when_disabled(self):
        client = llm.Client("sys", env=ENV, cache_key="ep")
        self.assertIn("prompt_cache_key", client.body("p", None, None))
        client.caching = False
        self.assertNotIn("prompt_cache_key", client.body("p", None, None))

    def test_long_shared_prefix_is_split_for_caching(self):
        client = llm.Client("sys", env=ENV)
        client.cache_prefix = "x" * 4000
        blocks = client.user_blocks("x" * 4000 + "tail")
        self.assertEqual(len(blocks), 2)
        self.assertIn("prompt_cache_breakpoint", blocks[0])


class TestReadStream(unittest.TestCase):
    def test_assembles_chunks(self):
        lines = [
            b'data: {"model": "gpt-5.6-sol", "choices": [{"delta": {"content": "Hel"}}]}\n',
            b'data: {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}\n',
            b'data: {"usage": {"prompt_tokens": 5}, "choices": []}\n',
            b"data: [DONE]\n",
        ]
        text, finish, usage, served = llm.Client.read_stream(lines)
        self.assertEqual((text, finish, served), ("Hello", "stop", "gpt-5.6-sol"))
        self.assertEqual(usage["prompt_tokens"], 5)


class FakeClient:
    system = "sys"

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def ask(self, prompt, max_tokens=None, effort=None):
        self.calls.append((prompt, effort))
        return self.answers.pop(0)


class TestStages(unittest.TestCase):
    def test_caches_by_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(['{"a": 1}'])
            stages = llm.Stages(client, Path(tmp))
            self.assertEqual(stages.run("s", "prompt", lambda o: None), {"a": 1})
            self.assertEqual(stages.run("s", "prompt", lambda o: None), {"a": 1})
            self.assertEqual(len(client.calls), 1, "second run must be a cache hit")

    def test_changed_prompt_reruns(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(['{"a": 1}', '{"a": 2}'])
            stages = llm.Stages(client, Path(tmp))
            stages.run("s", "prompt one", lambda o: None)
            self.assertEqual(stages.run("s", "prompt two", lambda o: None), {"a": 2})

    def test_validation_error_is_fed_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(['{"a": 0}', '{"a": 1}'])

            def must_be_one(obj):
                if obj["a"] != 1:
                    raise ValueError("a must be 1")

            self.assertEqual(llm.Stages(client, Path(tmp)).run("s", "p", must_be_one), {"a": 1})
            self.assertIn("a must be 1", client.calls[1][0])

    def test_gives_up_after_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(['{"a": 0}'] * 3)

            def never(obj):
                raise ValueError("no")

            with self.assertRaisesRegex(ValueError, "three times"):
                llm.Stages(client, Path(tmp)).run("s", "p", never)
            self.assertEqual(len(list((Path(tmp) / "stages").glob("*.attempt*.txt"))), 3)

    def test_effort_passed_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(["plain text"])
            llm.Stages(client, Path(tmp)).run("s", "p", effort="high")
            self.assertEqual(client.calls[0][1], "high")


class TestParseObject(unittest.TestCase):
    def test_strips_code_fence(self):
        self.assertEqual(llm.parse_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_rejects_non_object(self):
        with self.assertRaises(ValueError):
            llm.parse_object("[1, 2]")



class TestResponsesApi(unittest.TestCase):
    def test_responses_is_the_default(self):
        """Chat Completions stays silent while reasoning; the gateway drops it at 60s."""
        self.assertEqual(llm.Client("sys", env=ENV).api, "responses")

    def test_body_asks_for_a_reasoning_summary(self):
        body = llm.Client("sys", env=ENV).responses_body("p", None, "high")
        self.assertEqual(body["reasoning"], {"effort": "high", "summary": "auto"})
        self.assertTrue(body["stream"])
        self.assertEqual(body["instructions"], "sys")

    def test_cache_key_carried(self):
        body = llm.Client("sys", env=ENV, cache_key="ep").responses_body("p", None, None)
        self.assertEqual(body["prompt_cache_key"], "ep")

    def test_bad_api_rejected(self):
        with self.assertRaisesRegex(ValueError, "api"):
            llm.Client("sys", env=ENV, api="grpc")


if __name__ == "__main__":
    unittest.main()
