"""Transport: talk to the model. Knows nothing about video.

Ported from the original pipeline's ``draft_script_v2.Client``, which had earned every one of
its odd-looking details in production:

  * **Streamed**, because the gateway closes an idle connection after about a
    minute, and a planning or review stage reasons for longer than that before
    its first byte. Buffered requests died as RemoteDisconnected and burned all
    four retries every time.
  * **Explicit cache boundaries.** Caching is on by default, and a write costs
    25% more than plain input, so it only pays where a prefix is actually read
    back. Left to the API, about 85% of what was written was never read again,
    which turned a saving into a surcharge.
  * **Cache hints are dropped, not fatal.** An endpoint that does not know the
    parameters rejects the whole request with a 400; losing the saving is
    acceptable, losing an overnight batch is not.
  * **Older models are refused**, including when the endpoint silently serves
    one in place of the requested model.

Two changes from the original. The constructor takes plain arguments instead of
an argparse namespace, and reasoning effort is per call rather than fixed for
the client's lifetime -- research and review deserve high effort, a rewrite of
one short passage does not, and a prompt pack declares which is which.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from clipkit import config

DISABLED = re.compile(r"gpt[-_ ]?(?:4|35|3\.5)", re.I)
RETRY_CODES = (429, 500, 502, 503, 504)
DEFAULT_MODEL = "gpt-5.6-sol"


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def save(path: Path, value) -> None:
    """Write beside, then replace, so a crash cannot leave half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else dump(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_object(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def cache_block(text: str, boundary: bool = True) -> dict:
    part = {"type": "text", "text": text}
    if boundary:
        part["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return part


def cache_summary(usage: list[dict]) -> dict:
    """How much input was read back from cache. Worth printing: the failure
    mode is silent, and the only other place it shows is the monthly bill."""
    total = sum(u.get("prompt_tokens") or u.get("input_tokens") or 0 for u in usage)
    cached = sum(((u.get("prompt_tokens_details") or u.get("input_tokens_details") or {})
                  .get("cached_tokens") or 0) for u in usage)
    return {"calls": len(usage), "input_tokens": total, "cached_tokens": cached,
            "cached_share": round(cached / total, 3) if total else 0.0}


PROVIDERS = ("auto", "openai", "azure")


def resolve(provider: str, model: str | None, env: dict[str, str]) -> tuple[str, str]:
    """Which connection and model a call uses.

    ``CLIPKIT_PROVIDER`` moves every stage at once. Credentials come from the
    environment or the git-ignored ``.env`` only.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}")
    if provider == "auto":
        provider = env.get("CLIPKIT_PROVIDER") or "auto"
    if provider == "auto":
        provider = ("azure" if not env.get("OPENAI_API_KEY") and env.get("AZURE_OPENAI_ENDPOINT")
                    else "openai")
    chosen = model or env.get("CLIPKIT_MODEL") or DEFAULT_MODEL
    return provider, chosen


class Client:
    def __init__(
        self,
        system: str,
        *,
        model: str | None = None,
        provider: str = "auto",
        effort: str = "medium",
        max_tokens: int = 8000,
        timeout: int = 240,
        cache_key: str = "",
        env: dict[str, str] | None = None,
        api: str | None = None,
    ) -> None:
        env = env if env is not None else config.load_env()
        self.provider, self.model = resolve(provider, model, env)
        # Responses API by default; CLIPKIT_API=chat overrides.
        api = api or env.get("CLIPKIT_API") or "responses"
        if api not in ("responses", "chat"):
            raise ValueError("api must be 'responses' or 'chat'")
        self.api = api
        azure = self.provider == "azure"
        if DISABLED.search(self.model):
            raise ValueError("GPT-4/3.5 are disabled. Choose GPT-5.6 or GPT-5.5.")
        if azure:
            missing = [k for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY") if not env.get(k)]
            if missing:
                raise ValueError("Missing configuration: " + ", ".join(missing))
            self.url = env["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/chat/completions"
            self.headers = {"api-key": env["AZURE_OPENAI_KEY"]}
        else:
            if not env.get("OPENAI_API_KEY"):
                raise ValueError(
                    "Set OPENAI_API_KEY, or configure an Azure endpoint. "
                    f"{config.ENV_FILE} can chain to an existing credential file.")
            self.url = "https://api.openai.com/v1/chat/completions"
            self.headers = {"Authorization": "Bearer " + env["OPENAI_API_KEY"]}
        self.headers["Content-Type"] = "application/json"
        self.system = system
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        # One key per piece of work, so its stages share a cache. Shared across
        # unrelated work it would only thrash.
        self.cache_key = cache_key
        self.cache_prefix = ""
        self.caching = True
        self.usage: list[dict] = []
        self.served_models: set[str] = set()

    # --- request building ---------------------------------------------------

    def user_blocks(self, prompt: str) -> list[dict]:
        prefix = self.cache_prefix
        # Under roughly a thousand tokens nothing is cached at all.
        if self.caching and prefix and len(prefix) > 3000 and prompt.startswith(prefix):
            return [cache_block(prefix), cache_block(prompt[len(prefix):], boundary=False)]
        return [cache_block(prompt, boundary=False)]

    def body(self, prompt: str, max_tokens: int | None, effort: str | None) -> dict:
        body = {
            "model": self.model,
            "reasoning_effort": effort or self.effort,
            "max_completion_tokens": max_tokens or self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {"role": "system", "content": [cache_block(self.system, self.caching)]},
                {"role": "user", "content": self.user_blocks(prompt)},
            ],
        }
        if self.caching:
            body["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
            if self.cache_key:
                body["prompt_cache_key"] = self.cache_key
        return body

    @staticmethod
    def read_stream(response) -> tuple[str, str | None, dict, str]:
        pieces, finish, usage, served = [], None, {}, ""
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            served = chunk.get("model") or served
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    pieces.append(piece)
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        return "".join(pieces), finish, usage, served

    # --- calls --------------------------------------------------------------

    def ask(self, prompt: str, *, max_tokens: int | None = None,
            effort: str | None = None) -> str:
        """One model call, over the Responses API by default.

        Chat Completions streams only the answer, not the thinking, so a call
        that reasons for longer than a minute sends nothing at all until then --
        and the gateway closes a connection silent for sixty seconds. That is
        exactly how the first 90-second short failed: RemoteDisconnected at
        60.2s, before a single byte, four attempts running. The Responses API
        streams events while the model reasons, and a requested reasoning
        summary keeps them coming, so a long think is no longer an idle line.
        """
        if self.api == "responses":
            return self.ask_responses(prompt, max_tokens=max_tokens, effort=effort)
        return self.ask_chat(prompt, max_tokens=max_tokens, effort=effort)

    def responses_body(self, prompt: str, max_tokens: int | None, effort: str | None) -> dict:
        body = {
            "model": self.model,
            "instructions": self.system,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "reasoning": {"effort": effort or self.effort, "summary": "auto"},
            "max_output_tokens": max_tokens or self.max_tokens,
            "stream": True,
            "store": False,
        }
        if self.caching and self.cache_key:
            body["prompt_cache_key"] = self.cache_key
        return body

    def ask_responses(self, prompt: str, *, max_tokens: int | None = None,
                      effort: str | None = None) -> str:
        url = self.url.replace("chat/completions", "responses")
        body = self.responses_body(prompt, max_tokens, effort)
        for attempt in range(4):
            request = urllib.request.Request(url, data=dump(body).encode("utf-8"),
                                             headers=self.headers)
            pieces: list[str] = []
            completed = None
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    for raw in response:
                        line = raw.decode("utf-8").strip()
                        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                            continue
                        event = json.loads(line[5:])
                        kind = event.get("type", "")
                        if kind == "response.output_text.delta":
                            pieces.append(event.get("delta", ""))
                        elif kind == "response.completed":
                            completed = event["response"]
                        elif kind in ("error", "response.failed", "response.incomplete"):
                            partial = event.get("response", {}) or {}
                            self.usage.append(partial.get("usage", {}) or {})
                            reason = ((partial.get("incomplete_details") or {}).get("reason")
                                      or (event.get("error") or partial.get("error") or {}).get("message")
                                      or kind)
                            raise ValueError(f"Response did not complete: {reason}. "
                                             "Raise max_tokens if it was truncated.")
                if completed is None:
                    raise ConnectionError("stream ended before the response completed")
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and self.caching:
                    self.caching = False
                    body = self.responses_body(prompt, max_tokens, effort)
                    print("Endpoint rejected the prompt-cache key; continuing without it.",
                          flush=True)
                    continue
                if exc.code == 404:
                    raise RuntimeError(
                        f"Model '{self.model}' is unavailable on this endpoint. "
                        "No fallback to an older model was attempted.") from None
                if exc.code not in RETRY_CODES or attempt == 3:
                    raise RuntimeError(f"API HTTP {exc.code}.") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.HTTPException) as exc:
                if attempt == 3:
                    raise RuntimeError("API connection failed after four attempts "
                                       f"({type(exc).__name__}: {exc}).") from None
            time.sleep(min(2 ** (attempt + 2), 30))

        self.usage.append(completed.get("usage", {}) or {})
        served = completed.get("model", "")
        if served:
            if DISABLED.search(served):
                raise ValueError(f"Endpoint returned disabled model '{served}'. No output accepted.")
            self.served_models.add(served)
        text = "".join(pieces)
        if not text.strip():
            # Some gateways send the text only in the final object.
            for item in completed.get("output", []):
                for content in item.get("content", []) or []:
                    if content.get("type") == "output_text":
                        text += content.get("text", "")
        if not text.strip():
            raise ValueError("API returned nothing (possibly a refusal).")
        return text.strip()

    def ask_chat(self, prompt: str, *, max_tokens: int | None = None,
                 effort: str | None = None) -> str:
        body = self.body(prompt, max_tokens, effort)
        content = finish = served = None
        usage: dict = {}
        for attempt in range(4):
            request = urllib.request.Request(self.url, data=dump(body).encode("utf-8"),
                                             headers=self.headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    content, finish, usage, served = self.read_stream(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and self.caching:
                    self.caching = False
                    body = self.body(prompt, max_tokens, effort)
                    print("Endpoint rejected the prompt-cache hints; continuing without them.",
                          flush=True)
                    continue
                if exc.code == 404:
                    raise RuntimeError(
                        f"Model '{self.model}' is unavailable on this endpoint. "
                        "No fallback to an older model was attempted.") from None
                if exc.code not in RETRY_CODES or attempt == 3:
                    raise RuntimeError(f"API HTTP {exc.code}.") from None
            # Named, not swallowed: a bare "connection failed" once hid a
            # gateway timeout for two whole runs.
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.HTTPException) as exc:
                if attempt == 3:
                    raise RuntimeError("API connection failed after four attempts "
                                       f"({type(exc).__name__}: {exc}).") from None
            time.sleep(min(2 ** (attempt + 2), 30))
        self.usage.append(usage)
        if served:
            if DISABLED.search(served):
                raise ValueError(f"Endpoint returned disabled model '{served}'. No output accepted.")
            self.served_models.add(served)
        if finish != "stop":
            raise ValueError(f"Response incomplete or filtered: {finish}. "
                             "Raise max_tokens if it was truncated.")
        if not (content or "").strip():
            raise ValueError("API returned nothing (possibly a refusal).")
        return content.strip()


class Stages:
    """Cache each successful call by its content, so reruns resume for free.

    The fingerprint covers the system prompt and the full request, so changing
    either -- a prompt pack edit included -- correctly forces that stage to run
    again, and nothing else. A result that fails validation is retried with the
    error fed back to the model, three times, and every raw attempt is kept on
    disk for when it still fails.
    """

    def __init__(self, client: Client, out: Path) -> None:
        self.client, self.out = client, Path(out)

    def run(self, name: str, prompt: str, validator=None, *,
            max_tokens: int | None = None, effort: str | None = None):
        fingerprint = hashlib.sha256((self.client.system + prompt).encode()).hexdigest()
        path = self.out / "stages" / f"{name}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                if validator:
                    validator(cached["result"])
                print(f"  reusing {name}", flush=True)
                return cached["result"]
        print(f"  generating {name}...", flush=True)
        correction = ""
        last_error = ""
        for attempt in range(3):
            raw = self.client.ask(prompt + correction, max_tokens=max_tokens, effort=effort)
            save(self.out / "stages" / f"{name}.attempt{attempt + 1}.txt", raw)
            try:
                result = parse_object(raw) if validator else raw
                if validator:
                    validator(result)
                save(path, {"fingerprint": fingerprint, "result": result})
                return result
            except (ValueError, KeyError, TypeError) as exc:
                last_error = str(exc)
                correction = (f"\nYour previous response failed validation: {exc}. "
                              "Return a corrected complete JSON object.")
        raise ValueError(f"{name} failed validation three times ({last_error}). "
                         f"Attempts saved in {self.out / 'stages'}.")
