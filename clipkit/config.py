"""Find the things that live outside the package: env files, credentials, ComfyUI.

the original pipeline discovered its own root in five separate places, each written as a
count of parent directories from the file doing the asking --
``SCRIPT_DIR.parents[1]`` in the uploader, ``ROOT.parents[1]`` in the pipeline,
``ROOT.parents[2]`` for the image backend. That only ever worked because every
one of those files sat at exactly ``<root>/workflow/prototype/``. Moving a file
one level changed what it resolved to, silently.

Nothing here counts parents. A path is either configured, or it is derived from
one configured root, and every lookup says out loud what it looked for when it
fails.

    from clipkit import config
    env = config.load_env()                 # merged .env, environment wins
    secret = config.client_secret()         # raises if 0 or 2+ candidates
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = "CLIPKIT_ENV_FILE"
ROOT_VAR = "CLIPKIT_ROOT"


def root() -> Path:
    """The working root: where .env lives.

    Defaults to the current directory rather than somewhere relative to this
    file, because the library is installed and the consumer is what has a root.
    """
    configured = os.environ.get(ROOT_VAR)
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines. Missing file is empty, not an error."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    # utf-8-sig: these files get edited in Windows editors that add a BOM, and a
    # BOM on the first line turns OPENAI_API_KEY into ﻿OPENAI_API_KEY.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and not key.startswith("#"):
            values[key.strip()] = value.strip().strip("\"'")
    return values


def load_env(path: Path | None = None) -> dict[str, str]:
    """Merge the env file with the real environment; the environment wins.

    ``CLIPKIT_ENV_FILE`` inside the file chains to a second file, so a project
    can point at credentials it already has instead of copying a key into a
    second place. The chained file is the lower-priority base: anything set in
    the primary file, or in the environment, overrides it.

    """
    primary = path or (root() / ".env")
    env = {**read_env_file(primary), **os.environ}
    linked = env.get(ENV_FILE)
    if not linked:
        return env
    source = Path(linked).expanduser()
    if not source.is_absolute():
        source = primary.resolve().parent / source
    if not source.is_file():
        raise FileNotFoundError(f"{ENV_FILE} points at {source}, which does not exist.")
    return {**read_env_file(source), **env}
