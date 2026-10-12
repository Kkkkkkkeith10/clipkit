"""A prompt pack: everything the model is told, kept outside the code.

The library owns the machinery, the pack owns the words. the original pipeline fused them --
its master prompt was a file, which was right, but every per-stage instruction
was an f-string inside ``generate()``, where generic scaffolding and domain
judgement sat in the same paragraph:

    Respect each section's length budget: {allocation} {unit} in order, within
    20 per cent. ... Do not force pathological explanations for ordinary aging.

One of those sentences belongs to every video ever made. The other belongs to a
single kind of video. Splitting them is what lets a history pack reuse the first.

Templates use ``str.format`` syntax, so the JSON examples already written into
the original pipeline's prompts port unchanged -- ``{{"fixes": []}}`` stays a literal brace
and ``{greeting}`` stays a placeholder.

Two things are checked when a pack loads, both because the failure is otherwise
invisible until a script comes out wrong three stages later:

  * a template may only use placeholders the pack declares;
  * rendering a stage with a value missing is an error, not an empty string.

    pack = PromptPack.load(Path("packs/example"))
    prompt = pack.render("review", allocation="600, 700", unit="characters", ...)
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter

EFFORTS = ("none", "low", "medium", "high")


def placeholders_in(template: str) -> set[str]:
    """Field names in a format string, ignoring ``{{`` escapes.

    Formatter().parse yields None for the literal tail and for escaped braces,
    which is exactly the distinction a regex over ``{...}`` would get wrong on a
    prompt containing a JSON example.
    """
    return {name.split(".")[0].split("[")[0]
            for _, name, _, _ in Formatter().parse(template) if name}


@dataclass(frozen=True)
class Stage:
    """One model call in a pack's workflow."""

    name: str
    template: str
    effort: str = "medium"
    returns: str = "json"

    def __post_init__(self) -> None:
        if self.effort not in EFFORTS:
            raise ValueError(f"stage {self.name}: effort must be one of {', '.join(EFFORTS)}")
        if self.returns not in ("json", "text"):
            raise ValueError(f"stage {self.name}: returns must be 'json' or 'text'")


@dataclass(frozen=True)
class PromptPack:
    name: str
    path: Path
    system: str
    stages: tuple[Stage, ...]
    placeholders: frozenset[str]
    description: str = ""
    # Reference material a pack ships with -- a library of real hooks, a style
    # sheet -- loaded as text and filled into any template that names it.
    data: dict = field(default_factory=dict)

    # --- loading ------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "PromptPack":
        path = Path(path).resolve()
        manifest_path = path / "pack.toml"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No pack.toml in {path}")
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))

        for key in ("name", "system", "placeholders", "stages"):
            if key not in manifest:
                raise ValueError(f"{manifest_path}: missing '{key}'")

        system_path = path / manifest["system"]
        if not system_path.is_file():
            raise FileNotFoundError(f"{manifest_path}: system prompt {system_path} does not exist")
        system = system_path.read_text(encoding="utf-8")

        declared = frozenset(manifest["placeholders"])
        stages: list[Stage] = []
        seen: set[str] = set()
        for entry in manifest["stages"]:
            if "name" not in entry or "template" not in entry:
                raise ValueError(f"{manifest_path}: every stage needs 'name' and 'template'")
            if entry["name"] in seen:
                raise ValueError(f"{manifest_path}: duplicate stage {entry['name']!r}")
            seen.add(entry["name"])
            template_path = path / entry["template"]
            if not template_path.is_file():
                raise FileNotFoundError(
                    f"{manifest_path}: stage {entry['name']!r} template {template_path} does not exist")
            stages.append(Stage(
                name=entry["name"],
                template=template_path.read_text(encoding="utf-8"),
                effort=entry.get("effort", "medium"),
                returns=entry.get("returns", "json"),
            ))

        if not stages:
            raise ValueError(f"{manifest_path}: a pack needs at least one stage")

        data: dict[str, str] = {}
        for key, relative in (manifest.get("data") or {}).items():
            data_path = path / relative
            if not data_path.is_file():
                raise FileNotFoundError(f"{manifest_path}: data {key!r} file {data_path} does not exist")
            data[key] = data_path.read_text(encoding="utf-8")

        pack = cls(
            name=manifest["name"],
            path=path,
            system=system,
            stages=tuple(stages),
            placeholders=declared | frozenset(data),
            description=manifest.get("description", ""),
            data=data,
        )
        pack.check()
        return pack

    def check(self) -> None:
        """Reject a pack whose templates reach for values it never declared."""
        problems: list[str] = []
        for label, text in [("system.md", self.system),
                            *((f"stage {s.name}", s.template) for s in self.stages)]:
            undeclared = sorted(placeholders_in(text) - self.placeholders)
            if undeclared:
                problems.append(f"{label} uses undeclared {', '.join(undeclared)}")
        if problems:
            raise ValueError(
                f"Pack {self.name!r} at {self.path}:\n  "
                + "\n  ".join(problems)
                + "\nAdd them to 'placeholders' in pack.toml, or fix the template."
            )

    # --- use ----------------------------------------------------------------

    def stage(self, name: str) -> Stage:
        for candidate in self.stages:
            if candidate.name == name:
                return candidate
        raise KeyError(f"Pack {self.name!r} has no stage {name!r}. "
                       f"Has: {', '.join(s.name for s in self.stages)}")

    def render(self, name: str, /, **values: object) -> str:
        """Fill one stage's template. A missing value names itself."""
        stage = self.stage(name)
        values = {**self.data, **values}
        needed = placeholders_in(stage.template)
        missing = sorted(needed - values.keys())
        if missing:
            raise KeyError(f"Pack {self.name!r} stage {name!r} needs {', '.join(missing)}")
        return stage.template.format(**values)

    def render_system(self, **values: object) -> str:
        """Fill the system prompt, with the same missing-value check as a stage."""
        values = {**self.data, **values}
        missing = sorted(placeholders_in(self.system) - values.keys())
        if missing:
            raise KeyError(f"Pack {self.name!r} system prompt needs {', '.join(missing)}")
        return self.system.format(**values)

    @property
    def unused_placeholders(self) -> frozenset[str]:
        """Declared but never referenced -- harmless, but usually a typo."""
        used = placeholders_in(self.system)
        for stage in self.stages:
            used |= placeholders_in(stage.template)
        return self.placeholders - used
