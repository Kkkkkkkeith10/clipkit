# clipkit

A small Python library for turning a narrated script into a captioned video.

**Started:** 24 Sep 2026


It is the editing half of a video pipeline I built for my own projects: once a script exists, the steps after it (shot list, text-to-speech, captions, rendering) barely depend on what the video is about. clipkit keeps those steps in one place so different kinds of video can share them.

```
script -> shots -> TTS -> captions -> render
```

## What is in it

| Module | Does |
|---|---|
| `clipkit/config.py` | Resolves the working root and `.env` once (`CLIPKIT_ROOT`, `CLIPKIT_ENV_FILE`) |
| `clipkit/profile.py` | Render profiles: `LANDSCAPE_720` and `VERTICAL_1080` (size, caption length, safe areas, supersampling) |
| `clipkit/pack.py` | Loads a prompt pack (stage list + templates) and checks placeholders when it loads |
| `clipkit/llm.py` | LLM client: provider selection, retries, staged calls with a cache and a self-repair loop |
| `clipkit/media/shots.py` | The shot list: what each shot says, measured or estimated timing, and where it lands |
| `clipkit/media/tts.py` | Narration audio per shot (edge-tts) |
| `clipkit/media/captions.py` | Caption cues sized to the profile; optional Chinese word segmentation so words are not split |
| `clipkit/media/render.py` | ffmpeg rendering: a still with narration, a slow pan/zoom, burned-in subtitles |

Not included: channel setups, prompt packs, image generation, uploading or scheduling. Those live in the projects that use the library.

## Install

Needs Python 3.11+, and `ffmpeg`/`ffprobe` on PATH for rendering.

```bash
pip install -e .[media]        # add [segment] for Chinese captions, [dev] for tests
python -m unittest discover -s tests -t .
```

The core (config, profiles, packs, LLM client) uses the standard library only.

## Configuration

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or the Azure variables). `.env` is git-ignored.

## Status

Early (0.1.0). The tests cover config, profiles, packs, the LLM client, shots and captions.


## Progress log

This public repo is a cleaned snapshot. The full development history stays private.

### September 2026

- 24 Sep: Started the library: config, render profiles and prompt packs.
- 24 Sep: Read shot lists and found runs of shots that can stand on their own.
- 24–25 Sep: Assembled a vertical cut from an already rendered video; generated vertical stills and burned in captions.
- 26 Sep: Added caption box styling and cutting at a chosen length, recording which length made which file.
- 28 Sep: Moved from cutting existing footage to writing short scripts in stages (inventory, angle, write, review), with the hook as its own stage.
- 29 Sep: Voiced scripts line by line with pauses; added a script-only review mode that skips image generation.
- 29 Sep: Narrowed clipkit down to a reusable library.
- 30 Sep: Merged it with related shorts code into one package with a shared test suite.

## Licence

MIT. See `LICENSE`.
