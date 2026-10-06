---
name: manuscript-voice
description: >
  Author and render voice-lab manuscript JSON files: a format that defines
  named voices (consented WAV/MP3 references) and an ordered list of lines
  that each cite a voice. Use when writing, editing, validating, or
  synthesizing a manuscript for generate_manuscript.py, or when a user asks
  to turn a script or dialogue into speech with Pocket TTS Raven.
---

# Manuscript JSON format

A manuscript is one JSON file. It declares the voices that may speak and an
ordered list of lines. `generate_manuscript.py` reads the file and produces
one MP3 per line, in order.

## Top-level fields

| Field        | Type   | Required | Meaning |
|--------------|--------|----------|---------|
| `version`    | number | no       | Format version. Must be `1` if present. |
| `title`      | string | no       | Human-readable name. Default: the file stem. |
| `output_dir` | string | no       | Where audio is written. Relative to the manuscript file. Default: `out`. |
| `voices`     | object | yes      | Map of voice name to voice definition. At least one voice. |
| `settings`   | object | no       | Global synthesis settings applied to every line. |
| `lines`      | array  | yes      | Ordered list of lines. At least one line. |

## Voice definitions

Each key in `voices` is a voice name. The name must use filename-safe
letters, digits, `.`, `_`, or `-` (it becomes the default voice ID).

| Field        | Type   | Required | Meaning |
|--------------|--------|----------|---------|
| `reference`  | string | yes      | Path to a consented WAV or MP3 reference recording. Relative to the manuscript file. The file must exist. |
| `voice_id`   | string | no       | Native voice ID used by the server. Default: the voice name. Same filename-safe rules. |
| (overrides)  | mixed  | no       | Any of the synthesis setting keys below, applied to every line spoken by this voice. |

## Settings keys

Usable in `settings`, in a voice definition, or on a single line. A line
override wins over a voice override, which wins over global `settings`.
Allowed keys: `precision` (`"int8"` or `"fp32"`), `temperature` (number),
`lsd_steps` (positive integer), `threads` (non-negative integer),
`keep_commas` (boolean). No other keys are allowed.

Note: each unique combination of settings starts its own Raven server
session, so manuscripts render fastest when all lines share one settings set.

## Lines

Each entry in `lines` is an object:

| Field        | Type   | Required | Meaning |
|--------------|--------|----------|---------|
| `voice`      | string | yes      | Name of a declared voice. |
| `text`       | string | yes      | Sentence to speak. Must not be empty. Double quotes are removed before synthesis (Raven stability). |
| (overrides)  | mixed  | no       | Any of the setting keys, for this line only. |

Line numbering starts at 1 and follows array order. Output files are named
`NNN-<voice_id>.mp3` (with `001` for the first line).

## Example

```json
{
  "version": 1,
  "title": "Welcome call",
  "output_dir": "out",
  "settings": { "precision": "int8" },
  "voices": {
    "narrator": {
      "reference": "voices/reginald-ashworth.wav"
    },
    "deja": {
      "reference": "voices/deja-thoris.wav"
    }
  },
  "lines": [
    { "voice": "narrator", "text": "Welcome to voice-lab." },
    { "voice": "deja", "text": "Thank you for having me." },
    { "voice": "narrator", "text": "Our next topic is manuscript rendering." }
  ]
}
```

## Rendering

```sh
# Validate and print the plan without synthesizing
python3 generate_manuscript.py manuscript.json --dry-run

# Render: one MP3 per line, plus one joined file
python3 generate_manuscript.py manuscript.json --concat out/welcome.mp3
```

Options: `--out-dir` overrides `output_dir`, `--concat` joins all line MP3s
into one file, `--keep-wav` keeps the intermediate PCM16 WAV files,
`--dry-run` validates only. Rendering requires the Raven runtime; run
`python3 tools/setup_raven.py --check-installation` first if it is missing.

## Validation rules

The script fails with a clear error when:

- the file is missing, is not valid JSON, or uses another `version`
- `voices` or `lines` is missing, empty, or the wrong JSON type
- a `reference` path does not resolve to an existing WAV or MP3 file
- a line cites a voice name that is not declared in `voices`
- a `voice` or `text` value is missing, empty, or not a string
- a voice name or `voice_id` is not filename-safe
- a settings key is unknown

Prefer fixing the manuscript over editing the script. Run `--dry-run` after
every edit before spending synthesis time.
