# Built-in voice references

Two in-house reference voices, originally created for the guten-speak project
and recorded by the project owners themselves (first-party recordings; the
voice owners granted consent for cloning use in their own projects).

| Voice | File | SHA-256 |
| --- | --- | --- |
| Reginald Ashworth (male) | `reginald-ashworth.wav` | `9e24b0e5b576c6af321ec33cfa5a651cd945f2f41151c125ea05302120a129c1` |
| Deja Thoris (female) | `deja-thoris.wav` | `2ffb58f4d1e2c58cffe2c36fbf9004fbb97b3f5292430c0345583dea23dc14fd` |

Format: mono PCM16, 44100 Hz, ~12 s each. `generate_voice.py` conditions a
reference to mono 24000 Hz on first use and caches the derivative next to it
(`.raven-cache/`, gitignored).

These are user-supplied voice assets, not Raven installation artifacts: see
`docs/pocket-tts-raven-distribution-review.md` for the consent requirements
that apply to any other reference added here.
