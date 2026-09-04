# analyze_video.py

Uses Gemini's video understanding API to answer a set of yes/no and 1-10
scale questions about a YouTube video, then scores the video with
configurable weights and writes a Markdown report. Can analyze a single
video, a whole list of videos, or a full "project" folder of videos +
questions.

## Setup

```bash
pip install google-genai pydantic
export GEMINI_API_KEY="your-api-key-here"
```

You'll need a Gemini API key from [Google AI Studio](https://aistudio.google.com/).

> **Note:** At the time of writing, analyzing YouTube videos through the
> Gemini API is free, but limited to 8 hours of video per day.

## The three ways to run it

### 1. A single video

```bash
python analyze_video.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Quote the URL — a `youtu.be` link with a `?si=...` tracking parameter can
otherwise be mangled by your shell. This:

- Sends the video + the default question set (`DEFAULT_QUESTIONS`, defined
  in the script) to Gemini
- Prints the structured JSON result to stdout
- Prints the total weighted score to stderr
- Writes a Markdown report named after the video's title (e.g.
  `my-video-title.md`) in the current directory

### 2. A list of videos (`-file`)

```bash
python analyze_video.py -file dataset.md
```

Analyzes every video listed in `dataset.md`. Any YouTube URL found in the
file is picked up — it works with either:

- **A plain list of URLs**, one per line, or
- **A Markdown table** with rows shaped like
  `| # | Film | Creator | YouTube URL |`

Anything else in the file (headers, prose, blank lines) is ignored.

Results are written into a new folder named after the input file — so
`-file dataset.md` writes into `./dataset/` — with one `<rank>.json` +
`<rank>.md` per video, plus a summary `scores.csv` (Rank, Title, Score,
YouTube URL, Status). The batch is resumable: re-running the same command
skips videos that already have a saved result, so an interrupted run can
just be run again. Use `--force` to re-analyze everything.

### 3. A project folder (`-dir`)

```bash
python analyze_video.py -dir my-project
```

The most convenient way to keep a question set and dataset together.
Point it at a folder containing:

```
my-project/
  questions.md   # the question set to use (see "Custom questions" below)
  dataset.md     # the videos to analyze (same formats as -file, above)
```

Both files are loaded automatically, and results are written back into
`my-project/` itself (one `<rank>.json` + `<rank>.md` per video, plus
`scores.csv`). This repo's `examples/` folder has two ready-made examples
you can copy as a starting point:

- `examples/xprize-top50/` — the top 50 films, default question set
- `examples/xprize-muyot/` — a different dataset with the XPRIZE question set

## Custom questions

By default, the script asks the 11 hard-coded questions in
`DEFAULT_QUESTIONS`. To use your own question set instead, write a file
(see `xprize-questions.md` for a full example) as a series of blocks
separated by a line containing only `---`:

```
question: "Is the video less than or equal to 3 minutes and 15 seconds in length?"
type: boolean
weight_true: 0.0
weight_false: -3.0
---
question: "Does this story think big about humanity's future?"
type: scale
weight: 1.0
```

Each block is one question:

| Field | Applies to | Meaning |
|---|---|---|
| `question` | both | The question text, in quotes. |
| `type` | both | `boolean` (default) or `scale`. |
| `weight_true` | `boolean` | Added to the score if Gemini answers yes. Default `1.0`. |
| `weight_false` | `boolean` | Added to the score if Gemini answers no. Default `-1.0`. |
| `weight` | `scale` | Multiplier on Gemini's 1-10 rating. Default `1.0` (the rating itself is the contribution). |

`scale` questions are always rated 1-10 by Gemini; a `range` field is
accepted in the file for documentation but has no effect.

Pass the file with `-questions` (works with all three modes above), or
put it in a project folder as `questions.md` and use `-dir`:

```bash
python analyze_video.py "https://www.youtube.com/watch?v=VIDEO_ID" -questions xprize-questions.md
python analyze_video.py -file dataset.md -questions xprize-questions.md
python analyze_video.py -dir my-project   # looks for my-project/questions.md automatically
```

## Other options

| Flag | Applies to | Meaning |
|---|---|---|
| `--model MODEL` | all | Gemini model to use. Default: `gemini-2.5-flash`. |
| `--temperature N` | all | Sampling temperature, `0.0`-`2.0`. Default `0.0` (most deterministic; raise it for more varied answers across repeated runs). |
| `--agentic` | all | Enable Gemini's agentic video processing (the model dynamically picks which parts of the video to look at, instead of ingesting it all at a fixed frame rate). Requires `--model gemini-3.5-flash-lite`, `gemini-3.6-flash`, or `gemini-3.7-flash`. |
| `--showprompt true\|false` | all | Print the full prompt sent to Gemini (video URL, model, numbered questions) to stderr before each call. Default `false`. |
| `--api-key KEY` | all | Gemini API key. Defaults to the `GEMINI_API_KEY` environment variable. |
| `--output PATH` | single-video only | Also write the JSON result to this path. |
| `--report PATH` | single-video only | Write the Markdown report to this path instead of the auto-generated name. |
| `--no-report` | single-video only | Skip writing the Markdown report. |
| `--output-dir DIR` | `-file` / `-dir` | Override the results folder (default: named after the `-file` argument, or the `-dir` argument itself). |
| `--csv PATH` | `-file` / `-dir` | Override the summary CSV path (default: `<output-dir>/scores.csv`). |
| `--limit N` | `-file` / `-dir` | Only process the first N videos. Handy for a quick test run before committing to a full batch. |
| `--delay N` | `-file` / `-dir` | Seconds to pause between videos, to be gentle on API rate limits. Default `2.0`. |
| `--force` | `-file` / `-dir` | Re-analyze videos that already have a saved result, instead of skipping them. |

## Scoring

Every answer contributes to a single total score for the video:

- **Boolean questions**: `weight_true` is added if Gemini answered yes,
  `weight_false` if no.
- **Scale questions**: Gemini's 1-10 rating, multiplied by `weight`, is
  added directly.

The Markdown report breaks down every question's answer, its contribution
to the score, Gemini's confidence, and its explanation (often with MM:SS
timestamps into the video).

## Troubleshooting

- **`No matches found` / a URL seems to fail oddly**: quote the URL on the
  command line. An unquoted `?si=...` or other `?`-containing URL can be
  interpreted by your shell as a glob pattern.
- **`--agentic` fails with `module 'google.genai.types' has no attribute
  'MediaProcessing'`**: your installed SDK predates agentic processing.
  Run `pip install --upgrade google-genai`.
- **Ratings on scale questions seem to cluster on one value**: this is a
  known behavior at `temperature=0.0` — the model can settle on its single
  most-likely token for a field across many similar-quality videos. Try
  raising `--temperature` slightly, or a different `--model`.
