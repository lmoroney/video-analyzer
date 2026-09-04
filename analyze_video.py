#!/usr/bin/env python3
# Copyright 2026 Laurence Moroney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
analyze_video.py

Uses Gemini's video understanding API (google-genai SDK) to answer a fixed
set of questions (DEFAULT_QUESTIONS below) about a YouTube video, returning
a structured JSON result, a weighted score, and a Markdown report.

Most questions are yes/no. The last four are "scale" questions: instead of
true/false, Gemini rates how strongly the statement holds on a 1-10 scale,
and that rating (times the question's `weight` multiplier, default 1.0)
is added directly to the total score.

Docs: https://ai.google.dev/gemini-api/docs/video-understanding

Setup
-----
    pip install google-genai pydantic
    export GEMINI_API_KEY="your-api-key-here"

Usage
-----
    # Analyze a single video
    python analyze_video.py "https://www.youtube.com/watch?v=VIDEO_ID"

    # Pick a different model or write the JSON result to a file
    python analyze_video.py "https://www.youtube.com/watch?v=VIDEO_ID" \\
        --model gemini-2.5-pro --output result.json

    # Control the Markdown report's path (by default a .md report is always
    # written, named after the video's title)
    python analyze_video.py "https://www.youtube.com/watch?v=VIDEO_ID" \\
        --report my_report.md

    # Runs default to temperature 0.0 (most deterministic). Raise it if you
    # want more varied answers, e.g.:
    python analyze_video.py "https://www.youtube.com/watch?v=VIDEO_ID" \\
        --temperature 0.4

    # Batch mode: analyze every video listed in a file instead of a single
    # URL. Any YouTube URL found anywhere in the file is picked up -- a
    # plain list of URLs (one per line) works, and so does a Markdown table
    # with rows shaped like '| # | Film | Creator | YouTube URL |'. Results
    # are written into a new folder named after the input file (so
    # "-file muyot-dataset.md" writes into a "muyot-dataset" folder), one
    # .json + .md per video plus a summary scores.csv.
    python analyze_video.py -file muyot-dataset.md

    # Batch mode options: override the output folder, only do the first N
    # entries, and re-run entries that already have a saved result.
    python analyze_video.py -file muyot-dataset.md --output-dir results --limit 5 --force

    # Use a custom question set instead of the hard-coded DEFAULT_QUESTIONS.
    # The file is a series of blocks separated by '---', each block being
    # 'key: value' lines (see xprize-questions.md for the reference format):
    #
    #   question: "Is the video less than or equal to 3 minutes and 15 seconds in length?"
    #   type: boolean
    #   weight_true: 0.0
    #   weight_false: -3.0
    #   ---
    #   question: "Does this story think big about humanity's future?"
    #   type: scale
    #   weight: 1.0
    #
    # Works with either mode:
    python analyze_video.py "https://www.youtube.com/watch?v=VIDEO_ID" -questions xprize-questions.md
    python analyze_video.py -file muyot-dataset.md -questions xprize-questions.md

    # Try Gemini's agentic video processing (Sept 2026): the model picks
    # which parts of the video to look at instead of ingesting it all at a
    # fixed rate. Only gemini-3.5-flash-lite, gemini-3.6-flash, and
    # gemini-3.7-flash support it, so pick one of those with --model.
    python analyze_video.py "https://www.youtube.com/watch?v=VIDEO_ID" \\
        --model gemini-3.7-flash --agentic

    # Project mode: point at a directory containing questions.md (the
    # custom question set) and dataset.md (the videos to analyze), and
    # both are loaded automatically. Results are written into that same
    # directory, e.g.:
    #
    #   muyot-dataset/
    #     questions.md
    #     dataset.md
    #
    python analyze_video.py -dir muyot-dataset

    # By default the full prompt sent to Gemini is NOT printed. Turn it on
    # with --showprompt true (works in single-video, -file, and -dir modes).
    python analyze_video.py -dir muyot-dataset --showprompt true
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Default questions.
#
# response_type "boolean" (the default): Gemini answers true/false, and the
#   question's weight_if_true / weight_if_false is added to the score.
#
# response_type "scale": Gemini answers with an integer 1-10 rating of how
#   strongly the statement holds, and rating * weight is added to the score
#   (weight defaults to 1.0, i.e. the rating itself is the contribution).
# ---------------------------------------------------------------------------

@dataclass
class Question:
    text: str
    response_type: str = "boolean"  # "boolean" or "scale"
    weight_if_true: float = 1.0     # used when response_type == "boolean"
    weight_if_false: float = -1.0   # used when response_type == "boolean"
    weight: float = 1.0             # multiplier on the 1-10 rating, when response_type == "scale"


DEFAULT_QUESTIONS: List[Question] = [
    Question(
        "Is the video less than or equal to 3 minutes and 15 seconds in length?",
        weight_if_true=0.0,
        weight_if_false=-3.0,
    ),
    Question(
        "Are the last 15 seconds comprised of an end card containing sponsors "
        "like Google, XPRIZE, Jed McCaleb, and Salesforce?",
        weight_if_true=0.0,
        weight_if_false=-10.0,
    ),
    Question(
        "Is the content of this video science fiction?",
        weight_if_true=0.0,
        weight_if_false=-10.0,
    ),
    Question(
        "Does this portray a hopeful, optimistic, technology-forward vision "
        "of humanity's future?",
        weight_if_true=10.0,
        weight_if_false=-10.0,
    ),
    Question(
        "Is technology meaningfully integrated into the narrative? (Not "
        "just background,)",
        weight_if_true=5.0,
        weight_if_false=-3.0,
    ),
    Question(
        "Is there explicit violence, language, or sexual content in the video?",
        weight_if_true=-10.0,
        weight_if_false=0.0,
    ),
    Question(
        "Other than the end card, are there any recognizable brands used in "
        "the video?",
        weight_if_true=-15.0,
        weight_if_false=0.0,
    ),
    # --- Last four questions: rated 1-10 instead of true/false ---
    Question(
        "Does this video portray a compelling story, that is well-realized "
        "within production constraints?",
        response_type="scale",
    ),
    Question(
        "Does this story think big about humanity's future?",
        response_type="scale",
    ),
    Question(
        "Is this story fully aligned with the mission of the Future Vision "
        "XPRIZE competition in that it portrays a genuinely optimistic, "
        "technology-enabled future?",
        response_type="scale",
    ),
    Question(
        "Does this story have tech-forward storytelling, showing advanced "
        "technology meaningfully integrated into the narrative?",
        response_type="scale",
    ),
]

def parse_questions_file(path: str) -> List[Question]:
    """
    Load a custom question set from a file (see xprize-questions.md for the
    reference format), replacing DEFAULT_QUESTIONS. The file is a series of
    blocks separated by a line containing only '---', each block holding
    'key: value' lines:

        question: "Is the video less than or equal to 3 minutes and 15 seconds in length?"
        type: boolean
        weight_true: 0.0
        weight_false: -3.0
        ---
        question: "Does this story think big about humanity's future?"
        type: scale
        weight: 1.0

    `type` is "boolean" (the default if omitted) or "scale". Boolean
    questions read `weight_true` / `weight_false` (default 1.0 / -1.0 if
    omitted); scale questions read `weight`, the multiplier applied to
    Gemini's 1-10 rating (default 1.0, i.e. the rating itself is the
    contribution). A `range` field is accepted for documentation purposes
    but isn't used -- scale ratings are always 1-10, per the structured
    output schema.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    questions: List[Question] = []
    for raw_block in content.split("---"):
        block = raw_block.strip()
        if not block:
            continue

        fields = {}
        for line in block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            fields[key.strip().lower()] = value.strip()

        text = fields.get("question", "").strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1]
        if not text:
            continue

        response_type = fields.get("type", "boolean").strip().lower()
        if response_type == "scale":
            weight = float(fields.get("weight", 1.0))
            questions.append(
                Question(text=text, response_type="scale", weight=weight)
            )
        else:
            weight_if_true = float(fields.get("weight_true", 1.0))
            weight_if_false = float(fields.get("weight_false", -1.0))
            questions.append(
                Question(
                    text=text,
                    response_type="boolean",
                    weight_if_true=weight_if_true,
                    weight_if_false=weight_if_false,
                )
            )

    if not questions:
        raise ValueError(f"No questions found in {path}")

    return questions


# A model known to support video/multimodal input via the Gemini API.
# See https://ai.google.dev/gemini-api/docs/models for the current list.
#
# Rolled back to gemini-2.5-flash: the gemini-3.7-flash default tried
# afterward produced MORE collapsed 1-10 ratings on the scale questions
# (almost every film landing on the exact same value), not less, and two
# different prompt-engineering fixes (a prose rubric, then a forced
# band-classification field) failed to change that under gemini-3.7-flash.
# gemini-2.5-flash's original results had more natural spread on these
# questions, so this reverts both the model and the plain 1-10 prompt
# (no rubric, no band field) to that earlier known-working combination.
DEFAULT_MODEL = "gemini-2.5-flash"

# Lower temperature = less randomness in Gemini's sampling, which makes
# repeated runs on the same video much more consistent (though generative
# video understanding is never perfectly deterministic run-to-run, even at
# temperature 0, since the model's video sampling/frame selection can also
# vary slightly). Range is 0.0-2.0; 0.0 is the most deterministic setting.
DEFAULT_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class QuestionAnswer(BaseModel):
    """
    A single question and Gemini's structured answer about the video.

    Exactly one of `answer` / `rating` is populated, depending on whether
    the question was a boolean (yes/no) question or a scale (1-10) question:
    fill `answer` and leave `rating` null for a boolean question; fill
    `rating` and leave `answer` null for a scale question.
    """

    question: str = Field(description="The question that was asked.")
    answer: Optional[bool] = Field(
        default=None,
        description=(
            "For a yes/no question: True if the answer is yes, False if no. "
            "Leave this null for a 1-10 scale question."
        ),
    )
    rating: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description=(
            "For a 1-10 scale question: an integer from 1 (not at all) to "
            "10 (extremely strongly) rating how strongly the statement "
            "holds. Leave this null for a yes/no question."
        ),
    )
    explanation: str = Field(
        description=(
            "A brief (1-3 sentence) explanation of the evidence from the "
            "video that supports the answer, including timestamps (MM:SS) "
            "where relevant."
        )
    )
    confidence: str = Field(
        description="How confident the model is in this answer: 'high', 'medium', or 'low'."
    )


class VideoAnalysisResult(BaseModel):
    """The full structured result returned for a video."""

    video_url: str = Field(description="The YouTube URL that was analyzed.")
    video_title: str = Field(
        description=(
            "The actual title of the video, if it can be determined from an "
            "on-screen title card, spoken introduction, channel branding, or "
            "other evidence in the video itself. If no explicit title is "
            "identifiable, provide a concise (under 10 words) descriptive "
            "name for the video instead."
        )
    )
    questions: List[QuestionAnswer] = Field(
        description="The list of questions asked and their answers, in the order asked."
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def normalize_youtube_url(url: str) -> str:
    """
    Normalize any common YouTube URL shape into the canonical
    https://www.youtube.com/watch?v=VIDEO_ID form (no extra query params
    like the ?si= sharing token), which is the form Gemini's video
    understanding API reliably matches. Short links (youtu.be/ID),
    Shorts links (/shorts/ID), and mobile (m.youtube.com) links are all
    handled; anything else is returned unchanged.
    """
    url = url.strip()
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    video_id: Optional[str] = None

    if "youtu.be" in host:
        # https://youtu.be/VIDEO_ID?si=...
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif "youtube.com" in host:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        else:
            # /shorts/VIDEO_ID, /embed/VIDEO_ID, /live/VIDEO_ID, etc.
            match = re.match(r"^/(?:shorts|embed|live)/([^/?]+)", parsed.path)
            if match:
                video_id = match.group(1)

    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"

    # Fall back to the original URL if we didn't recognize the shape.
    return url


def build_prompt(questions: Optional[List[Question]] = None) -> str:
    questions = questions if questions is not None else DEFAULT_QUESTIONS
    lines = []
    for i, q in enumerate(questions):
        if q.response_type == "scale":
            hint = (
                "Answer with an integer rating from 1 (not at all) to 10 "
                "(extremely strongly); set 'rating', leave 'answer' null."
            )
        else:
            hint = "Answer yes or no; set 'answer' to true/false, leave 'rating' null."
        lines.append(f"{i + 1}. {q.text} [{hint}]")
    numbered = "\n".join(lines)

    return (
        "You are analyzing the attached video. Watch it carefully, including "
        "its audio, visuals, and any on-screen text, and then answer each of "
        "the following questions.\n\n"
        f"{numbered}\n\n"
        "Follow the answer-format instructions in brackets for each "
        "question exactly. Also give a brief explanation citing specific "
        "evidence from the video (use MM:SS timestamps where it helps). "
        "Answer every question, in the same order they were given, and be "
        "precise about video length and timing when a question involves "
        "duration."
    )


def analyze_video(
    youtube_url: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    questions: Optional[List[Question]] = None,
    agentic: bool = False,
    show_prompt: bool = False,
) -> VideoAnalysisResult:
    """
    Send a YouTube URL and a set of questions (DEFAULT_QUESTIONS unless
    `questions` is given) to Gemini and return a structured
    (Pydantic-validated) result.

    `show_prompt=True` prints the full prompt sent to Gemini (video URL,
    model, and the complete numbered question text) to stderr before the
    call. Off by default.

    `agentic=True` turns on Gemini's agentic video processing (announced
    September 2026, supported on gemini-3.5-flash-lite, gemini-3.6-flash,
    and gemini-3.7-flash): instead of ingesting the whole video at a fixed
    frame rate, the model dynamically chooses which segments, modalities,
    and frame rates to actually look at. Google reports this can cut
    tokens/cost substantially and improve accuracy, especially on longer
    videos -- but it changes what the model actually "sees", so treat a
    switch to it like a model change and re-validate scores before trusting
    it (this matters here: this pipeline's default model, gemini-2.5-flash,
    predates agentic processing, and gemini-3.7-flash previously produced
    worse-collapsed 1-10 ratings -- see DEFAULT_MODEL's comment above).
    """
    normalized_url = normalize_youtube_url(youtube_url)
    if normalized_url != youtube_url:
        print(f"Normalized URL: {youtube_url} -> {normalized_url}", file=sys.stderr)
    youtube_url = normalized_url

    # genai.Client() will automatically pick up the GEMINI_API_KEY
    # environment variable if api_key is not passed explicitly.
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    prompt = build_prompt(questions)

    if show_prompt:
        print("=" * 70, file=sys.stderr)
        print(f"Prompt sent to Gemini ({model}) for {youtube_url}:", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(prompt, file=sys.stderr)
        print("=" * 70, file=sys.stderr)

    if agentic:
        media_processing_enum = getattr(types, "MediaProcessing", None)
        if media_processing_enum is None:
            raise RuntimeError(
                "--agentic requires a newer google-genai SDK than the one "
                "installed: this version's google.genai.types has no "
                "MediaProcessing type. Run 'pip install --upgrade "
                "google-genai' (or 'pip install --upgrade google-genai "
                "--break-system-packages' if that's how it was installed), "
                "then try again."
            )
        # The API rejects media_processing unless mime_type is also set on
        # the FileData, even for a YouTube URL (where the actual container
        # format is unknown/irrelevant) -- "video/mp4" is accepted as a
        # generic placeholder here.
        video_part = types.Part(
            file_data=types.FileData(file_uri=youtube_url, mime_type="video/mp4")
        )
        video_part.media_processing = media_processing_enum.AGENTIC
    else:
        video_part = types.Part(file_data=types.FileData(file_uri=youtube_url))

    contents = types.Content(
        parts=[
            video_part,
            types.Part(text=prompt),
        ]
    )

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=VideoAnalysisResult,
        temperature=temperature,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    # The SDK also exposes response.parsed (a VideoAnalysisResult instance)
    # when response_schema is a Pydantic model, but we validate explicitly
    # here so this works even if that convenience field is unavailable.
    return VideoAnalysisResult.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoredQuestion:
    """One question's answer plus the weight/contribution used to score it."""

    question: str
    response_type: str
    answer: Optional[bool]
    rating: Optional[int]
    explanation: str
    confidence: str
    weight_if_true: float
    weight_if_false: float
    weight: float

    @property
    def contribution(self) -> float:
        if self.response_type == "scale":
            return (self.rating or 0) * self.weight
        return self.weight_if_true if self.answer else self.weight_if_false

    @property
    def display_answer(self) -> str:
        if self.response_type == "scale":
            return f"{self.rating}/10" if self.rating is not None else "n/a"
        return "Yes" if self.answer else "No"


@dataclass
class ScoreResult:
    """The overall score for a video plus the per-question breakdown."""

    total_score: float
    items: List[ScoredQuestion] = field(default_factory=list)


def score_video(
    result: VideoAnalysisResult, questions: Optional[List[Question]] = None
) -> ScoreResult:
    """
    Score a VideoAnalysisResult using the settings defined on `questions`
    (DEFAULT_QUESTIONS unless a custom question set is given -- this should
    be the same set that was passed to analyze_video() for this result):

    - Boolean questions: weight_if_true is added if Gemini answered True,
      weight_if_false (typically negative) is added if it answered False.
    - Scale questions: the 1-10 rating Gemini gave, multiplied by `weight`,
      is added directly.

    Answers are matched back to `questions` by question text; if a
    returned question can't be matched (e.g. the model reworded it), it's
    treated as a boolean question with the standard +1 / -1 weights.
    """
    questions = questions if questions is not None else DEFAULT_QUESTIONS
    questions_by_text = {q.text: q for q in questions}

    items: List[ScoredQuestion] = []
    total = 0.0

    for qa in result.questions:
        question_def = questions_by_text.get(qa.question)
        if question_def is not None:
            response_type = question_def.response_type
            weight_if_true = question_def.weight_if_true
            weight_if_false = question_def.weight_if_false
            weight = question_def.weight
        else:
            response_type = "boolean"
            weight_if_true = 1.0
            weight_if_false = -1.0
            weight = 1.0

        scored = ScoredQuestion(
            question=qa.question,
            response_type=response_type,
            answer=qa.answer,
            rating=qa.rating,
            explanation=qa.explanation,
            confidence=qa.confidence,
            weight_if_true=weight_if_true,
            weight_if_false=weight_if_false,
            weight=weight,
        )
        items.append(scored)
        total += scored.contribution

    return ScoreResult(total_score=total, items=items)


def _slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return slug[:max_length].rstrip("-") or "video"


def write_markdown_report(
    result: VideoAnalysisResult,
    score: ScoreResult,
    output_path: str,
) -> None:
    """
    Write a Markdown report containing the video's name, its overall score,
    and a Question / Answer / Explanation / Confidence breakdown for every
    question asked.
    """
    lines: List[str] = []
    title = result.video_title or result.video_url

    lines.append(f"# Video Analysis: {title}")
    lines.append("")
    lines.append(f"**Video URL:** {result.video_url}")
    lines.append("")
    lines.append(f"**Score:** {score.total_score:+g}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| # | Question | Answer | Weight | Confidence | Explanation |")
    lines.append("|---|----------|--------|--------|------------|-------------|")

    for i, item in enumerate(score.items, start=1):
        weight_text = f"{item.contribution:+g}"
        # Escape pipe characters so the table doesn't break.
        question = item.question.replace("|", "\\|")
        explanation = item.explanation.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | {question} | {item.display_answer} | {weight_text} | "
            f"{item.confidence} | {explanation} |"
        )

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Batch mode: analyze every video listed in a file (-file)
# ---------------------------------------------------------------------------

@dataclass
class BatchEntry:
    rank: int
    title: Optional[str]  # known only if the file gave one (e.g. a table row)
    url: str


# Matches a Markdown table row like:
# | 12 | After Eternity | Gabriel Garcia | https://www.youtube.com/watch?v=CY-8nPEaBXs |
TABLE_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(https?://\S+?)\s*\|\s*$"
)

# Matches any bare YouTube URL appearing anywhere in a line (a plain list of
# URLs, one per line, or a URL embedded in other text/markup).
YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com|youtu\.be)/\S+"
)


def parse_batch_file(path: str) -> List[BatchEntry]:
    """
    Extract every video to analyze from `path`, in order of first
    appearance, de-duplicated by URL. Two line shapes are recognized:

    - A Markdown table row like '| # | Film | Creator | YouTube URL |'
      (e.g. top50.md's format) -- the rank and title are taken from the
      row.
    - Any other line containing a bare YouTube URL (a plain list of URLs,
      one per line, is the common case) -- the title is unknown (None) and
      the rank is assigned sequentially in file order.

    Blank lines, headers, separators, and anything without a recognizable
    YouTube URL are skipped.
    """
    entries: List[BatchEntry] = []
    seen_urls = set()

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            table_match = TABLE_ROW_RE.match(line)
            if table_match:
                rank_str, title, _creator, url = table_match.groups()
                rank: Optional[int] = int(rank_str)
            else:
                url_match = YOUTUBE_URL_RE.search(line)
                if not url_match:
                    continue
                url = url_match.group(0).rstrip(").,]>\"'")
                title = None
                rank = None

            if url in seen_urls:
                continue
            seen_urls.add(url)
            entries.append(BatchEntry(rank=rank, title=title, url=url))

    # Fill in sequential ranks for entries that didn't come from a numbered
    # table row, preserving file order.
    for i, entry in enumerate(entries, start=1):
        if entry.rank is None:
            entry.rank = i

    return entries


def write_batch_csv(csv_path: str, rows: List[dict]) -> None:
    """(Re)write the summary CSV from scratch with the rows gathered so far."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Title", "Score", "YouTube URL", "Status"])
        for row in rows:
            writer.writerow(
                [row["rank"], row["title"], row["score"], row["url"], row["status"]]
            )


def run_batch(
    input_path: str,
    output_dir: str,
    csv_path: Optional[str],
    model: str,
    temperature: float,
    api_key: Optional[str],
    limit: Optional[int],
    delay: float,
    force: bool,
    questions: Optional[List[Question]] = None,
    agentic: bool = False,
    show_prompt: bool = False,
) -> int:
    """
    Analyze every video listed in `input_path`, writing each video's JSON
    result and Markdown report into `output_dir`, plus a summary CSV.
    Already-analyzed videos (an existing JSON result in output_dir) are
    skipped unless `force` is set, and a single video's failure doesn't
    stop the rest of the batch. `questions` (DEFAULT_QUESTIONS if omitted)
    is used for both asking Gemini and scoring the result. `agentic`
    enables Gemini's agentic video processing mode (see analyze_video()).
    `show_prompt` prints the full prompt sent to Gemini for every video.
    """
    entries = parse_batch_file(input_path)
    if limit is not None:
        entries = entries[:limit]

    if not entries:
        print(f"No video entries found in {input_path}", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)
    csv_path = csv_path or os.path.join(output_dir, "scores.csv")

    print(f"Found {len(entries)} video(s) in {input_path}", file=sys.stderr)

    rows: List[dict] = []

    for i, entry in enumerate(entries):
        label = entry.title or entry.url
        base_name = (
            f"{entry.rank:02d}-{_slugify(entry.title)}" if entry.title else f"{entry.rank:02d}"
        )
        json_path = os.path.join(output_dir, f"{base_name}.json")
        md_path = os.path.join(output_dir, f"{base_name}.md")

        if os.path.exists(json_path) and not force:
            print(
                f"[{entry.rank}/{len(entries)}] Skipping (already analyzed): {label}",
                file=sys.stderr,
            )
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    cached_result = VideoAnalysisResult.model_validate_json(f.read())
                cached_score = score_video(cached_result, questions)
                rows.append(
                    {
                        "rank": entry.rank,
                        "title": entry.title or cached_result.video_title,
                        "score": f"{cached_score.total_score:g}",
                        "url": entry.url,
                        "status": "cached",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "rank": entry.rank,
                        "title": entry.title or entry.url,
                        "score": "",
                        "url": entry.url,
                        "status": f"error reading cached result: {exc}",
                    }
                )
            write_batch_csv(csv_path, rows)
            continue

        print(
            f"[{entry.rank}/{len(entries)}] Analyzing: {label}",
            file=sys.stderr,
        )

        try:
            result = analyze_video(
                youtube_url=entry.url,
                model=model,
                api_key=api_key,
                temperature=temperature,
                questions=questions,
                agentic=agentic,
                show_prompt=show_prompt,
            )
            score = score_video(result, questions)

            with open(json_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            write_markdown_report(result, score, md_path)

            print(f"    Score: {score.total_score:+g}  ->  {json_path}", file=sys.stderr)
            rows.append(
                {
                    "rank": entry.rank,
                    "title": entry.title or result.video_title,
                    "score": f"{score.total_score:g}",
                    "url": entry.url,
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep going on a per-video failure
            print(f"    Error analyzing {label}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "rank": entry.rank,
                    "title": entry.title or entry.url,
                    "score": "",
                    "url": entry.url,
                    "status": f"error: {exc}",
                }
            )

        # Rewrite the CSV after every video so progress survives an
        # interruption partway through the batch.
        write_batch_csv(csv_path, rows)

        if delay and i < len(entries) - 1:
            time.sleep(delay)

    ok_count = sum(1 for r in rows if r["status"] in ("ok", "cached"))
    print(
        f"\nDone. {ok_count}/{len(entries)} video(s) scored. Summary CSV: {csv_path}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected 'true' or 'false', got {value!r}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Answer DEFAULT_QUESTIONS about a YouTube video using Gemini's "
            "video understanding API, then score and report the result. "
            "Pass a single YouTube URL for one video, -file a text/"
            "Markdown file to batch-analyze every video URL it lists, or "
            "-dir a directory containing questions.md and dataset.md."
        )
    )
    parser.add_argument(
        "youtube_url",
        nargs="?",
        default=None,
        help="The YouTube video URL to analyze. Omit this when using -file.",
    )
    parser.add_argument(
        "-file",
        "--file",
        "-input",
        "--input",
        dest="input_file",
        default=None,
        help=(
            "Path to a file listing multiple videos: any YouTube URL found "
            "in the file is picked up, whether it's a plain list of URLs "
            "(one per line) or a Markdown table with rows shaped like "
            "'| # | Film | Creator | YouTube URL |'. When given, every "
            "video found is analyzed in batch instead of the youtube_url "
            "argument, and results are written into a new folder named "
            "after this file (e.g. -file muyot-dataset.md -> "
            "./muyot-dataset/), unless --output-dir overrides that."
        ),
    )
    parser.add_argument(
        "-questions",
        "--questions",
        dest="questions_file",
        default=None,
        help=(
            "Path to a custom question set file (see xprize-questions.md "
            "for the format), used instead of the hard-coded "
            "DEFAULT_QUESTIONS. Applies to single-video, -file, and -dir "
            "mode; overrides -dir's questions.md if both are given."
        ),
    )
    parser.add_argument(
        "-dir",
        "--dir",
        dest="project_dir",
        default=None,
        help=(
            "Path to a directory containing questions.md (a custom "
            "question set, see xprize-questions.md for the format) and "
            "dataset.md (the videos to analyze, see -file for the "
            "supported formats). Both are loaded automatically and every "
            "video in dataset.md is analyzed in batch, with results "
            "written into this same directory. Mutually exclusive with "
            "youtube_url and -file."
        ),
    )
    parser.add_argument(
        "--showprompt",
        type=_str2bool,
        default=False,
        metavar="{true,false}",
        help=(
            "Print the full prompt sent to Gemini (video URL, model, and "
            "the complete numbered question text) to stderr before each "
            "call. Default: false."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Gemini model to use (default: {DEFAULT_MODEL}). Only "
            "gemini-3.5-flash-lite, gemini-3.6-flash, and gemini-3.7-flash "
            "support --agentic."
        ),
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        help=(
            "Enable Gemini's agentic video processing (Sept 2026): instead "
            "of ingesting the whole video at a fixed frame rate, the model "
            "dynamically picks which segments/modalities/frame rates to "
            "look at. Can cut tokens and cost substantially and may "
            "improve accuracy, especially on longer videos -- but changes "
            "what the model sees, so treat it like a model change and "
            "re-validate scores on a small batch first. Requires --model "
            "gemini-3.5-flash-lite, gemini-3.6-flash, or gemini-3.7-flash."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "Sampling temperature (0.0-2.0). Lower is more deterministic; "
            f"default is {DEFAULT_TEMPERATURE} for maximally consistent "
            "answers across repeated runs."
        ),
    )
    parser.add_argument(
        "--output",
        help="Single-video mode only: optional path to write the JSON result to. Defaults to stdout only.",
    )
    parser.add_argument(
        "--api-key",
        help=(
            "Gemini API key. Defaults to the GEMINI_API_KEY environment "
            "variable if not provided."
        ),
    )
    parser.add_argument(
        "--report",
        dest="report_path",
        help=(
            "Single-video mode only: path to write the Markdown score "
            "report to. Defaults to a filename derived from the video's "
            "title."
        ),
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Single-video mode only: skip writing the Markdown score report.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Batch mode only: directory to write each video's JSON result "
            "and Markdown report into. Defaults to a new folder named after "
            "the -file argument (its filename without the extension), or "
            "to the -dir argument itself when using -dir."
        ),
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Batch mode only: path to the summary CSV (default: <output-dir>/scores.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Batch mode only: only process the first N entries from -file.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Batch mode only: seconds to pause between videos, to be gentle on API rate limits (default: 2.0).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Batch mode only: re-analyze videos that already have a saved JSON result, instead of skipping them.",
    )

    args = parser.parse_args(argv)

    modes_given = sum(
        1 for v in (args.youtube_url, args.input_file, args.project_dir) if v
    )
    if modes_given == 0:
        parser.error(
            "provide a youtube_url, -file <file.md>, or -dir <directory>."
        )
    if modes_given > 1:
        parser.error(
            "provide only one of: youtube_url, -file <file.md>, or "
            "-dir <directory>."
        )

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "Warning: no API key provided via --api-key or GEMINI_API_KEY; "
            "the client will fail unless credentials are configured another way.",
            file=sys.stderr,
        )

    # -dir <directory> is a convenience wrapper: it supplies both the
    # dataset (<directory>/dataset.md) and the question set
    # (<directory>/questions.md), and defaults the output folder to that
    # same directory. An explicit -questions still overrides the
    # directory's questions.md.
    input_path = args.input_file
    questions_file = args.questions_file
    output_dir = args.output_dir
    if args.project_dir:
        input_path = os.path.join(args.project_dir, "dataset.md")
        if not questions_file:
            questions_file = os.path.join(args.project_dir, "questions.md")
        if not output_dir:
            output_dir = args.project_dir

    questions: Optional[List[Question]] = None
    if questions_file:
        try:
            questions = parse_questions_file(questions_file)
        except Exception as exc:  # noqa: BLE001
            print(f"Error reading questions file {questions_file}: {exc}", file=sys.stderr)
            return 1
        print(
            f"Loaded {len(questions)} question(s) from {questions_file}",
            file=sys.stderr,
        )

    if input_path:
        # Default the output folder to the input file's own name, e.g.
        # -file muyot-dataset.md -> ./muyot-dataset/ (this only applies to
        # -file; -dir already set output_dir to the directory itself above).
        if not output_dir:
            base = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = base or "batch-results"

        return run_batch(
            input_path=input_path,
            output_dir=output_dir,
            csv_path=args.csv_path,
            model=args.model,
            temperature=args.temperature,
            api_key=args.api_key,
            limit=args.limit,
            delay=args.delay,
            force=args.force,
            questions=questions,
            agentic=args.agentic,
            show_prompt=args.showprompt,
        )

    try:
        result = analyze_video(
            youtube_url=args.youtube_url,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            questions=questions,
            agentic=args.agentic,
            show_prompt=args.showprompt,
        )
    except Exception as exc:  # noqa: BLE001 - surface any API/validation error clearly
        print(f"Error analyzing video: {exc}", file=sys.stderr)
        return 1

    output_json = result.model_dump_json(indent=2)
    print(output_json)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\nWrote result to {args.output}", file=sys.stderr)

    score = score_video(result, questions)
    print(f"\nScore: {score.total_score:+g}", file=sys.stderr)

    if not args.no_report:
        report_path = args.report_path or f"{_slugify(result.video_title)}.md"
        write_markdown_report(result, score, report_path)
        print(f"Wrote report to {report_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
