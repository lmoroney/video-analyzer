#!/usr/bin/env python3
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
"""

from __future__ import annotations

import argparse
import os
import re
import sys
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
        weight_if_true=5.0,
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

# A model known to support video/multimodal input via the Gemini API.
# See https://ai.google.dev/gemini-api/docs/models for the current list.
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


def build_prompt() -> str:
    lines = []
    for i, q in enumerate(DEFAULT_QUESTIONS):
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
) -> VideoAnalysisResult:
    """
    Send a YouTube URL and DEFAULT_QUESTIONS to Gemini and return a
    structured (Pydantic-validated) result.
    """
    normalized_url = normalize_youtube_url(youtube_url)
    if normalized_url != youtube_url:
        print(f"Normalized URL: {youtube_url} -> {normalized_url}", file=sys.stderr)
    youtube_url = normalized_url

    # genai.Client() will automatically pick up the GEMINI_API_KEY
    # environment variable if api_key is not passed explicitly.
    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    prompt = build_prompt()

    contents = types.Content(
        parts=[
            types.Part(file_data=types.FileData(file_uri=youtube_url)),
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


def score_video(result: VideoAnalysisResult) -> ScoreResult:
    """
    Score a VideoAnalysisResult using the settings defined on
    DEFAULT_QUESTIONS:

    - Boolean questions: weight_if_true is added if Gemini answered True,
      weight_if_false (typically negative) is added if it answered False.
    - Scale questions: the 1-10 rating Gemini gave, multiplied by `weight`,
      is added directly.

    Answers are matched back to DEFAULT_QUESTIONS by question text; if a
    returned question can't be matched (e.g. the model reworded it), it's
    treated as a boolean question with the standard +1 / -1 weights.
    """
    questions_by_text = {q.text: q for q in DEFAULT_QUESTIONS}

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
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Answer DEFAULT_QUESTIONS about a YouTube video using Gemini's "
            "video understanding API, then score and report the result."
        )
    )
    parser.add_argument("youtube_url", help="The YouTube video URL to analyze.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL}).",
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
        help="Optional path to write the JSON result to. Defaults to stdout only.",
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
            "Path to write the Markdown score report to. Defaults to a "
            "filename derived from the video's title."
        ),
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the Markdown score report.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "Warning: no API key provided via --api-key or GEMINI_API_KEY; "
            "the client will fail unless credentials are configured another way.",
            file=sys.stderr,
        )

    try:
        result = analyze_video(
            youtube_url=args.youtube_url,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
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

    score = score_video(result)
    print(f"\nScore: {score.total_score:+g}", file=sys.stderr)

    if not args.no_report:
        report_path = args.report_path or f"{_slugify(result.video_title)}.md"
        write_markdown_report(result, score, report_path)
        print(f"Wrote report to {report_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
