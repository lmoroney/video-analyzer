#!/usr/bin/env python3
"""
batch_top50.py

Runs analyze_video.py's Gemini video analysis + scoring over every film
listed in top50.md (the '| # | Film | Creator | YouTube |' table), saving
each film's full JSON result and Markdown report into top50-results/, plus
a single summary CSV listing every film's name and total score.

Requires analyze_video.py to be in the same directory (it's imported for
the actual Gemini call, scoring, and report writing).

Setup
-----
    pip install google-genai pydantic
    export GEMINI_API_KEY="your-api-key-here"

Usage
-----
    # Analyze every film in top50.md, writing into top50-results/
    python batch_top50.py

    # Point at a different input file / output directory
    python batch_top50.py --input top50.md --output-dir top50-results

    # Just try the first 3 films (useful for a quick test run)
    python batch_top50.py --limit 3

    # Re-run films that already have a saved result (by default, films
    # with an existing JSON result in --output-dir are skipped and their
    # cached score is reused, so an interrupted batch can be resumed by
    # just running the same command again)
    python batch_top50.py --force

    # Slow down / speed up the pause between videos (default 2 seconds,
    # to be gentle on API rate limits over 50 requests)
    python batch_top50.py --delay 5
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

from analyze_video import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    VideoAnalysisResult,
    _slugify,
    analyze_video,
    score_video,
    write_markdown_report,
)


@dataclass
class Entry:
    rank: int
    title: str
    url: str


# Matches a Markdown table row like:
# | 12 | After Eternity | Gabriel Garcia | https://www.youtube.com/watch?v=CY-8nPEaBXs |
TABLE_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(https?://\S+?)\s*\|\s*$"
)


def parse_top50(path: str) -> List[Entry]:
    """
    Parse the '| # | Film | Creator | YouTube |' table rows out of a
    top50.md-style file, line by line. Any line that isn't a numbered
    table row with a trailing YouTube URL (headers, separators, the plain
    URL list at the bottom, blank lines) is skipped.
    """
    entries: List[Entry] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = TABLE_ROW_RE.match(line.strip())
            if not match:
                continue
            rank_str, title, _creator, url = match.groups()
            entries.append(Entry(rank=int(rank_str), title=title, url=url))
    return entries


def write_csv(csv_path: str, rows: List[dict]) -> None:
    """(Re)write the summary CSV from scratch with the rows gathered so far."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Title", "Score", "YouTube URL", "Status"])
        for row in rows:
            writer.writerow(
                [row["rank"], row["title"], row["score"], row["url"], row["status"]]
            )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-analyze every film listed in top50.md with Gemini's video understanding API."
    )
    parser.add_argument(
        "--input",
        default="top50.md",
        help="Path to the top50.md file (default: top50.md).",
    )
    parser.add_argument(
        "--output-dir",
        default="top50-results",
        help="Directory to write each film's JSON result and Markdown report into (default: top50-results).",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Path to the summary CSV (default: <output-dir>/scores.csv).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--api-key",
        help="Gemini API key. Defaults to the GEMINI_API_KEY environment variable if not provided.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N entries in top50.md (useful for a quick test run).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to pause between videos, to be gentle on API rate limits (default: 2.0).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyze films that already have a saved JSON result, instead of skipping them.",
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

    entries = parse_top50(args.input)
    if args.limit is not None:
        entries = entries[: args.limit]

    if not entries:
        print(f"No film entries found in {args.input}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = args.csv_path or os.path.join(args.output_dir, "scores.csv")

    print(f"Found {len(entries)} film(s) in {args.input}", file=sys.stderr)

    rows: List[dict] = []

    for i, entry in enumerate(entries):
        base_name = f"{entry.rank:02d}-{_slugify(entry.title)}"
        json_path = os.path.join(args.output_dir, f"{base_name}.json")
        md_path = os.path.join(args.output_dir, f"{base_name}.md")

        if os.path.exists(json_path) and not args.force:
            print(
                f"[{entry.rank}/{len(entries)}] Skipping (already analyzed): {entry.title}",
                file=sys.stderr,
            )
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    cached_result = VideoAnalysisResult.model_validate_json(f.read())
                cached_score = score_video(cached_result)
                rows.append(
                    {
                        "rank": entry.rank,
                        "title": entry.title,
                        "score": f"{cached_score.total_score:g}",
                        "url": entry.url,
                        "status": "cached",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "rank": entry.rank,
                        "title": entry.title,
                        "score": "",
                        "url": entry.url,
                        "status": f"error reading cached result: {exc}",
                    }
                )
            write_csv(csv_path, rows)
            continue

        print(
            f"[{entry.rank}/{len(entries)}] Analyzing: {entry.title} ({entry.url})",
            file=sys.stderr,
        )

        try:
            result = analyze_video(
                youtube_url=entry.url,
                model=args.model,
                api_key=args.api_key,
                temperature=args.temperature,
            )
            score = score_video(result)

            with open(json_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            write_markdown_report(result, score, md_path)

            print(f"    Score: {score.total_score:+g}  ->  {json_path}", file=sys.stderr)
            rows.append(
                {
                    "rank": entry.rank,
                    "title": entry.title,
                    "score": f"{score.total_score:g}",
                    "url": entry.url,
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep going on a per-video failure
            print(f"    Error analyzing {entry.title}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "rank": entry.rank,
                    "title": entry.title,
                    "score": "",
                    "url": entry.url,
                    "status": f"error: {exc}",
                }
            )

        # Rewrite the CSV after every video so progress survives an
        # interruption partway through the batch.
        write_csv(csv_path, rows)

        if args.delay and i < len(entries) - 1:
            time.sleep(args.delay)

    ok_count = sum(1 for r in rows if r["status"] in ("ok", "cached"))
    print(
        f"\nDone. {ok_count}/{len(entries)} film(s) scored. Summary CSV: {csv_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
