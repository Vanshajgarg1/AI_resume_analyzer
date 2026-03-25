"""
utils.py
--------
Utility / helper functions used across the project.
Includes: text cleaning, token estimation, result formatting, keyword highlighting.
"""

import re
import logging
import textwrap
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_resume_text(text: str) -> str:
    """
    Clean raw extracted text from a resume.

    - Normalises unicode dashes and bullets
    - Strips whitespace on each line
    - Collapses excessive blank lines
    """
    if not text:
        return ""

    text = text.replace("\u2022", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u00a0", " ")

    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []
    blank_streak = 0
    for line in lines:
        if line == "":
            blank_streak += 1
            if blank_streak <= 2:
                cleaned_lines.append(line)
        else:
            blank_streak = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def truncate_text(text: str, max_chars: int = 12000) -> str:
    """Truncate text to max_chars, preserving whole words."""
    if len(text) <= max_chars:
        return text
    logger.warning(f"Truncating resume from {len(text)} → {max_chars} chars.")
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.9:
        truncated = truncated[:last_space]
    return truncated + "\n\n[... resume truncated for length ...]"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_resume_text(text: str, min_chars: int = 100) -> Optional[str]:
    """
    Returns an error message string if text is invalid, else None.
    """
    if not text or not text.strip():
        return "Resume appears to be empty. Please upload a valid file."
    if len(text.strip()) < min_chars:
        return f"Resume text is too short ({len(text.strip())} chars). Upload a more complete resume."
    if len(text.split()) < 20:
        return "Resume has very few words. It may be image-based. Please upload a text-selectable file."
    return None


def validate_job_role(job_role: str) -> Optional[str]:
    """Returns error string if invalid, else None."""
    if not job_role or not job_role.strip():
        return "Please enter a target job role (e.g., 'Data Scientist')."
    if len(job_role.strip()) < 3:
        return "Job role is too short. Please be more specific."
    if len(job_role.strip()) > 100:
        return "Job role is too long. Keep it under 100 characters."
    return None


# ---------------------------------------------------------------------------
# Formatting & Scoring helpers
# ---------------------------------------------------------------------------

def format_ats_score(score: int) -> str:
    """Return human-readable ATS score label."""
    if score >= 80:
        return f"{score}/100 — Excellent ✅"
    elif score >= 60:
        return f"{score}/100 — Good 🟡"
    elif score >= 40:
        return f"{score}/100 — Fair 🟠"
    return f"{score}/100 — Needs Work 🔴"


def get_ats_color(score: int) -> str:
    """Return hex colour string for an ATS score."""
    if score >= 80:
        return "#22c55e"
    elif score >= 60:
        return "#eab308"
    elif score >= 40:
        return "#f97316"
    return "#ef4444"


def get_ats_label(score: int) -> str:
    """Return short text label for a score."""
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Fair"
    return "Needs Work"


def compute_skill_match_percent(detected: List[str], required: List[str]) -> float:
    """
    What percentage of required skills are present in detected skills?

    Returns:
        float: 0.0 – 100.0
    """
    if not required:
        return 0.0
    detected_lower = {s.lower() for s in detected}
    matched = sum(1 for s in required if s.lower() in detected_lower)
    return round(matched / len(required) * 100, 1)


# ---------------------------------------------------------------------------
# Keyword Highlighting
# ---------------------------------------------------------------------------

def highlight_keywords_in_text(text: str, keywords: List[str], tag: str = "mark") -> str:
    """
    Wrap every occurrence of each keyword in the given HTML tag.

    Args:
        text (str): The resume text to highlight.
        keywords (list): Keywords to highlight (case-insensitive).
        tag (str): HTML tag to wrap matches in. Default 'mark'.

    Returns:
        str: Text with matched keywords wrapped in <tag>...</tag>.
    """
    if not text or not keywords:
        return text

    # Sort longest first to avoid partial-match conflicts
    sorted_kws = sorted(keywords, key=len, reverse=True)

    for kw in sorted_kws:
        pattern = re.compile(r"(" + re.escape(kw) + r")", re.IGNORECASE)
        text = pattern.sub(rf"<{tag}>\1</{tag}>", text)

    return text


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def section_divider(title: str, width: int = 60) -> str:
    return f"\n{'─' * width}\n  {title.upper()}\n{'─' * width}\n"


def wrap_text(text: str, width: int = 80) -> str:
    return textwrap.fill(text, width=width)


def estimate_token_count(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def safe_int(value, default: int = 0) -> int:
    """Safely cast value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
