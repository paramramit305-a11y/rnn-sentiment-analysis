from __future__ import annotations

import html
import re


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<.*?>")
NON_TEXT_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize free-form review text for training and inference."""
    if text is None:
        return ""

    normalized = html.unescape(str(text)).lower()
    normalized = URL_RE.sub(" ", normalized)
    normalized = HTML_TAG_RE.sub(" ", normalized)
    normalized = NON_TEXT_RE.sub(" ", normalized)
    normalized = MULTISPACE_RE.sub(" ", normalized).strip()
    return normalized
