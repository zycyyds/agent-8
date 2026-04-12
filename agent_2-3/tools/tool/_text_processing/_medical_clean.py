# -*- coding: utf-8 -*-
"""Medical text cleaning tool functions."""
import re
import unicodedata
from typing import Iterable

from agentscope.message import TextBlock
from agentscope.tool._response import ToolResponse


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _standardize_dashes(text: str) -> str:
    return (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2212", "-")
        .replace("\u00ad", "")
    )


def _allowed_symbols() -> set[str]:
    return set(
        [
            "%",
            "+",
            "-",
            "/",
            "\\",
            ".",
            ",",
            ":",
            ";",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "*",
            "×",
            "°",
            "μ",
            "±",
            "<",
            ">",
            "≤",
            "≥",
        ]
    )


def _is_allowed_char(ch: str, allow: set[str]) -> bool:
    if ch.isalnum():
        return True
    if ch in allow:
        return True
    if ch in {" ", "\t", "\n"}:
        return True
    return False


def _remove_disallowed(text: str, allow: set[str]) -> str:
    return "".join(_ for _ in text if _is_allowed_char(_, allow))


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _pipeline(steps: Iterable) -> callable:
    def _run(text: str) -> str:
        for fn in steps:
            text = fn(text)
        return text

    return _run


def clean_medical_text(text: str, preserve_symbols: bool = True) -> ToolResponse:
    """Clean medical text by removing irrelevant characters, ensuring UTF-8
    normalization, and standardizing spaces and newlines.

    Args:
        text (str):
            The raw input text to be cleaned.
        preserve_symbols (bool):
            Whether to preserve common medical symbols such as μ, ±, ≤, ≥,
            °, ×, %, and unit-related punctuation.

    Returns:
        ToolResponse:
            A tool response containing a single text block with the cleaned
            content.
    """
    allow = _allowed_symbols() if preserve_symbols else set()
    pipeline = _pipeline(
        [
            _normalize_unicode,
            _standardize_dashes,
            lambda s: _remove_disallowed(s, allow),
            _normalize_whitespace,
        ]
    )
    cleaned = pipeline(text)
    return ToolResponse(content=[TextBlock(type="text", text=cleaned)])

