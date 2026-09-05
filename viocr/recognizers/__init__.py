"""Recognizer registry."""

from __future__ import annotations

from typing import Dict

from .base import Recognizer
from .paddle_rec import PaddleOCRRecognizer
from .trocr_rec import TrOCRRecognizer
from .vietocr_rec import VietOCRRecognizer

BUILDERS = {
    "VietOCR": VietOCRRecognizer,
    "PaddleOCR": PaddleOCRRecognizer,
    "TrOCR": TrOCRRecognizer,
}

__all__ = [
    "Recognizer",
    "VietOCRRecognizer",
    "PaddleOCRRecognizer",
    "TrOCRRecognizer",
    "BUILDERS",
    "build",
]


def build(name: str, cfg, **kwargs) -> Recognizer:
    if name not in BUILDERS:
        raise KeyError(f"unknown recognizer {name!r}; available: {sorted(BUILDERS)}")
    return BUILDERS[name](cfg, **kwargs)
