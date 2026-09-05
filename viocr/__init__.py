"""Vietnamese OCR ensemble: PaddleOCR + VietOCR + TrOCR."""

from .config import Config
from .data import read_labels
from .metrics import classify, error_distribution, metric_text, score
from .voting import build_lexicon, combine, vote_char, vote_line, vote_lexicon

__version__ = "1.0.0"

__all__ = [
    "Config", "read_labels", "score", "metric_text", "classify",
    "error_distribution", "combine", "vote_line", "vote_char",
    "vote_lexicon", "build_lexicon",
]
