"""Common interface for the three recognizers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence


class Recognizer:
    """A recognizer maps image files to normalised text.

    Implementations load their weights lazily, on the first call to
    :meth:`recognize`, so that constructing the object stays cheap and a model
    whose checkpoint is missing does not prevent the others from running.
    """

    name: str = "base"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._loaded = False

    def load(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    def recognize(self, paths: Sequence[Path]) -> List[str]:  # pragma: no cover
        raise NotImplementedError

    def recognize_map(self, data_dir: Path, keys: Sequence[str]) -> Dict[str, str]:
        """Recognize ``keys`` given relative to ``data_dir``."""
        texts = self.recognize([data_dir / k for k in keys])
        return dict(zip(keys, texts))
