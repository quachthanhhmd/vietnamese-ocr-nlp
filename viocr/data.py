"""Reading the label files of the ``vi_rec_100k`` dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from .metrics import nfc


def read_labels(path: Path, limit: int | None = None) -> List[Tuple[str, str]]:
    """Read ``<image path>\\t<text>`` lines into ``(image, reference)`` pairs."""
    rows: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue
            rel, text = line.split("\t", 1)
            rows.append((os.path.normpath(rel), nfc(text)))
            if limit and len(rows) >= limit:
                break
    return rows


def check_images(data_dir: Path, rows: List[Tuple[str, str]]) -> None:
    """Verify that the first entry resolves, i.e. that ``data_dir`` is right."""
    if not rows:
        raise ValueError("the label file is empty")
    first = data_dir / rows[0][0]
    if not first.exists():
        raise FileNotFoundError(
            f"the label file points to {rows[0][0]} but {first} does not exist; "
            "data_dir must be the directory the label paths are relative to"
        )
