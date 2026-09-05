"""Ensemble combination of several recognizer outputs.

Three strategies are implemented, all of them driven by the same priority
order (the first model of the order is the *pivot* and also the fallback):

``vote_line``      compare the complete strings and keep the one produced by at
                   least two models; fall back to the pivot otherwise.
``vote_lexicon``   same as ``vote_line``, but when the three models disagree the
                   hypothesis with the fewest out-of-vocabulary words is kept,
                   the vocabulary being built from the training labels.
``vote_char``      align the other hypotheses to the pivot with the Levenshtein
                   edit operations and vote independently at every character
                   position and at every insertion gap (ROVER).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from rapidfuzz.distance import Levenshtein

from .metrics import metric_text, nfc

WORD_RE = re.compile(r"[^0-9A-Za-zÀ-ỹ]+")

Predictions = Dict[str, Dict[str, str]]


def vote_line(
    preds: Predictions,
    order: Sequence[str],
    rel: str,
    ignore_space: bool = True,
) -> Tuple[str, str, str]:
    """Return (text, reason, deciding model) for line-level majority voting."""
    texts = [preds[n].get(rel, "") for n in order]
    keys = [metric_text(t, ignore_space) for t in texts]
    top, n_top = Counter(keys).most_common(1)[0]

    if n_top >= 2:
        for name, key, text in zip(order, keys, texts):
            if key == top:
                return text, f"consensus {n_top}/{len(order)}", name

    return texts[0], "no agreement", order[0]


def align(pivot: str, other: str) -> Tuple[List[str], List[str]]:
    """Project ``other`` onto the coordinate frame of ``pivot``.

    Returns ``(gaps, subs)`` where ``subs[i]`` is the character aligned to
    ``pivot[i]`` (empty string for a deletion) and ``gaps[i]`` is the string
    inserted just before ``pivot[i]``. ``gaps`` has one extra slot for the
    insertion after the last character.
    """
    gaps = [""] * (len(pivot) + 1)
    subs = [""] * len(pivot)

    for op, i1, i2, j1, j2 in Levenshtein.opcodes(pivot, other):
        if op in ("equal", "replace"):
            segment = other[j1:j2]
            for k in range(i2 - i1):
                subs[i1 + k] = segment[k] if k < len(segment) else ""
            if len(segment) > i2 - i1 and i2 > i1:
                subs[i2 - 1] += segment[i2 - i1:]
        elif op == "delete":
            for k in range(i1, i2):
                subs[k] = ""
        elif op == "insert":
            gaps[i1] += other[j1:j2]

    return gaps, subs


def vote_char(preds: Predictions, order: Sequence[str], rel: str) -> str:
    """Character-level (ROVER) voting; ties are resolved by the pivot."""
    pivot = preds[order[0]].get(rel, "")
    aligned = [align(pivot, preds[n].get(rel, "")) for n in order]

    out: List[str] = []
    for i in range(len(pivot) + 1):
        top, count = Counter(a[0][i] for a in aligned).most_common(1)[0]
        out.append(top if count >= 2 else aligned[0][0][i])
        if i < len(pivot):
            top, count = Counter(a[1][i] for a in aligned).most_common(1)[0]
            out.append(top if count >= 2 else pivot[i])

    return "".join(out)


def build_lexicon(label_file: Path) -> Counter:
    """Word frequencies taken from the second column of a label file."""
    vocabulary: Counter = Counter()
    with open(label_file, encoding="utf-8") as handle:
        for line in handle:
            if "\t" not in line:
                continue
            text = nfc(line.split("\t", 1)[1])
            for word in WORD_RE.split(text):
                if word:
                    vocabulary[word.lower()] += 1
    return vocabulary


def count_oov(text: str, lexicon: Counter) -> int:
    return sum(
        1
        for word in WORD_RE.split(nfc(text))
        if word and lexicon[word.lower()] == 0
    )


def vote_lexicon(
    preds: Predictions,
    order: Sequence[str],
    rel: str,
    lexicon: Counter,
    ignore_space: bool = True,
) -> Tuple[str, str, str]:
    """Line-level voting with a lexicon arbitrating the full-disagreement case."""
    text, why, src = vote_line(preds, order, rel, ignore_space)
    if not lexicon or why != "no agreement":
        return text, why, src

    texts = [preds[n].get(rel, "") for n in order]
    best = min(range(len(order)), key=lambda j: (count_oov(texts[j], lexicon), j))
    if best == 0:
        return text, why, src
    return texts[best], "lexicon arbitration", order[best]


def combine(
    preds: Predictions,
    order: Sequence[str],
    keys: Sequence[str],
    lexicon: Counter | None = None,
    ignore_space: bool = True,
) -> Dict[str, dict]:
    """Run every strategy over ``keys`` and return the three prediction maps."""
    line: Dict[str, str] = {}
    why: Dict[str, str] = {}
    source: Dict[str, str] = {}
    lex: Dict[str, str] = {}
    char: Dict[str, str] = {}

    for rel in keys:
        line[rel], why[rel], source[rel] = vote_line(preds, order, rel, ignore_space)
        char[rel] = vote_char(preds, order, rel)
        if lexicon:
            lex[rel] = vote_lexicon(preds, order, rel, lexicon, ignore_space)[0]
        else:
            lex[rel] = line[rel]

    return {"line": line, "char": char, "lexicon": lex, "why": why, "source": source}
