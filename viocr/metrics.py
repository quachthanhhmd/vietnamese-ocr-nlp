"""Scoring functions shared by every script in this repository.

The definitions here are the ones used in the Kaggle notebooks:

* every string is normalised to Unicode NFC before it is compared, because an
  accented Vietnamese character can be stored either as a single code point or
  as a base letter followed by a combining mark;
* ``ignore_space`` removes all whitespace before comparing, so that the score
  does not depend on how each framework segments the spaces;
* ``cer`` is computed at corpus level (total distance / total reference
  length), while ``norm_edit_dis`` averages the per-line normalised distance.
  The two are therefore not complementary.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from rapidfuzz.distance import Levenshtein

TONE_MARKS = {"̀", "́", "̃", "̉", "̣"}

ERROR_TYPES = ("tone", "letter", "repeat")

ERROR_TYPE_VI = {
    "tone": "sai dau thanh",
    "letter": "sai chu cai",
    "repeat": "lap ky tu",
}


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def metric_text(text: str, ignore_space: bool = True) -> str:
    """Normalise a string the way the metrics compare it."""
    text = nfc(text)
    return text.replace(" ", "") if ignore_space else text


def score(
    gt_rows: Sequence[Tuple[str, str]],
    preds: Dict[str, str],
    ignore_space: bool = True,
) -> Tuple[Dict[str, float], List[dict]]:
    """Return (metrics, per-line records) for one prediction dictionary.

    ``gt_rows`` is a sequence of ``(image_path, reference)`` pairs and ``preds``
    maps an image path to the predicted string. A missing key counts as an
    empty prediction, which is what happens when a recognizer fails on a line.
    """
    records: List[dict] = []
    normalised: List[float] = []
    correct = dist_sum = len_sum = word_dist = word_len = 0

    for rel, gt in gt_rows:
        pred = preds.get(rel, "")
        a, b = metric_text(pred, ignore_space), metric_text(gt, ignore_space)
        dist = Levenshtein.distance(a, b)

        correct += int(a == b)
        normalised.append(Levenshtein.normalized_distance(a, b))
        dist_sum += dist
        len_sum += len(b)

        gt_words = nfc(gt).split()
        word_dist += Levenshtein.distance(nfc(pred).split(), gt_words)
        word_len += len(gt_words)

        records.append({"image": rel, "gt": gt, "pred": pred, "dist": dist})

    metrics = {
        "n": len(gt_rows),
        "acc": correct / max(len(gt_rows), 1),
        "norm_edit_dis": 1 - float(np.mean(normalised)) if normalised else 0.0,
        "cer": dist_sum / max(len_sum, 1),
        "wer": word_dist / max(word_len, 1),
    }
    return metrics, records


def strip_tone(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return nfc("".join(c for c in decomposed if c not in TONE_MARKS))


def deaccent(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )


def classify(gt: str, pred: str) -> str:
    """Assign one of ``ERROR_TYPES`` to a failed line.

    ``tone``   the two strings become equal once every tone mark is removed;
    ``repeat`` an inserted segment only repeats one of its neighbours
               (the ``nhunng`` / ``nguuoi`` pattern of the task description);
    ``letter`` anything else.
    """
    gt, pred = nfc(gt), nfc(pred)
    if gt != pred and strip_tone(gt) == strip_tone(pred):
        return "tone"

    for op, i1, i2, j1, j2 in Levenshtein.opcodes(gt, pred):
        if op != "insert":
            continue
        segment = deaccent(pred[j1:j2])
        left = deaccent(pred[j1 - 1]) if j1 > 0 else ""
        right = deaccent(pred[j2]) if j2 < len(pred) else ""
        if segment and all(c == left or c == right for c in segment):
            return "repeat"

    return "letter"


def error_distribution(records: Iterable[dict]) -> Dict[str, int]:
    """Count the error types over the records whose prediction differs."""
    counts = {t: 0 for t in ERROR_TYPES}
    for record in records:
        if nfc(record["gt"]) != nfc(record["pred"]):
            counts[classify(record["gt"], record["pred"])] += 1
    return counts
