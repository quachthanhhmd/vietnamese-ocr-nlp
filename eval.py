#!/usr/bin/env python3
"""Score a prediction file.

    python eval.py results/pred_test_voting_char.jsonl
    python eval.py results/pred_test_baseline.jsonl --keep-space
    python eval.py results/*.jsonl --errors

Each line of the input must be a JSON object with the keys ``image``, ``gt``
and ``pred``. ``acc`` and ``CER`` ignore whitespace by default, which is the
convention used throughout the report; ``--keep-space`` switches it off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from viocr.metrics import ERROR_TYPE_VI, ERROR_TYPES, error_distribution, score


def read_jsonl(path: Path) -> Tuple[List[Tuple[str, str]], dict]:
    rows: List[Tuple[str, str]] = []
    preds = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append((record["image"], record["gt"]))
            preds[record["image"]] = record["pred"]
    return rows, preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", nargs="+", type=Path,
                        help="prediction file(s) with image / gt / pred per line")
    parser.add_argument("--keep-space", action="store_true",
                        help="compare the spaces too (default: ignore them)")
    parser.add_argument("--errors", action="store_true",
                        help="also print the error-type breakdown")
    args = parser.parse_args()

    ignore_space = not args.keep_space

    for path in args.jsonl:
        rows, preds = read_jsonl(path)
        metrics, records = score(rows, preds, ignore_space)

        print(f"\n{path}")
        print(f"  n              : {metrics['n']:,}")
        print(f"  ignore_space   : {ignore_space}")
        print(f"  acc            : {metrics['acc']:.4f}")
        print(f"  norm_edit_dis  : {metrics['norm_edit_dis']:.4f}")
        print(f"  CER            : {metrics['cer']:.4f}")
        print(f"  WER            : {metrics['wer']:.4f}")

        if args.errors:
            counts = error_distribution(records)
            total = max(sum(counts.values()), 1)
            print(f"  wrong lines    : {total:,}")
            for key in ERROR_TYPES:
                print(f"    {key:<8} ({ERROR_TYPE_VI[key]:<14}) "
                      f"{counts[key]:>6}  {counts[key] / total * 100:5.1f}%")


if __name__ == "__main__":
    main()
