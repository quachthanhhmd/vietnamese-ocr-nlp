#!/usr/bin/env python3
"""Run the ensemble on any image, without needing a label file.

    python predict.py page_line_01.jpg
    python predict.py samples/*.jpg --save results/demo.jsonl
    python predict.py line.png --models VietOCR TrOCR
    python predict.py line.png --show-all

The three recognizers are run on the given images and combined with the same
voting rules as in the evaluation, so the output of this script is what the
system would produce for these lines on the test set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from viocr.config import Config
from viocr.recognizers import build
from viocr.voting import build_lexicon, vote_char, vote_line, vote_lexicon

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(inputs: List[str]) -> List[Path]:
    images: List[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            images += sorted(
                p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.exists():
            images.append(path)
        else:
            raise FileNotFoundError(item)
    if not images:
        raise SystemExit("no image found")
    return [p.resolve() for p in images]


def common_root(images: List[Path]) -> Path:
    """PaddleOCR addresses images relative to a root directory."""
    if len(images) == 1:
        return images[0].parent
    return Path(os.path.commonpath([str(p.parent) for p in images]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images", nargs="+", help="image files, globs or a directory")
    parser.add_argument("-c", "--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--models", nargs="+", default=None,
                        help="subset of recognizers to run (default: all three)")
    parser.add_argument("--save", type=Path, default=None, help="write a JSONL file")
    parser.add_argument("--show-all", action="store_true",
                        help="print every recognizer, not only the ensemble")
    parser.add_argument("--device", default=None, help="cpu | cuda:0 | auto")
    args = parser.parse_args()

    images = collect_images(args.images)
    cfg = Config.load(args.config, device=args.device)
    cfg.data_dir = common_root(images).resolve()
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    wanted = args.models or list(cfg.priority)
    keys = [str(p.relative_to(cfg.data_dir)) for p in images]

    preds: Dict[str, Dict[str, str]] = {}
    for name in wanted:
        try:
            recognizer = build(name, cfg)
            preds[name] = recognizer.recognize_map(cfg.data_dir, keys)
        except Exception as exc:  # a missing checkpoint must not stop the others
            print(f"[{name}] skipped: {type(exc).__name__}: {exc}", file=sys.stderr)

    order = [n for n in cfg.priority if n in preds] + [
        n for n in preds if n not in cfg.priority
    ]
    if not order:
        raise SystemExit("no recognizer could be loaded; check the paths in the config")

    lexicon = None
    if cfg.use_lexicon and cfg.train_file and Path(cfg.train_file).exists():
        lexicon = build_lexicon(Path(cfg.train_file))

    records = []
    for key, image in zip(keys, images):
        line, why, source = vote_line(preds, order, key, cfg.ignore_space)
        char = vote_char(preds, order, key)
        final = char
        if lexicon:
            line, why, source = vote_lexicon(preds, order, key, lexicon, cfg.ignore_space)

        print(f"\n{image}")
        if args.show_all or len(order) > 1:
            width = max(len(n) for n in order + ["VOTING (line)", "VOTING (char)"])
            for name in order:
                print(f"  {name:<{width}} : {preds[name].get(key, '')}")
            print(f"  {'VOTING (line)':<{width}} : {line}      [{why} -> {source}]")
            print(f"  {'VOTING (char)':<{width}} : {char}")
        else:
            print(f"  {final}")

        records.append(
            {
                "image": str(image),
                "pred": final,
                "voting_line": line,
                "voting_char": char,
                "reason": why,
                "decided_by": source,
                **{f"pred_{n.lower()}": preds[n].get(key, "") for n in order},
            }
        )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nsaved {len(records)} predictions to {args.save}")


if __name__ == "__main__":
    main()
