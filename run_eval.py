#!/usr/bin/env python3
"""Evaluate the three recognizers and the ensemble on the test set.

    python run_eval.py                          # full pipeline
    python run_eval.py --baseline               # add the non-fine-tuned PaddleOCR
    python run_eval.py --limit 200              # quick smoke run
    python run_eval.py --from-predictions results   # re-score saved JSONL files

Everything the report needs is written to ``results/``: one JSONL per system,
the comparison table, the voting analysis and the error analysis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import pandas as pd

from viocr.config import Config
from viocr.data import check_images, read_labels
from viocr.metrics import metric_text, score
from viocr.recognizers import build
from viocr.report import (
    comparison_table, consensus_table, delta_table, error_samples,
    error_type_table, plot_consensus, plot_error_types, plot_metrics,
    venn_table, write_jsonl,
)
from viocr.voting import build_lexicon, vote_char, vote_lexicon, vote_line

LINE, LEX, CHAR = "VOTING (line)", "VOTING (line+lexicon)", "VOTING (char)"


def slug(name: str) -> str:
    """File-name form of a system name: ``VOTING (char)`` -> ``voting_char``."""
    keep = [c if c.isalnum() else " " for c in name.lower()]
    return "_".join("".join(keep).split())


def sanity_check(preds: Dict[str, str], rows, ignore_space: bool, n: int = 300):
    """Reject a recognizer whose output is empty or unrelated to the references."""
    subset = rows[:n]
    empty = sum(1 for rel, _ in subset if not preds.get(rel, "").strip())
    metrics, _ = score(subset, preds, ignore_space)
    if empty > len(subset) * 0.5:
        return False, f"{empty}/{len(subset)} empty lines"
    if metrics["acc"] < 0.01 and metrics["cer"] > 0.5:
        return False, f"acc {metrics['acc'] * 100:.1f}%, CER {metrics['cer'] * 100:.0f}%"
    return True, f"acc {metrics['acc'] * 100:.1f}%, CER {metrics['cer'] * 100:.2f}%"


def load_saved(path: Path) -> Dict[str, str]:
    preds = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                preds[record["image"]] = record["pred"]
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--baseline", action="store_true",
                        help="also evaluate the released PaddleOCR (requirement 2.1)")
    parser.add_argument("--from-predictions", type=Path, default=None,
                        help="re-score existing JSONL files instead of running the models")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config, device=args.device)
    cfg.check()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    rows = read_labels(cfg.test_file, args.limit)
    check_images(cfg.data_dir, rows)
    keys = [rel for rel, _ in rows]
    gt = dict(rows)
    print(f"test set: {len(rows):,} lines, "
          f"{sum(len(metric_text(t, cfg.ignore_space)) for _, t in rows):,} characters")

    wanted = args.models or list(cfg.priority)
    preds: Dict[str, Dict[str, str]] = {}
    seconds: Dict[str, float] = {}

    if args.baseline:
        wanted = ["PaddleOCR-baseline"] + wanted

    for name in wanted:
        if args.from_predictions:
            path = args.from_predictions / f"pred_test_{slug(name)}.jsonl"
            if not path.exists():
                print(f"[{name}] no saved predictions at {path}")
                continue
            preds[name] = load_saved(path)
            seconds[name] = 0.0
            print(f"[{name}] loaded {len(preds[name]):,} saved predictions")
            continue

        try:
            recognizer = build(
                "PaddleOCR" if name.startswith("PaddleOCR-baseline") else name,
                cfg,
                **({"baseline": True} if name.endswith("baseline") else {}),
            )
            start = time.time()
            result = recognizer.recognize_map(cfg.data_dir, keys)
            seconds[name] = time.time() - start
        except Exception as exc:
            print(f"[{name}] failed: {type(exc).__name__}: {exc}")
            continue

        write_jsonl(cfg.results_dir / f"pred_test_{slug(name)}.jsonl", rows, result)
        ok, why = sanity_check(result, rows, cfg.ignore_space)
        print(f"[{name}] {len(rows):,} lines in {seconds[name]:.0f}s "
              f"({seconds[name] / max(len(rows), 1) * 1000:.0f} ms/line) — {why}")
        if not ok:
            print(f"[{name}] rejected, excluded from the ensemble")
            continue
        preds[name] = result

    ensemble = {n: p for n, p in preds.items() if not n.endswith("baseline")}
    order = [n for n in cfg.priority if n in ensemble] + [
        n for n in ensemble if n not in cfg.priority
    ]
    if not order:
        raise SystemExit("no recognizer produced usable predictions")

    lexicon = None
    if cfg.use_lexicon and cfg.train_file and Path(cfg.train_file).exists():
        lexicon = build_lexicon(Path(cfg.train_file))
        print(f"lexicon: {len(lexicon):,} distinct words from {Path(cfg.train_file).name}")

    line_pred, lex_pred, char_pred = {}, {}, {}
    why, source = {}, {}
    for rel in keys:
        line_pred[rel], why[rel], source[rel] = vote_line(ensemble, order, rel, cfg.ignore_space)
        char_pred[rel] = vote_char(ensemble, order, rel)
        lex_pred[rel] = (
            vote_lexicon(ensemble, order, rel, lexicon, cfg.ignore_space)[0]
            if lexicon else line_pred[rel]
        )

    allpred = {**ensemble, LINE: line_pred, LEX: lex_pred, CHAR: char_pred}
    if "PaddleOCR-baseline" in preds:
        allpred["PaddleOCR-baseline"] = preds["PaddleOCR-baseline"]

    for name, mapping in ((LINE, line_pred), (LEX, lex_pred), (CHAR, char_pred)):
        write_jsonl(cfg.results_dir / f"pred_test_{slug(name)}.jsonl", rows, mapping)

    metrics, records = {}, {}
    for name, mapping in allpred.items():
        metrics[name], records[name] = score(rows, mapping, cfg.ignore_space)
    correct = {
        name: {rel for rel, g in rows
               if metric_text(mapping.get(rel, ""), cfg.ignore_space) == metric_text(g, cfg.ignore_space)}
        for name, mapping in allpred.items()
    }
    oracle = set().union(*[correct[n] for n in order])

    entries = ([n for n in preds if n.endswith("baseline")] + order + [LINE, LEX, CHAR])
    table = comparison_table(entries, metrics, correct, oracle, len(rows), seconds)
    table.to_csv(cfg.results_dir / "comparison_test.csv", index=False)
    print("\n" + table.to_string(index=False))

    best = max(order, key=lambda n: metrics[n]["acc"])
    for name in (LINE, LEX, CHAR):
        print(f"{name:<24} {metrics[name]['acc'] * 100:6.2f}%   "
              f"{(metrics[name]['acc'] - metrics[best]['acc']) * 100:+.2f} vs {best}")
    print(f"{'ORACLE':<24} {len(oracle) / len(rows) * 100:6.2f}%   "
          f"{(len(oracle) / len(rows) - metrics[best]['acc']) * 100:+.2f} — upper bound")

    final = CHAR
    cons = consensus_table(rows, why, correct[LINE])
    venn = venn_table(order, correct, len(rows))
    delta = delta_table(order, [LINE, LEX, CHAR], correct)
    samples = error_samples(records[final], cfg.seed)
    types = error_type_table(records[final])

    for name, frame in (
        ("vote_situations", cons), ("venn_regions", venn), ("voting_delta", delta),
        ("error_samples", samples), ("error_types", types),
    ):
        frame.to_csv(cfg.results_dir / f"{name}.csv", index=False)

    print("\n" + cons.to_string(index=False))
    print("\n" + types.to_string(index=False))

    plot_metrics(entries, metrics, len(oracle) / len(rows), cfg.results_dir / "fig_metrics.png")
    plot_consensus(cons, cfg.results_dir / "fig_consensus.png")
    plot_error_types(types, cfg.results_dir / "fig_error_types.png")

    summary = {
        "test_set": str(cfg.test_file),
        "n": len(rows),
        "seed": cfg.seed,
        "ignore_space": cfg.ignore_space,
        "priority": order,
        "models": {n: {k: round(metrics[n][k], 4) for k in ("acc", "norm_edit_dis", "cer", "wer")}
                   for n in entries},
        "inference_seconds": {n: round(v, 1) for n, v in seconds.items()},
        "oracle_acc": round(len(oracle) / len(rows), 4),
        "vote_situations": cons.to_dict("records"),
        "venn_regions": venn.to_dict("records"),
        "voting_delta": delta.to_dict("records"),
        "error_types": types.to_dict("records"),
    }
    (cfg.results_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nresults written to {cfg.results_dir}")
    for path in sorted(cfg.results_dir.iterdir()):
        print(f"  {path.name:<40}{path.stat().st_size / 1024:>9.1f} KB")


if __name__ == "__main__":
    main()
