"""Tables and figures for the report."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import ERROR_TYPE_VI, ERROR_TYPES, classify, metric_text, nfc

PALETTE = ["#4C78A8", "#54A24B", "#72B7B2", "#E45756", "#F58518", "#B279A2"]


def write_jsonl(path: Path, rows: Sequence[Tuple[str, str]], preds: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for rel, gt in rows:
            record = {
                "image": rel.replace("\\", "/"),
                "gt": gt,
                "pred": preds.get(rel, ""),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def comparison_table(
    entries: Sequence[str],
    metrics: Dict[str, dict],
    correct: Dict[str, set],
    oracle: set,
    n: int,
    seconds: Dict[str, float] | None = None,
) -> pd.DataFrame:
    seconds = seconds or {}
    rows = []
    for name in entries:
        m = metrics[name]
        rows.append(
            {
                "Model": name,
                "acc": round(m["acc"], 4),
                "norm_edit_dis": round(m["norm_edit_dis"], 4),
                "CER": round(m["cer"], 4),
                "WER": round(m["wer"], 4),
                "correct_lines": len(correct[name]),
                "inference_s": round(seconds[name], 1) if name in seconds else None,
            }
        )
    rows.append(
        {
            "Model": "ORACLE",
            "acc": round(len(oracle) / max(n, 1), 4),
            "norm_edit_dis": None,
            "CER": None,
            "WER": None,
            "correct_lines": len(oracle),
            "inference_s": None,
        }
    )
    return pd.DataFrame(rows)


def consensus_table(
    rows: Sequence[Tuple[str, str]],
    why: Dict[str, str],
    correct_final: set,
) -> pd.DataFrame:
    labels = sorted(
        set(why.values()),
        key=lambda s: (1, 0) if "/" not in s else (0, -int(s.split("/")[0][-1])),
    )
    out = []
    for label in labels:
        keys = [rel for rel, _ in rows if why[rel] == label]
        hit = sum(1 for rel in keys if rel in correct_final)
        out.append(
            {
                "situation": label,
                "lines": len(keys),
                "share_%": round(len(keys) / max(len(rows), 1) * 100, 2),
                "correct": hit,
                "accuracy_%": round(hit / max(len(keys), 1) * 100, 2),
            }
        )
    return pd.DataFrame(out)


def venn_table(order: Sequence[str], correct: Dict[str, set], total: int) -> pd.DataFrame:
    regions = []
    covered = set()
    for size in range(len(order), 0, -1):
        for combo in itertools.combinations(order, size):
            others = [correct[n] for n in order if n not in combo]
            group = set.intersection(*[correct[n] for n in combo]) - set().union(*others, set())
            regions.append(("+".join(combo), len(group)))
            covered |= group
    regions.append(("none", total - len(covered)))
    return pd.DataFrame(regions, columns=["recognized_by", "lines"])


def delta_table(
    order: Sequence[str], voting_names: Sequence[str], correct: Dict[str, set]
) -> pd.DataFrame:
    rows = []
    for voting in voting_names:
        for reference in order:
            fixed = len(correct[voting] - correct[reference])
            broken = len(correct[reference] - correct[voting])
            rows.append(
                {
                    "voting": voting,
                    "vs": reference,
                    "fixed": fixed,
                    "broken": broken,
                    "net": fixed - broken,
                }
            )
    return pd.DataFrame(rows)


def error_samples(records: Sequence[dict], seed: int, k: int = 20) -> pd.DataFrame:
    wrong = [r for r in records if nfc(r["gt"]) != nfc(r["pred"])]
    if not wrong:
        return pd.DataFrame(columns=["image", "gt", "pred", "type", "type_vi"])
    sample = random.Random(seed).sample(wrong, min(k, len(wrong)))
    return pd.DataFrame(
        [
            {
                "image": r["image"],
                "gt": r["gt"],
                "pred": r["pred"],
                "type": classify(r["gt"], r["pred"]),
                "type_vi": ERROR_TYPE_VI[classify(r["gt"], r["pred"])],
            }
            for r in sample
        ]
    )


def error_type_table(records: Sequence[dict]) -> pd.DataFrame:
    wrong = [r for r in records if nfc(r["gt"]) != nfc(r["pred"])]
    counts = {t: 0 for t in ERROR_TYPES}
    for record in wrong:
        counts[classify(record["gt"], record["pred"])] += 1
    total = max(len(wrong), 1)
    return pd.DataFrame(
        [
            {
                "type": t,
                "type_vi": ERROR_TYPE_VI[t],
                "lines": counts[t],
                "share_%": round(counts[t] / total * 100, 1),
            }
            for t in ERROR_TYPES
        ]
    )


def plot_metrics(
    entries: Sequence[str], metrics: Dict[str, dict], oracle_acc: float, out: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"figure.dpi": 110, "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
         "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(entries))]
    labels = [e.replace(" (", "\n(") for e in entries]

    panels = [
        ("acc", "Accuracy (%)", 100, "%.2f", 100.0, True),
        ("norm_edit_dis", "norm_edit_dis (higher is better)", 1, "%.4f", 1.0, False),
        ("cer", "CER (%) — lower is better", 100, "%.2f", 100.0, False),
        ("wer", "WER (%) — lower is better", 100, "%.2f", 100.0, False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    for ax, (key, title, mul, fmt, cap, mark) in zip(axes, panels):
        values = [metrics[n][key] * mul for n in entries]
        bars = ax.bar(labels, values, color=colors, width=0.65)
        ax.bar_label(bars, fmt=fmt, padding=2, fontsize=9)
        low, high = min(values), max(values)
        pad = max((high - low) * 0.45, high * 0.05)
        if mark:
            high = max(high, oracle_acc * 100)
            ax.axhline(oracle_acc * 100, ls="--", lw=1.2, c="#888")
            ax.text(-0.45, oracle_acc * 100, f"oracle {oracle_acc * 100:.2f}",
                    va="bottom", ha="left", fontsize=8, color="#666")
        ax.set_ylim(max(0, low - pad), min(high + pad * 1.5, cap))
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_consensus(table: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = ["#54A24B", "#F58518", "#E45756"][: len(table)]

    axes[0].pie(table["lines"], labels=table["situation"], colors=colors,
                autopct="%1.1f%%", startangle=110,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[0].set_title("Agreement between the recognizers", fontsize=11)

    x = np.arange(len(table))
    hit = table["correct"].to_numpy()
    total = table["lines"].to_numpy()
    axes[1].bar(x, hit, color="#54A24B", label="ensemble correct")
    axes[1].bar(x, total - hit, bottom=hit, color="#E45756", label="ensemble wrong")
    for i, (h, t) in enumerate(zip(hit, total)):
        axes[1].text(i, t, f"{h / max(t, 1) * 100:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s.replace(" ", "\n", 1) for s in table["situation"]], fontsize=9)
    axes[1].set_ylim(0, total.max() * 1.18)
    axes[1].set(title="Reliability of each situation", ylabel="lines")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25, axis="y")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_error_types(table: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.4))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    bars = ax.bar(table["type_vi"], table["share_%"], color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    ax.set(title=f"Error types over {int(table['lines'].sum()):,} wrong lines", ylabel="%")
    ax.set_ylim(0, max(table["share_%"]) * 1.2)
    ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
