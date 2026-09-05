"""PaddleOCR (PP-OCRv5 mobile, CTC head) recognizer.

The submitted checkpoint is a *training* checkpoint (``model.pdparams``), not an
exported inference model, so recognition is done by calling
``tools/infer_rec.py`` of the PaddleOCR repository with a generated config. The
repository is cloned on first use and pinned to the tag that was used for
training.

Two modes are available:

``baseline``   the released Latin model with its own character dictionary, used
               for the required non-fine-tuned baseline;
``finetuned``  the same architecture with ``vi_dict.txt`` as the output
               dictionary and the group's checkpoint.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

from ..config import PADDLE_BASE_CONFIG, PADDLE_PRETRAINED_URL, PADDLE_REF, PADDLE_REPO
from .base import Recognizer

BACKBONE_FILES = ["rec_lcnetv3.py", "rec_hgnet.py", "rec_pphgnetv2.py"]
BACKBONE_OLD = "x = F.adaptive_avg_pool2d(x, [1, 40])"
BACKBONE_NEW = "x = F.adaptive_avg_pool2d(x, [1, max(1, x.shape[3] // 2)])"


def run(args: Sequence[str], cwd: Path) -> None:
    process = subprocess.run(
        [str(a) for a in args], cwd=str(cwd), capture_output=True, text=True
    )
    if process.returncode != 0:
        print(process.stdout[-2000:])
        print(process.stderr[-2000:])
        raise RuntimeError("PaddleOCR command failed: " + " ".join(map(str, args)))


def ensure_repo(repo_dir: Path) -> Path:
    """Clone PaddleOCR at the pinned tag and patch the recognition backbone.

    The backbone forces the CTC feature map to 40 columns while training but
    uses ``W/8`` at inference. Our labels are much longer than the 320-pixel
    configuration the original recipe was written for, so with 40 time steps a
    large part of the training targets is impossible (CTC needs ``T >= L``).
    The patch makes both branches agree; at width 320 it reproduces the
    original behaviour exactly.
    """
    repo_dir = repo_dir.resolve()
    if not (repo_dir / ".git").exists():
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            ["git", "clone", "--depth", "1", "--branch", PADDLE_REF, PADDLE_REPO, repo_dir.name],
            cwd=repo_dir.parent,
        )

    for name in BACKBONE_FILES:
        path = repo_dir / "ppocr/modeling/backbones" / name
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if BACKBONE_OLD in text:
                path.write_text(text.replace(BACKBONE_OLD, BACKBONE_NEW), encoding="utf-8")

    install_repo_requirements(repo_dir)
    return repo_dir


def install_repo_requirements(repo_dir: Path) -> None:
    """Install what ``ppocr`` imports, once per interpreter.

    ``tools/infer_rec.py`` pulls in ``ppocr.data.imaug``, which imports
    ``albumentations`` at module level even for inference, so the repository
    requirements have to be present before the first call.
    """
    marker = repo_dir / ".requirements_installed"
    if marker.exists():
        return
    requirements = repo_dir / "requirements.txt"
    if requirements.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
            capture_output=True,
            text=True,
        )
    marker.write_text(sys.executable, encoding="utf-8")


def download_pretrained(target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PADDLE_PRETRAINED_URL, target)
    return target


def checkpoint_prefix(path: Path) -> str:
    """``tools/infer_rec.py`` expects the path without the ``.pdparams`` suffix."""
    if path.is_dir():
        found = sorted(path.glob("*.pdparams"))
        if not found:
            raise FileNotFoundError(
                f"no *.pdparams in {path}; it contains "
                f"{sorted(p.name for p in path.iterdir())[:15]}"
            )
        path = found[0]
    return str(path)[: -len(".pdparams")]


class PaddleOCRRecognizer(Recognizer):
    name = "PaddleOCR"

    def __init__(self, cfg, baseline: bool = False) -> None:
        super().__init__(cfg)
        self.baseline = baseline
        if baseline:
            self.name = "PaddleOCR-baseline"

    def load(self) -> None:
        self.repo = ensure_repo(Path(self.cfg.work_dir).resolve() / "PaddleOCR")

        if self.baseline:
            weights = Path(self.cfg.paddle_baseline) if self.cfg.paddle_baseline else (
                Path(self.cfg.work_dir) / "weights" / "latin_PP-OCRv5_mobile_rec_pretrained.pdparams"
            )
            self.prefix = checkpoint_prefix(download_pretrained(weights))
        else:
            ckpt = self.cfg.paddle_ckpt
            if ckpt is None or not Path(ckpt).exists():
                raise FileNotFoundError(f"paddle_ckpt: {ckpt} does not exist")
            self.prefix = checkpoint_prefix(Path(ckpt))

    def _write_config(self, list_file: Path, out_file: Path) -> Path:
        cfg = yaml.safe_load((self.repo / PADDLE_BASE_CONFIG).read_text(encoding="utf-8"))

        if not self.baseline:
            if self.cfg.dict_file is None or not Path(self.cfg.dict_file).exists():
                raise FileNotFoundError(f"dict_file: {self.cfg.dict_file} does not exist")
            target = self.repo / "ppocr/utils/dict/vi_dict.txt"
            target.write_text(
                Path(self.cfg.dict_file).read_text(encoding="utf-8"), encoding="utf-8"
            )
            cfg["Global"]["character_dict_path"] = str(target)

        cfg["Global"].update(
            {
                "use_gpu": self.cfg.resolve_device().startswith("cuda"),
                "distributed": False,
                "use_amp": False,
                "infer_img": str(self.cfg.data_dir),
                "infer_list": str(list_file),
                "save_res_path": str(out_file),
            }
        )
        cfg["Metric"]["ignore_space"] = self.cfg.ignore_space
        for section in ("Train", "Eval"):
            if section in cfg:
                cfg[section]["dataset"]["data_dir"] = str(self.cfg.data_dir)

        path = Path(self.cfg.work_dir) / (
            "paddle_baseline.yml" if self.baseline else "paddle_finetune.yml"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        return path

    def recognize(self, paths: Sequence[Path]) -> List[str]:
        self._ensure_loaded()
        work = Path(self.cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)

        data_dir = Path(self.cfg.data_dir)
        relatives = [os.path.relpath(Path(p).resolve(), data_dir.resolve()) for p in paths]

        list_file = work / "paddle_list.txt"
        list_file.write_text(
            "".join(f"{rel}\t-\n" for rel in relatives), encoding="utf-8"
        )
        out_file = work / "paddle_pred.txt"
        config = self._write_config(list_file, out_file)

        run(
            [sys.executable, "tools/infer_rec.py", "-c", str(config),
             "-o", f"Global.pretrained_model={self.prefix}"],
            cwd=self.repo,
        )

        predictions: Dict[str, str] = {}
        for line in out_file.read_text(encoding="utf-8").splitlines():
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            key = os.path.normpath(os.path.relpath(parts[0], data_dir))
            predictions[key] = unicodedata.normalize("NFC", parts[1])

        return [predictions.get(os.path.normpath(rel), "") for rel in relatives]
