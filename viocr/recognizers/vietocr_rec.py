"""VietOCR (``vgg_transformer``) recognizer.

Two details of the checkpoint handling are worth knowing:

* a checkpoint saved as an *extracted* torch archive (a directory containing
  ``data.pkl``) has to be zipped again before ``torch.load`` accepts it, and the
  files inside carry a 1979 timestamp that ``ZipFile.write`` refuses, so the
  entries are written with an explicit date;
* a zip whose members sit at the root instead of under ``archive/`` triggers
  ``file in archive is not in a subdirectory: byteorder`` and is repacked.
"""

from __future__ import annotations

import unicodedata
import zipfile
from pathlib import Path
from typing import List, Sequence

from ..config import VIETOCR_CFG, VIETOCR_VOCAB
from .base import Recognizer

FIXED_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def resolve_checkpoint(path: Path, work_dir: Path) -> Path:
    """Return a path that ``torch.load`` can open."""
    work_dir.mkdir(parents=True, exist_ok=True)

    if path.is_dir():
        if (path / "data.pkl").exists():
            out = work_dir / "vietocr_repacked.pth"
            with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
                for item in sorted(path.rglob("*")):
                    if item.is_file():
                        info = zipfile.ZipInfo(
                            "archive/" + str(item.relative_to(path)),
                            date_time=FIXED_ZIP_DATE,
                        )
                        archive.writestr(info, item.read_bytes())
            return out

        candidates = (
            sorted(path.glob("*.pth")) + sorted(path.glob("*.pt")) + sorted(path.glob("*.zip"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"no checkpoint in {path}; it contains "
                f"{sorted(p.name for p in path.iterdir())[:15]}"
            )
        path = candidates[0]

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if not n.endswith("/")]
        if any("/" not in n for n in names):
            out = work_dir / "vietocr_fixed.pth"
            with zipfile.ZipFile(path) as src, zipfile.ZipFile(
                out, "w", zipfile.ZIP_STORED
            ) as dst:
                for info in src.infolist():
                    if not info.is_dir():
                        dst.writestr("archive/" + info.filename, src.read(info.filename))
            return out

    return path


class VietOCRRecognizer(Recognizer):
    name = "VietOCR"

    def load(self) -> None:
        import torch
        import yaml
        from vietocr.tool.predictor import Predictor

        ckpt = self.cfg.vietocr_ckpt
        if ckpt is None or not Path(ckpt).exists():
            raise FileNotFoundError(f"vietocr_ckpt: {ckpt} does not exist")

        weights = resolve_checkpoint(Path(ckpt), self.cfg.work_dir)
        obj = torch.load(weights, map_location="cpu", weights_only=False)

        state = obj
        for key in ("state_dict", "model_state_dict", "model"):
            inner = obj.get(key) if isinstance(obj, dict) else None
            if isinstance(inner, dict) and inner and all(
                torch.is_tensor(v) for v in inner.values()
            ):
                state = inner
                break
        if any(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}

        clean = self.cfg.work_dir / "vietocr_state.pth"
        torch.save(state, clean)

        if self.cfg.vietocr_yaml and Path(self.cfg.vietocr_yaml).exists():
            cfg = dict(yaml.safe_load(Path(self.cfg.vietocr_yaml).read_text(encoding="utf-8")))
        else:
            cfg = dict(VIETOCR_CFG)

        for key, value in VIETOCR_CFG.items():
            cfg.setdefault(key, value)
        cfg["cnn"] = {**VIETOCR_CFG["cnn"], **cfg.get("cnn", {}), "pretrained": False}
        cfg["dataset"] = {**VIETOCR_CFG["dataset"], **cfg.get("dataset", {})}

        needed = int(state["transformer.fc.bias"].shape[0]) - 4
        vocab = cfg.get("vocab") or ""
        if len(vocab) != needed and len(VIETOCR_VOCAB) == needed:
            vocab = VIETOCR_VOCAB
        if len(vocab) != needed:
            raise ValueError(
                f"vocabulary has {len(vocab)} characters but the checkpoint has "
                f"{needed} output classes; the order must match the training run"
            )

        cfg["vocab"] = vocab
        cfg["weights"] = str(clean)
        cfg["device"] = self.cfg.resolve_device()
        cfg["predictor"]["beamsearch"] = False
        self.predictor = Predictor(cfg)

    def recognize(self, paths: Sequence[Path]) -> List[str]:
        from PIL import Image

        self._ensure_loaded()
        batch = self.cfg.vietocr_batch
        out: List[str] = []

        for start in range(0, len(paths), batch):
            chunk = list(paths[start : start + batch])
            images = []
            for path in chunk:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception:
                    images.append(None)

            usable = [(i, im) for i, im in enumerate(images) if im is not None]
            texts = [""] * len(chunk)
            if usable:
                try:
                    predicted = self.predictor.predict_batch([im for _, im in usable])
                except Exception:
                    predicted = [self.predictor.predict(im) for _, im in usable]
                for (i, _), text in zip(usable, predicted):
                    texts[i] = text

            out.extend(unicodedata.normalize("NFC", str(t)) for t in texts)

        return out
