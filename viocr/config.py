"""Configuration objects and the constants that must match the training runs.

Every value in ``VIETOCR_CFG`` is part of the model definition, not a free
choice at inference time. ``image_height``, ``image_max_width`` and the
``cnn.ss`` / ``cnn.ks`` pooling strides have no learnable parameters, so a wrong
value loads silently and destroys the accuracy instead of raising an error.
``VIETOCR_VOCAB`` is an ordered string: character *i* is output class *i + 4*,
so a permutation turns the predictions into a substitution cipher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

VIETOCR_VOCAB = (
    "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhH"
    "iIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦ"
    "ũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]"
    "^_`{|}~ «²¹»äçüāēěīōūǎǒǔ–—’“”•…"
)

VIETOCR_CFG: Dict = {
    "vocab": VIETOCR_VOCAB,
    "backbone": "vgg19_bn",
    "seq_modeling": "transformer",
    "cnn": {
        "pretrained": False,
        "hidden": 256,
        "ss": [[2, 2], [2, 2], [2, 1], [2, 1], [2, 1]],
        "ks": [[2, 2], [2, 2], [2, 1], [2, 1], [2, 1]],
    },
    "transformer": {
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "max_seq_length": 1024,
        "pos_dropout": 0.1,
        "trans_dropout": 0.1,
    },
    "dataset": {"image_height": 64, "image_min_width": 32, "image_max_width": 1024},
    "predictor": {"beamsearch": False},
    "quiet": True,
}

PADDLE_BASE_CONFIG = "configs/rec/PP-OCRv5/multi_language/latin_PP-OCRv5_mobile_rec.yml"
PADDLE_REPO = "https://github.com/PaddlePaddle/PaddleOCR.git"
PADDLE_REF = "v3.3.0"
PADDLE_PRETRAINED_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
    "official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams"
)

DEFAULT_PRIORITY = ["VietOCR", "PaddleOCR", "TrOCR"]


def _as_path(value: Optional[str]) -> Optional[Path]:
    """Absolute path, because PaddleOCR is invoked with its own working directory."""
    return Path(value).expanduser().resolve() if value else None


@dataclass
class Config:
    """Everything the pipeline needs, loaded from ``configs/default.yaml``."""

    data_dir: Optional[Path] = None
    test_file: Optional[Path] = None
    train_file: Optional[Path] = None
    dict_file: Optional[Path] = None

    vietocr_ckpt: Optional[Path] = None
    vietocr_yaml: Optional[Path] = None
    paddle_ckpt: Optional[Path] = None
    paddle_baseline: Optional[Path] = None
    trocr_ckpt: Optional[Path] = None

    results_dir: Path = Path("results")
    work_dir: Path = Path(".viocr_cache")

    seed: int = 2026
    ignore_space: bool = True
    priority: List[str] = field(default_factory=lambda: list(DEFAULT_PRIORITY))
    vietocr_batch: int = 16
    trocr_batch: int = 24
    device: str = "auto"
    use_lexicon: bool = True

    @classmethod
    def load(cls, path: Optional[Path] = None, **overrides) -> "Config":
        raw: Dict = {}
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})

        path_fields = {
            "data_dir", "test_file", "train_file", "dict_file",
            "vietocr_ckpt", "vietocr_yaml", "paddle_ckpt", "paddle_baseline",
            "trocr_ckpt", "results_dir", "work_dir",
        }
        kwargs = {}
        for key, value in raw.items():
            if key not in cls.__dataclass_fields__:
                continue
            kwargs[key] = _as_path(value) if key in path_fields else value

        cfg = cls(**kwargs)
        if cfg.data_dir and cfg.test_file is None:
            cfg.test_file = cfg.data_dir / "rec_test.txt"
        if cfg.data_dir and cfg.train_file is None:
            cfg.train_file = cfg.data_dir / "rec_train.txt"
        if cfg.data_dir and cfg.dict_file is None:
            cfg.dict_file = cfg.data_dir / "vi_dict.txt"
        cfg.results_dir = Path(cfg.results_dir).expanduser().resolve()
        cfg.work_dir = Path(cfg.work_dir).expanduser().resolve()
        return cfg

    def check(self, need_test: bool = True) -> None:
        """Fail early with a message that says which path is wrong."""
        required = [("data_dir", self.data_dir)]
        if need_test:
            required += [("test_file", self.test_file)]
        for label, value in required:
            if value is None or not value.exists():
                raise FileNotFoundError(f"{label}: {value} does not exist")

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
