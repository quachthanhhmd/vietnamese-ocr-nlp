"""TrOCR (vision encoder-decoder) recognizer.

The exported model directory is read without ``TrOCRProcessor`` on purpose.
``transformers`` 5 writes the image-processor settings into
``processor_config.json`` while version 4 expects ``preprocessor_config.json``,
so loading through the processor only works on the version that saved the
model. Building ``ViTImageProcessor`` from the JSON directly works on both, and
when neither file is present the settings are rebuilt from ``config.json`` plus
the standard TrOCR values, which produces bit-identical tensors.

``use_cache`` is also re-enabled: the value saved after training is ``false``,
which is correct while training but makes generation several times slower.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import List, Sequence

from .base import Recognizer

TROCR_IMAGE_DEFAULTS = {
    "do_resize": True,
    "resample": 2,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}


def build_image_processor(model_dir: Path):
    """Rebuild the image processor of an exported TrOCR directory."""
    try:
        from transformers import ViTImageProcessorFast as ImageProcessor
    except ImportError:  # pragma: no cover - very old transformers
        from transformers import ViTImageProcessor as ImageProcessor

    source = next(
        (
            model_dir / name
            for name in ("processor_config.json", "preprocessor_config.json")
            if (model_dir / name).exists()
        ),
        None,
    )

    if source is not None:
        raw = json.loads(source.read_text(encoding="utf-8"))
        settings = {
            k: v
            for k, v in raw.get("image_processor", raw).items()
            if k not in ("image_processor_type", "processor_class")
        }
    else:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        size = config["encoder"]["image_size"]
        settings = {"size": {"height": size, "width": size}, **TROCR_IMAGE_DEFAULTS}

    return ImageProcessor(**settings)


class TrOCRRecognizer(Recognizer):
    name = "TrOCR"

    def load(self) -> None:
        import transformers
        from transformers import AutoTokenizer, VisionEncoderDecoderModel

        transformers.logging.set_verbosity_error()

        model_dir = self.cfg.trocr_ckpt
        if model_dir is None or not Path(model_dir).is_dir():
            raise FileNotFoundError(
                f"trocr_ckpt: {model_dir} is not an exported model directory; "
                "it must contain config.json, model.safetensors and tokenizer.json"
            )
        model_dir = Path(model_dir)

        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"no config.json in {model_dir}; it contains "
                f"{sorted(p.name for p in model_dir.iterdir())[:15]}"
            )
        if not any((model_dir / n).exists() for n in ("tokenizer.json", "vocab.json")):
            raise FileNotFoundError(
                f"no tokenizer in {model_dir}; the tokenizer cannot be rebuilt, "
                "re-export the model directory"
            )

        self.processor = build_image_processor(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
        self.model.config.decoder.use_cache = True
        self.model.generation_config.use_cache = True
        self.model.eval()

        self.device = self.cfg.resolve_device()
        self.model.to(self.device)

    def recognize(self, paths: Sequence[Path]) -> List[str]:
        import torch
        from PIL import Image

        self._ensure_loaded()
        batch = self.cfg.trocr_batch
        out: List[str] = []

        for start in range(0, len(paths), batch):
            chunk = list(paths[start : start + batch])
            images = []
            for path in chunk:
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception:
                    images.append(Image.new("RGB", (512, 64), "white"))

            pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            use_amp = self.device.startswith("cuda")
            with torch.no_grad():
                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        ids = self.model.generate(pixel_values)
                else:
                    ids = self.model.generate(pixel_values)

            out.extend(
                unicodedata.normalize("NFC", t)
                for t in self.tokenizer.batch_decode(ids, skip_special_tokens=True)
            )

        return out
