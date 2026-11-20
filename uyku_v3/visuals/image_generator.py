"""
Görsel Üretim Motoru (Bölüm 8): SDXL 1.0 (T4 GPU optimize) + isteğe bağlı Imagen 3.
Mevcut görselleri gereksiz yere yeniden üretmez; yalnızca eksik veya doğrulamadan
geçemeyen (bozuk/yanlış boyutlu/uygunsuz) görselleri yeniden üretim kuyruğuna alır.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from PIL import Image

from ..config.model_config import SDXLConfig
from ..config.quality_thresholds import ImageThresholds
from ..core.file_manager import atomic_copy_file
from .image_validator import ImageValidationResult, validate_image_file


class ImageGenerator:
    def __init__(
        self,
        config: Optional[SDXLConfig] = None,
        thresholds: Optional[ImageThresholds] = None,
        generator_fn: Optional[Callable[[str, str], Image.Image]] = None,
    ) -> None:
        self.config = config or SDXLConfig()
        self.thresholds = thresholds or ImageThresholds()
        self._generator_fn = generator_fn
        self._pipe: Any = None

    def load_sdxl(self, device: str = "cuda") -> None:
        if self._generator_fn is not None or self._pipe is not None:
            return
        if device != "cuda":
            raise RuntimeError("SDXL 1.0 üretimi için CUDA/T4 GPU gereklidir; CPU üzerinde sessiz çalıştırma engellendi.")
        import torch  # type: ignore
        from diffusers import StableDiffusionXLPipeline  # type: ignore

        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            variant=self.config.variant,
            use_safetensors=True,
        ).to(device)
        self._pipe.enable_attention_slicing()
        self._pipe.vae.enable_tiling()

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        gc.collect()
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def ensure_valid_scene_image(
        self,
        target_path: Path,
        prompt: str,
        local_tmp_dir: Path,
        max_retries: int = 2,
    ) -> Tuple[bool, ImageValidationResult, bool]:
        """
        Mevcut görsel geçerliyse korur (regenerated=False).
        Eksik veya geçersizse yerel SSD'de üretir, 1920x1080'e ölçekler, doğrular ve
        atomik olarak hedefe kopyalar.
        Döndürür: (passed, validation_result, regenerated)
        """
        target = Path(target_path)
        if target.exists():
            existing_val = validate_image_file(target, self.thresholds, require_exact_1080p=True)
            if existing_val.passed:
                return True, existing_val, False

        local_tmp_dir.mkdir(parents=True, exist_ok=True)
        last_val = ImageValidationResult(str(target), False, 0, 0, 0.0, 0, 0.0, 0.0, ["Üretim başlatılamadı."])

        for attempt in range(1, max_retries + 1):
            if self._generator_fn is not None:
                img = self._generator_fn(prompt, self.config.negative_prompt)
            else:
                if self._pipe is None:
                    raise RuntimeError("SDXL pipeline yüklenmeden görsel üretilemez.")
                img = self._pipe(
                    prompt=prompt,
                    negative_prompt=self.config.negative_prompt,
                    width=self.config.gen_width,
                    height=self.config.gen_height,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                ).images[0]

            img_1080 = img.convert("RGB").resize(
                (self.config.final_width, self.config.final_height),
                Image.Resampling.LANCZOS,
            )
            tmp_out = local_tmp_dir / f"gen_{target.stem}_a{attempt}.jpg"
            img_1080.save(tmp_out, quality=95)
            val = validate_image_file(tmp_out, self.thresholds, require_exact_1080p=True)
            last_val = val
            if val.passed:
                atomic_copy_file(tmp_out, target, verify_hash=True)
                try:
                    tmp_out.unlink()
                except OSError:
                    pass
                return True, val, True

        return False, last_val, True
