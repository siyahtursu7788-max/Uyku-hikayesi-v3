"""
Sahne Görseli Doğrulama Modülü (Bölüm 8).
P1-04 DÜZELTMESİ: Görselin yalnızca var olmasına veya dosya boyutuna bakmaz;
bozuk dosya, yanlış en-boy oranı (16:9), yanlış çözünürlük (1920x1080),
uyku atmosferini bozan aşırı parlaklık ve bulanıklık kontrollerini çalıştırır.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from ..config.quality_thresholds import ImageThresholds


@dataclass
class ImageValidationResult:
    path: str
    passed: bool
    width: int
    height: int
    aspect_ratio: float
    file_size_bytes: int
    mean_brightness: float
    sharpness_laplacian: float
    issues: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def validate_image_file(
    image_path: Union[str, Path],
    thresholds: Optional[ImageThresholds] = None,
    require_exact_1080p: bool = True,
) -> ImageValidationResult:
    th = thresholds or ImageThresholds()
    p = Path(image_path)
    issues: List[str] = []

    if not p.exists():
        return ImageValidationResult(str(p), False, 0, 0, 0.0, 0, 0.0, 0.0, ["Görsel dosyası bulunamadı."])

    size = p.stat().st_size
    if size < th.min_file_size_bytes:
        issues.append(f"Dosya boyutu çok küçük ({size} byte < {th.min_file_size_bytes} byte).")

    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im2:
            rgb = im2.convert("RGB")
            w, h = rgb.size
            arr = np.array(rgb, dtype=np.uint8)
    except Exception as exc:
        return ImageValidationResult(str(p), False, 0, 0, 0.0, size, 0.0, 0.0, [f"Görsel bozuk veya açılamıyor: {exc}"])

    ar = w / float(max(1, h))
    if abs(ar - th.expected_aspect_ratio) > th.aspect_ratio_tolerance:
        issues.append(f"Yanlış en-boy oranı: {w}x{h} ({ar:.3f} != 16:9={th.expected_aspect_ratio:.3f}).")

    if require_exact_1080p and (w != th.expected_width or h != th.expected_height):
        issues.append(f"Beklenen 1920x1080 çözünürlüğü sağlanmadı: {w}x{h}.")

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness > th.max_mean_brightness:
        issues.append(f"Uyku atmosferi için aşırı parlak ({brightness:.1f} > {th.max_mean_brightness}).")
    if brightness < th.min_mean_brightness:
        issues.append(f"Görsel neredeyse tamamen siyah/boş ({brightness:.1f} < {th.min_mean_brightness}).")
    if sharpness < th.min_sharpness_laplacian:
        issues.append(f"Görsel aşırı bulanık (Laplacian={sharpness:.1f} < {th.min_sharpness_laplacian}).")

    return ImageValidationResult(
        path=str(p),
        passed=len(issues) == 0,
        width=w,
        height=h,
        aspect_ratio=round(ar, 4),
        file_size_bytes=size,
        mean_brightness=round(brightness, 2),
        sharpness_laplacian=round(sharpness, 2),
        issues=issues,
    )
