"""
OpenCV / NumPy Görsel Kararlılık, Hareket, Flicker, Morfing, Bulanıklık ve Döngü Denetçisi (Bölüm 9).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from ..config.quality_thresholds import VideoThresholds


@dataclass
class MotionValidationResult:
    passed: bool
    mean_brightness: float
    max_flicker_delta: float
    mean_motion: float
    max_frame_jump: float
    morphing_drift: float
    sharpness_ratio: float
    loop_seam_delta: float
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_video_frames(
    frames: List[Image.Image],
    source_image: Image.Image,
    thresholds: Optional[VideoThresholds] = None,
) -> MotionValidationResult:
    """
    Üretilen video karelerini OpenCV ve NumPy ile bağımsız olarak ölçer:
    - Parlaklık (gündüz patlaması / uyku güvenliği)
    - Ani parlaklık sıçraması (flicker)
    - Kareler arası hareket (aşırı hareketsizlik veya aşırı hareket)
    - Kaynaktan sapma (morphing / drift)
    - Bulanıklık (Laplacian keskinlik oranı)
    - Döngü başlangıç-bitiş uyumu
    """
    th = thresholds or VideoThresholds()
    if len(frames) < 2:
        return MotionValidationResult(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ["En az 2 video karesi gerekli."])

    arrs = [np.array(im.convert("RGB").resize((512, 288)), dtype=np.float32) for im in frames]
    ref_arr = np.array(source_image.convert("RGB").resize((512, 288)), dtype=np.float32)

    lumalar = [float(np.mean(a)) for a in arrs]
    mean_brightness = float(np.mean(lumalar))
    max_flicker_delta = max(abs(lumalar[i] - lumalar[i - 1]) for i in range(1, len(lumalar)))

    frame_diffs = [float(np.mean(np.abs(arrs[i] - arrs[i - 1]))) for i in range(1, len(arrs))]
    mean_motion = float(np.mean(frame_diffs))
    max_frame_jump = float(np.max(frame_diffs))

    morphing_drift = float(np.mean([float(np.mean(np.abs(a - ref_arr))) for a in arrs[1:]]))
    loop_seam_delta = float(np.mean(np.abs(arrs[-1] - arrs[0])))

    # Baş, 1/4, orta, 3/4, son karelerde Laplacian keskinlik ölçümü
    sample_idx = sorted(set([0, len(arrs) // 4, len(arrs) // 2, (3 * len(arrs)) // 4, len(arrs) - 1]))
    sharp_vals: List[float] = []
    for idx in sample_idx:
        gray = cv2.cvtColor(arrs[idx].astype(np.uint8), cv2.COLOR_RGB2GRAY)
        sharp_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    ref_gray = cv2.cvtColor(ref_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ref_sharp = max(float(cv2.Laplacian(ref_gray, cv2.CV_64F).var()), 1.0)
    sharpness_ratio = min(sharp_vals) / ref_sharp

    issues: List[str] = []
    if max_flicker_delta > th.max_flicker_delta:
        issues.append(f"Ani parlaklık sıçraması / flicker ({max_flicker_delta:.2f} > {th.max_flicker_delta}).")
    if mean_brightness > th.max_brightness_mean:
        issues.append(f"Uyku atmosferi için aşırı parlak ({mean_brightness:.1f} > {th.max_brightness_mean}).")
    if morphing_drift > th.max_morphing_drift:
        issues.append(f"Aşırı şekil bozulması / morphing drift ({morphing_drift:.2f} > {th.max_morphing_drift}).")
    if max_frame_jump > th.max_frame_jump:
        issues.append(f"Kareler arası ani sıçrama ({max_frame_jump:.2f} > {th.max_frame_jump}).")
    if mean_motion < th.min_motion_mean:
        issues.append(f"Aşırı hareketsiz / donuk klip ({mean_motion:.2f} < {th.min_motion_mean}).")
    if mean_motion > th.max_motion_mean:
        issues.append(f"Aşırı hızlı / dengesiz hareket ({mean_motion:.2f} > {th.max_motion_mean}).")
    if sharpness_ratio < th.min_sharpness_ratio:
        issues.append(f"Karelerde bulanıklaşma tespit edildi (oran={sharpness_ratio:.2f} < {th.min_sharpness_ratio}).")

    return MotionValidationResult(
        passed=len(issues) == 0,
        mean_brightness=round(mean_brightness, 2),
        max_flicker_delta=round(max_flicker_delta, 2),
        mean_motion=round(mean_motion, 3),
        max_frame_jump=round(max_frame_jump, 2),
        morphing_drift=round(morphing_drift, 2),
        sharpness_ratio=round(sharpness_ratio, 3),
        loop_seam_delta=round(loop_seam_delta, 2),
        issues=issues,
    )


def audit_extra_frame_anomalies(
    frames: List[Image.Image],
    black_luma_threshold: float = 2.0,
    blank_std_threshold: float = 1.5,
    repeat_diff_threshold: float = 0.01,
    max_repeat_ratio: float = 0.85,
) -> Dict[str, Any]:
    """OpenCV/NumPy ek kalite kontrolü: siyah kare, boş/tek renk kare ve aşırı kare tekrarı."""
    if not frames:
        return {
            "passed": False,
            "black_frames": 0,
            "blank_frames": 0,
            "repeated_pairs": 0,
            "repeat_ratio": 1.0,
            "issues": ["Klip boş (0 kare)."],
        }
    grays = [np.asarray(f.convert("L").resize((256, 144)), dtype=np.float32) for f in frames]
    black_cnt = sum(1 for g in grays if float(np.mean(g)) < black_luma_threshold)
    blank_cnt = sum(1 for g in grays if float(np.std(g)) < blank_std_threshold)
    rep_pairs = 0
    for i in range(len(grays) - 1):
        if float(np.mean(np.abs(grays[i + 1] - grays[i]))) < repeat_diff_threshold:
            rep_pairs += 1
    rep_ratio = rep_pairs / max(1, len(grays) - 1)
    issues: List[str] = []
    if black_cnt > 0:
        issues.append(f"Siyah kare tespit edildi: {black_cnt}")
    if blank_cnt > 0:
        issues.append(f"Boş/düz kare tespit edildi: {blank_cnt}")
    if len(grays) >= 4 and rep_ratio > max_repeat_ratio:
        issues.append(f"Aşırı kare tekrarı tespit edildi: {rep_ratio:.2f} > {max_repeat_ratio:.2f}")
    return {
        "passed": len(issues) == 0,
        "black_frames": black_cnt,
        "blank_frames": blank_cnt,
        "repeated_pairs": rep_pairs,
        "repeat_ratio": round(rep_ratio, 4),
        "issues": issues,
    }
