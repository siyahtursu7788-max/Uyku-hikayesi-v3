"""
Sahne Yönetimi ve Hareket Profili Eşleştirici (Bölüm 8, 9, 10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..config.model_config import MOTION_PROFILES, MotionProfile


@dataclass
class SceneSpec:
    scene_index: int
    image_filename: str
    start_paragraph: str
    end_paragraph: str
    motion_profile_name: str
    clip_query: str
    sdxl_prompt: str

    @property
    def motion_profile(self) -> MotionProfile:
        return MOTION_PROFILES.get(self.motion_profile_name, MOTION_PROFILES["indoor_calm"])


class SceneManager:
    def __init__(self, scenes: List[SceneSpec]) -> None:
        self.scenes = scenes

    def resolve_scene_windows(
        self,
        paragraph_timestamps: Dict[str, float],
        outro_start_sec: float,
    ) -> List[Tuple[SceneSpec, float, float]]:
        """Her sahnenin (baslangic_sn, bitis_sn) penceresini paragraf zaman damgalarından hesaplar."""
        windows: List[Tuple[SceneSpec, float, float]] = []
        total_p = max(1, len(paragraph_timestamps))

        def _lookup(code: str) -> float:
            if code == "OUTRO":
                return outro_start_sec
            if code in paragraph_timestamps:
                return float(paragraph_timestamps[code])
            idx = int(code.replace("P", ""))
            return (idx / float(total_p)) * outro_start_sec

        for idx, sc in enumerate(self.scenes):
            t0 = 0.0 if idx == 0 else _lookup(sc.start_paragraph)
            t1 = _lookup(sc.end_paragraph)
            if t1 <= t0:
                raise ValueError(f"Sahne {sc.scene_index:02d} zaman aralığı geçersiz: {t0:.2f} -> {t1:.2f}")
            windows.append((sc, round(t0, 3), round(t1, 3)))
        return windows
