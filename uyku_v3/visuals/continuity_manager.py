"""Bölüm 2.4 — Görsel devamlılık, karakter/mekân/ışık rehberi ve sahne prompt yöneticisi."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from ..core.file_manager import sha256_dict


@dataclass
class CharacterProfile:
    name: str
    age_appearance: str
    hair: str
    clothing: str
    face_notes: str
    reference_image_path: Optional[str] = None


@dataclass
class VisualStyleBible:
    lighting: str = "low-key moody midnight navy and soft muted warm amber lantern glow"
    color_palette: str = "deep midnight navy, dark walnut brown, muted antique brass, soft silver moonlight"
    atmosphere: str = "tranquil dreamlike nocturnal solitude, sleep storytelling mood"
    camera_lens: str = "Arri Alexa 35mm anamorphic lens, shallow depth of field, subtle film grain"
    negative_prompt: str = (
        "bright daylight, harsh sun, neon, oversaturated, modern plastic, text, watermark, "
        "logo, signature, blurry, deformed, cartoon, anime, CGI look"
    )
    characters: Dict[str, CharacterProfile] = field(default_factory=dict)
    key_objects: Dict[str, str] = field(
        default_factory=lambda: {
            "pirinc_anahtar": "antique smooth brass key with delicate leaf carvings on the handle",
            "ceviz_kapi": "towering carved walnut wood door without knocker or lock, leaf and resting bird reliefs",
            "cam_sise": "round antique clear glass memory bottle with soft pale mist glow and handwritten cream paper label",
        }
    )


class VisualContinuityManager:
    """Sahneler arası karakter, mekân, nesne, ışık ve renk paleti tutarlılığını yönetir."""

    def __init__(self, style_bible: Optional[VisualStyleBible] = None) -> None:
        self.style_bible = style_bible or VisualStyleBible()

    def build_scene_prompt(
        self,
        scene_id: str,
        location_desc: str,
        characters_present: Optional[List[str]] = None,
        objects_present: Optional[List[str]] = None,
        allow_characters: bool = False,
    ) -> Dict[str, Any]:
        parts = [location_desc.strip()]
        for obj_k in objects_present or []:
            if obj_k in self.style_bible.key_objects:
                parts.append(self.style_bible.key_objects[obj_k])

        if allow_characters and characters_present:
            for cname in characters_present:
                prof = self.style_bible.characters.get(cname)
                if prof:
                    parts.append(
                        f"{prof.name} ({prof.age_appearance}, {prof.hair}, wearing {prof.clothing}, {prof.face_notes})"
                    )
        else:
            parts.append("first-person atmospheric perspective, no people, no faces")

        parts.extend([
            self.style_bible.lighting,
            self.style_bible.color_palette,
            self.style_bible.atmosphere,
            self.style_bible.camera_lens,
        ])
        positive = ", ".join(p for p in parts if p)
        negative = self.style_bible.negative_prompt
        if not allow_characters:
            negative += ", human, person, face, hands, silhouette of person"

        params = {
            "scene_id": scene_id,
            "positive_prompt": positive,
            "negative_prompt": negative,
            "style_hash": sha256_dict(asdict(self.style_bible)),
        }
        return params

    @staticmethod
    def evaluate_cross_scene_continuity(
        image_paths: List[Path],
        max_mean_luma_drift: float = 45.0,
        max_color_channel_drift: float = 42.0,
    ) -> Dict[str, Any]:
        """Ardışık sahne görselleri arasında ışık ve renk paleti tutarlılığını ölçer."""
        if len(image_paths) < 2:
            return {"passed": True, "pairs_checked": 0, "max_luma_drift": 0.0, "max_rgb_drift": 0.0, "issues": []}

        stats = []
        for p in image_paths:
            arr = np.asarray(Image.open(p).convert("RGB").resize((256, 144)), dtype=np.float32)
            luma = float(np.mean(0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]))
            rgb = [float(np.mean(arr[:, :, c])) for c in range(3)]
            stats.append((p.name, luma, rgb))

        max_luma = 0.0
        max_rgb = 0.0
        issues: List[str] = []
        for i in range(len(stats) - 1):
            n1, l1, r1 = stats[i]
            n2, l2, r2 = stats[i + 1]
            dl = abs(l2 - l1)
            dr = float(np.linalg.norm(np.array(r2) - np.array(r1)))
            max_luma = max(max_luma, dl)
            max_rgb = max(max_rgb, dr)
            if dl > max_mean_luma_drift:
                issues.append(f"Işık devamlılığı kırıldı ({n1}->{n2}: Δluma={dl:.1f} > {max_mean_luma_drift})")
            if dr > max_color_channel_drift:
                issues.append(f"Renk paleti sapması ({n1}->{n2}: Δrgb={dr:.1f} > {max_color_channel_drift})")

        return {
            "passed": len(issues) == 0,
            "pairs_checked": len(stats) - 1,
            "max_luma_drift": round(max_luma, 2),
            "max_rgb_drift": round(max_rgb, 2),
            "issues": issues,
        }
