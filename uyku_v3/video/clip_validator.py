"""
CLIP Metin-Görsel Uyum ve Negatif Risk Denetçisi (Bölüm 9).
P1-02 DÜZELTMESİ:
- CLIP softmax skorunu gerçek olasılık veya kesin kalite ölçüsü olarak KULLANMAZ!
- Her metin kavramı için bağımsız normalize edilmiş kosinüs benzerliği (cosine similarity)
  hesaplar ve pozitif uyum ile en yüksek negatif kavram arasındaki güvenlik marjını (margin) ölçer.
- Başlangıç, 1/4, orta, 3/4 ve son kareleri (en az 5 kare) denetler.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..config.quality_thresholds import VideoThresholds


DEFAULT_NEGATIVE_CONCEPTS: List[str] = [
    "bright sunny daylight scene",
    "human face portrait or person figure",
    "glitchy distorted melted morphing artifacts",
    "text overlay watermark logo signature",
]


@dataclass
class CLIPValidationResult:
    passed: bool
    positive_cosine_mean: float
    positive_cosine_min: float
    max_negative_cosine: float
    worst_negative_concept: str
    margin: float
    sampled_frame_indices: List[int]
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_audit_frame_indices(num_frames: int) -> List[int]:
    """Başlangıç, 1/4, orta, 3/4 ve son kare indekslerini seçer."""
    if num_frames <= 0:
        return []
    if num_frames <= 5:
        return list(range(num_frames))
    raw = [0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames - 1]
    return sorted(set(raw))


class CLIPValidator:
    """
    Bağımsız CLIP kosinüs benzerliği denetçisi.
    Testlerde score_fn(frames, texts) -> np.ndarray [N_frames, N_texts] (cosine similarities)
    enjekte edilerek GPU olmadan deterministik test yapılabilir.
    """

    def __init__(
        self,
        thresholds: Optional[VideoThresholds] = None,
        score_fn: Optional[Callable[[List[Image.Image], List[str]], np.ndarray]] = None,
        negative_concepts: Optional[List[str]] = None,
    ) -> None:
        self.thresholds = thresholds or VideoThresholds()
        self._score_fn = score_fn
        self.negative_concepts = negative_concepts or list(DEFAULT_NEGATIVE_CONCEPTS)
        self._model: Any = None
        self._proc: Any = None
        self._device: str = "cuda"

    def load(self, device: str = "cuda") -> None:
        if self._score_fn is not None:
            return
        self._device = device
        from transformers import CLIPModel, CLIPProcessor  # type: ignore

        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self._proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._model.eval()

    def _compute_cosine_matrix(self, frames: List[Image.Image], texts: List[str]) -> np.ndarray:
        if self._score_fn is not None:
            return np.asarray(self._score_fn(frames, texts), dtype=np.float32)
        if self._model is None or self._proc is None:
            raise RuntimeError("CLIP modeli yüklenmedi.")
        import torch  # type: ignore

        inputs = self._proc(text=texts, images=frames, return_tensors="pt", padding=True).to(self._device)
        def _as_tensor(out: Any) -> Any:
            if hasattr(out, "norm"):
                return out
            for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
                val = getattr(out, attr, None)
                if val is not None and hasattr(val, "norm"):
                    return val
            if isinstance(out, (tuple, list)) and out and hasattr(out[0], "norm"):
                return out[0]
            raise TypeError(f"Beklenmeyen CLIP çıktı türü: {type(out).__name__}")

        with torch.no_grad():
            img_raw = self._model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_raw = self._model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            img_feats = _as_tensor(img_raw)
            txt_feats = _as_tensor(txt_raw)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            cos_sim = (img_feats @ txt_feats.T).cpu().numpy()
        return cos_sim.astype(np.float32)

    def validate_frames(self, frames: List[Image.Image], positive_query: str) -> CLIPValidationResult:
        th = self.thresholds
        indices = select_audit_frame_indices(len(frames))
        if not indices:
            return CLIPValidationResult(False, 0.0, 0.0, 1.0, "empty", -1.0, [], ["Denetlenecek video karesi yok."])

        sampled = [frames[i] for i in indices]
        texts = [positive_query] + self.negative_concepts
        cos_mat = self._compute_cosine_matrix(sampled, texts)  # [N_frames, 1 + N_neg]

        pos_scores = cos_mat[:, 0]
        neg_scores = cos_mat[:, 1:]

        pos_mean = float(np.mean(pos_scores))
        pos_min = float(np.min(pos_scores))
        max_neg_idx = int(np.unravel_index(int(np.argmax(neg_scores)), neg_scores.shape)[1])
        max_neg = float(np.max(neg_scores))
        worst_neg = self.negative_concepts[max_neg_idx] if self.negative_concepts else ""
        margin = pos_mean - max_neg

        issues: List[str] = []
        if pos_mean < th.min_clip_cosine_positive:
            issues.append(f"CLIP metin-görsel kosinüs uyumu düşük ({pos_mean:.3f} < {th.min_clip_cosine_positive}).")
        if max_neg > th.max_clip_cosine_negative:
            issues.append(f"CLIP negatif kavram riski yüksek: '{worst_neg}' ({max_neg:.3f} > {th.max_clip_cosine_negative}).")
        if margin < th.min_clip_margin:
            issues.append(f"CLIP pozitif-negatif güvenlik marjı yetersiz ({margin:.3f} < {th.min_clip_margin}).")

        return CLIPValidationResult(
            passed=len(issues) == 0,
            positive_cosine_mean=round(pos_mean, 4),
            positive_cosine_min=round(pos_min, 4),
            max_negative_cosine=round(max_neg, 4),
            worst_negative_concept=worst_neg,
            margin=round(margin, 4),
            sampled_frame_indices=indices,
            issues=issues,
        )
