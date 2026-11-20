"""
Zaman Çizelgesi ve YouTube Bölüm (Chapter) Yöneticisi (Bölüm 11).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


@dataclass
class TimelineSegment:
    scene_index: int
    clip_path: str
    start_sec: float
    end_sec: float
    is_black_outro: bool = False

    @property
    def duration_sec(self) -> float:
        return round(max(0.0, self.end_sec - self.start_sec), 3)

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["duration_sec"] = self.duration_sec
        return d


def build_scene_timeline(
    scene_windows: List[Tuple[int, str, float, float]],
    total_audio_sec: float,
    outro_start_sec: float,
    black_outro_path: str,
) -> List[TimelineSegment]:
    """
    Sahne pencerelerinin boşluksuz ve çakışmasız sıralandığını doğrular,
    sonuna saf siyah ekran outro segmentini ekler.
    """
    segments: List[TimelineSegment] = []
    prev_end = 0.0
    for idx, clip_path, t0, t1 in scene_windows:
        if t0 < prev_end - 0.05:
            raise ValueError(f"Sahne {idx:02d} önceki sahneyle çakışıyor: {t0:.2f} < {prev_end:.2f}")
        if t1 <= t0:
            raise ValueError(f"Sahne {idx:02d} süresi geçersiz: {t0:.2f} -> {t1:.2f}")
        segments.append(TimelineSegment(scene_index=idx, clip_path=clip_path, start_sec=t0, end_sec=t1))
        prev_end = t1

    outro_dur = max(0.0, total_audio_sec - outro_start_sec)
    if outro_dur > 0.5:
        segments.append(
            TimelineSegment(
                scene_index=99,
                clip_path=black_outro_path,
                start_sec=outro_start_sec,
                end_sec=total_audio_sec,
                is_black_outro=True,
            )
        )
    return segments


def format_hms(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def format_youtube_chapters(title: str, chapters: List[Tuple[str, float]]) -> str:
    """YouTube 00:00:00 kuralına uygun bölüm zaman damgası metni üretir."""
    lines = [title, ""]
    for i, (label, ts) in enumerate(chapters):
        t = 0.0 if i == 0 else ts
        lines.append(f"{format_hms(t)} {label}")
    return "\n".join(lines) + "\n"
