"""
Hatasız Türkçe SRT Üretim Motoru (Bölüm 12).
P0-04 DÜZELTMESİ:
- `ms == 1000` taşma hatası tamamen giderildi (toplam milisaniye tamsayı aritmetiği).
- Maksimum 42 karakter x 2 satır kuralı uygulanırken HİÇBİR KELİME KIRPILMAZ:
  2 satırı aşan uzun cümleler, kelime sayısına orantılı zaman dağılımıyla ardışık
  altyazı bloklarına (cue) bölünür.
- Ardışık altyazılar arasında sıfır çakışma (overlap clamping) garanti edilir.
- Siyah ekran (outro) başlangıcından sonra tek bir hayalet altyazı üretilmez.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from ..config.quality_thresholds import SRTThresholds
from ..core.file_manager import atomic_write_text
from ..core.state_manager import ArtifactStatus
from .alignment import AlignedSentence


@dataclass
class SRTCue:
    index: int
    start_sec: float
    end_sec: float
    lines: List[str]


def format_srt_timestamp(seconds: float) -> str:
    """Milisaniye taşması (ms=1000) olmadan kesin SRT zaman damgası (HH:MM:SS,mmm)."""
    total_ms = max(0, int(round(seconds * 1000.0)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap_and_split_text(
    text: str,
    max_chars_per_line: int = 42,
    max_lines_per_cue: int = 2,
) -> List[List[str]]:
    """
    Metni en fazla `max_chars_per_line` karakterlik satırlara sarar ve
    `max_lines_per_cue` (2) satırı aşan kısımları HİÇBİR KELİMEYİ SİLMEDEN
    ardışık altyazı bloklarına böler.
    """
    words = text.strip().split()
    if not words:
        return []

    lines: List[str] = []
    cur = ""
    for w in words:
        # Tek kelime 42 karakterden uzunsa güvenli böl
        if len(w) > max_chars_per_line:
            if cur:
                lines.append(cur)
                cur = ""
            for i in range(0, len(w), max_chars_per_line):
                lines.append(w[i:i + max_chars_per_line])
            continue
        cand = f"{cur} {w}".strip() if cur else w
        if len(cand) <= max_chars_per_line:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    cues_lines: List[List[str]] = []
    for i in range(0, len(lines), max_lines_per_cue):
        cues_lines.append(lines[i:i + max_lines_per_cue])
    return cues_lines


def generate_srt_cues(
    aligned_sentences: List[AlignedSentence],
    outro_start_sec: Optional[float] = None,
    thresholds: Optional[SRTThresholds] = None,
) -> List[SRTCue]:
    """
    Onaylı hizalanmış cümlelerden sıfır-çakışmalı, 42-karakter x 2-satır uyumlu,
    okunabilir SRT blokları üretir.
    """
    th = thresholds or SRTThresholds()
    raw_cues: List[Tuple[float, float, List[str]]] = []

    for item in aligned_sentences:
        if item.status != ArtifactStatus.APPROVED or item.start_sec is None or item.end_sec is None:
            continue
        t_start = item.start_sec
        t_end = item.end_sec + 0.15
        if outro_start_sec is not None:
            if t_start >= outro_start_sec:
                continue
            t_end = min(t_end, outro_start_sec)
        if t_end <= t_start:
            continue

        blocks = wrap_and_split_text(item.text, th.max_chars_per_line, th.max_lines_per_cue)
        if not blocks:
            continue

        total_chars = sum(sum(len(l) for l in blk) for blk in blocks)
        span = max(th.min_cue_duration_sec * len(blocks), t_end - t_start)
        if outro_start_sec is not None:
            span = min(span, max(0.2, outro_start_sec - t_start))

        cursor = t_start
        for b_idx, blk in enumerate(blocks):
            blk_chars = sum(len(l) for l in blk)
            frac = blk_chars / float(max(1, total_chars))
            dur = max(th.min_cue_duration_sec, min(th.max_cue_duration_sec, span * frac))
            c_end = cursor + dur if b_idx < len(blocks) - 1 else min(t_start + span, cursor + th.max_cue_duration_sec)
            if outro_start_sec is not None:
                c_end = min(c_end, outro_start_sec)
            if c_end > cursor + 0.05:
                raw_cues.append((cursor, c_end, blk))
            cursor = c_end

    # Çakışma giderme (overlap clamping) ve sıralama
    raw_cues.sort(key=lambda x: x[0])
    final_cues: List[SRTCue] = []
    for i, (st, en, lines) in enumerate(raw_cues):
        if i + 1 < len(raw_cues):
            next_st = raw_cues[i + 1][0]
            if en > next_st - 0.02:
                en = max(st + 0.25, next_st - 0.02)
        if final_cues and st < final_cues[-1].end_sec:
            st = final_cues[-1].end_sec + 0.01
            if en <= st:
                en = st + 0.25
        if outro_start_sec is not None:
            if st >= outro_start_sec:
                continue
            en = min(en, outro_start_sec)
        if en > st + 0.05:
            final_cues.append(SRTCue(index=len(final_cues) + 1, start_sec=round(st, 3), end_sec=round(en, 3), lines=lines))

    return final_cues


def render_srt_text(cues: List[SRTCue]) -> str:
    blocks: List[str] = []
    for c in cues:
        ts = f"{format_srt_timestamp(c.start_sec)} --> {format_srt_timestamp(c.end_sec)}"
        body = "\n".join(c.lines)
        blocks.append(f"{c.index}\n{ts}\n{body}\n")
    return "\n".join(blocks)


def write_srt_file(cues: List[SRTCue], output_path: Path) -> str:
    content = render_srt_text(cues)
    atomic_write_text(output_path, content)
    return content
