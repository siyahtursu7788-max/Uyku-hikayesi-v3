"""
Bağımsız SRT Altyazı Doğrulayıcısı (Bölüm 12).
Üretilen SRT dosyasını bağımsız olarak okur ve şunları denetler:
- Zaman çakışması (overlap)
- Geçersiz zaman damgası (örn. ms >= 1000 veya bitiş <= başlangıç)
- Maksimum 42 karakter / satır ve en fazla 2 satır
- Çok kısa (<0.8s) veya çok uzun (>9.0s) gösterim süresi
- Siyah ekran (outro) üzerinde hayalet altyazı
- Türkçe UTF-8 karakter bütünlüğü (mojibake tespiti)
- Kaynak senaryoya göre eksik kelime kapsamı
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..audio.text_normalizer import tr_norm_tokens
from ..config.quality_thresholds import SRTThresholds


_TS_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})$")
_MOJIBAKE_MARKERS = ("Ã§", "Ä±", "Ã¼", "ÅŸ", "Ã¶", "ÄŸ", "Ã‡", "Ä°", "Ãœ", "Åž", "Ã–", "Äž", "\ufffd")


@dataclass
class SRTValidationResult:
    passed: bool
    cue_count: int
    overlap_count: int
    invalid_timestamp_count: int
    long_line_count: int
    too_many_lines_count: int
    short_duration_count: int
    long_duration_count: int
    black_screen_violations: int
    encoding_issues: int
    word_coverage_ratio: float
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_srt_file(
    srt_path: Union[str, Path],
    expected_sentences: Optional[List[str]] = None,
    outro_start_sec: Optional[float] = None,
    thresholds: Optional[SRTThresholds] = None,
) -> SRTValidationResult:
    th = thresholds or SRTThresholds()
    p = Path(srt_path)
    if not p.exists():
        return SRTValidationResult(False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, ["SRT dosyası bulunamadı."])

    try:
        raw = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return SRTValidationResult(False, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.0, [f"UTF-8 okuma hatası: {exc}"])

    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw.strip()) if b.strip()]
    issues: List[str] = []

    overlaps = 0
    invalid_ts = 0
    long_lines = 0
    too_many_lines = 0
    short_dur = 0
    long_dur = 0
    black_violations = 0
    encoding_issues = 0
    all_srt_text_parts: List[str] = []

    prev_end = -1.0
    for idx, blk in enumerate(blocks, 1):
        lines = blk.splitlines()
        if len(lines) < 3:
            invalid_ts += 1
            issues.append(f"Blok {idx}: Eksik satır sayısı.")
            continue

        m = _TS_RE.match(lines[1].strip())
        if not m:
            invalid_ts += 1
            issues.append(f"Blok {idx}: Geçersiz zaman damgası formatı '{lines[1]}'.")
            continue

        g = [int(x) for x in m.groups()]
        if g[3] >= 1000 or g[7] >= 1000 or g[1] >= 60 or g[2] >= 60 or g[5] >= 60 or g[6] >= 60:
            invalid_ts += 1
            issues.append(f"Blok {idx}: Zaman damgası aralık dışı.")
            continue

        st = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        en = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        if en <= st:
            invalid_ts += 1
            issues.append(f"Blok {idx}: Bitiş zamanı başlangıçtan küçük/eşit.")
        if prev_end >= 0 and st < prev_end - 1e-4:
            overlaps += 1
        prev_end = en

        dur = en - st
        if dur < th.min_cue_duration_sec - 0.05:
            short_dur += 1
        if dur > th.max_cue_duration_sec + 0.05:
            long_dur += 1

        if outro_start_sec is not None and en > outro_start_sec + 0.05:
            black_violations += 1

        text_lines = lines[2:]
        if len(text_lines) > th.max_lines_per_cue:
            too_many_lines += 1
        for tl in text_lines:
            if len(tl) > th.max_chars_per_line:
                long_lines += 1
            if any(mk in tl for mk in _MOJIBAKE_MARKERS):
                encoding_issues += 1
        all_srt_text_parts.append(" ".join(text_lines))

    coverage = 1.0
    if expected_sentences:
        exp_toks = tr_norm_tokens(" ".join(expected_sentences))
        act_toks = tr_norm_tokens(" ".join(all_srt_text_parts))
        if exp_toks:
            # Sıralı kelime kapsama oranı
            from ..audio.whisper_validator import align_words_dp
            _, _, coverage = align_words_dp(exp_toks, act_toks)
            if coverage < th.min_total_coverage_ratio:
                issues.append(f"SRT kelime kapsama oranı yetersiz ({coverage:.3f} < {th.min_total_coverage_ratio}).")

    if overlaps > 0:
        issues.append(f"{overlaps} adet altyazı zaman çakışması tespit edildi.")
    if long_lines > 0:
        issues.append(f"{long_lines} adet satır {th.max_chars_per_line} karakter sınırını aşıyor.")
    if too_many_lines > 0:
        issues.append(f"{too_many_lines} adet blok {th.max_lines_per_cue} satır sınırını aşıyor.")
    if black_violations > 0:
        issues.append(f"Siyah ekran (outro) üzerinde {black_violations} adet hayalet altyazı var.")
    if encoding_issues > 0:
        issues.append(f"{encoding_issues} adet Türkçe karakter kodlama (mojibake) hatası tespit edildi.")
    if len(blocks) == 0:
        issues.append("SRT dosyası hiç altyazı bloğu içermiyor.")

    return SRTValidationResult(
        passed=len(issues) == 0,
        cue_count=len(blocks),
        overlap_count=overlaps,
        invalid_timestamp_count=invalid_ts,
        long_line_count=long_lines,
        too_many_lines_count=too_many_lines,
        short_duration_count=short_dur,
        long_duration_count=long_dur,
        black_screen_violations=black_violations,
        encoding_issues=encoding_issues,
        word_coverage_ratio=round(coverage, 4),
        issues=issues,
    )
