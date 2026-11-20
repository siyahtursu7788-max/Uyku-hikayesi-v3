"""
Akustik Ses Kalite Analizi ve Güvenli Hayalet Temizliği (Bölüm 6 & 7).
- 4x aşırı örneklemeli gerçek True Peak (dBTP) ölçümü.
- Clipping, RMS, konuşma hızı (WPM), anormal iç sessizlik, cümle sonu kesilme ve
  birleştirme (splice) tıklama/ani seviye değişimi denetimi.
- Enerji kapısı (Downward Expander Gate) ASLA kullanılmaz; cümle sonundaki gerçek konuşma korunur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import scipy.signal as sps
from scipy.ndimage import uniform_filter1d

from ..config.quality_thresholds import AudioThresholds


@dataclass
class AudioAcousticMetrics:
    duration_sec: float
    rms_db: float
    peak_linear: float
    true_peak_dbtp: float
    clipped_samples: int
    wpm: float
    max_internal_silence_sec: float
    abrupt_tail_cut: bool
    leading_silence_sec: float
    trailing_silence_sec: float
    passed: bool
    issues: List[str]
    applied_max_wpm: float = 145.0
    word_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def compute_true_peak_dbtp(signal: np.ndarray, oversample: int = 4) -> float:
    """4x polifaz aşırı örnekleme ile gerçek True Peak (dBTP) ölçer."""
    if len(signal) == 0:
        return -120.0
    sig = np.asarray(signal, dtype=np.float32)
    if oversample > 1 and len(sig) >= 16:
        up = sps.resample_poly(sig, oversample, 1)
        peak = float(np.max(np.abs(up)))
    else:
        peak = float(np.max(np.abs(sig)))
    return round(20.0 * math.log10(peak + 1e-12), 2)


def analyze_audio_acoustics(
    signal: np.ndarray,
    sr: int,
    expected_word_count: int,
    thresholds: AudioThresholds | None = None,
) -> AudioAcousticMetrics:
    """Tek cümle veya paragraf sesinin akustik sağlığını bağımsız olarak denetler."""
    th = thresholds or AudioThresholds()
    w = np.asarray(signal, dtype=np.float32)
    issues: List[str] = []

    if len(w) == 0:
        return AudioAcousticMetrics(
            duration_sec=0.0,
            rms_db=-120.0,
            peak_linear=0.0,
            true_peak_dbtp=-120.0,
            clipped_samples=0,
            wpm=0.0,
            max_internal_silence_sec=0.0,
            abrupt_tail_cut=True,
            leading_silence_sec=0.0,
            trailing_silence_sec=0.0,
            passed=False,
            issues=["Ses sinyali boş (0 örnek)."],
        )

    duration_sec = len(w) / float(sr)
    peak_linear = float(np.max(np.abs(w)))
    clipped_samples = int(np.sum(np.abs(w) >= th.max_peak_linear))
    true_peak_dbtp = compute_true_peak_dbtp(w)

    win = max(1, int(0.025 * sr))
    env = np.sqrt(uniform_filter1d(w ** 2, size=win) + 1e-12)
    active_mask = env > 0.006
    if np.any(active_mask):
        rms_val = float(np.sqrt(np.mean(w[active_mask] ** 2) + 1e-12))
    else:
        rms_val = float(np.sqrt(np.mean(w ** 2) + 1e-12))
    rms_db = round(20.0 * math.log10(rms_val + 1e-12), 2)

    active_idx = np.where(active_mask)[0]
    if len(active_idx) == 0:
        issues.append("Seste aktif konuşma enerjisi bulunamadı (tamamen sessiz).")
        leading_sil = duration_sec
        trailing_sil = duration_sec
        max_int_sil = duration_sec
        abrupt_tail = True
        wpm = 0.0
    else:
        leading_sil = float(active_idx[0] / sr)
        trailing_sil = float((len(w) - 1 - active_idx[-1]) / sr)
        speech_span = max(0.25, (active_idx[-1] - active_idx[0]) / float(sr))
        wpm = round((expected_word_count / speech_span) * 60.0, 1) if expected_word_count > 0 else 0.0

        # İç sessizlik analizi (konuşma başı ile sonu arasında)
        inner = ~active_mask[active_idx[0]:active_idx[-1] + 1]
        diffs = np.diff(np.concatenate(([0], inner.astype(int), [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        max_int_sil = float(max((e - s for s, e in zip(starts, ends)), default=0) / sr)

        # Cümle sonu ani kesilme (abrupt tail cut) tespiti:
        # Son 25 ms'deki RMS, konuşma medyanının %35'inden büyükse ve fade-out yoksa ses ortadan kesilmiştir
        tail_samples = max(1, int(0.025 * sr))
        tail_rms = float(np.sqrt(np.mean(w[-tail_samples:] ** 2) + 1e-12))
        abrupt_tail = (trailing_sil < 0.015) and (tail_rms > rms_val * 0.35)

    if clipped_samples > 0:
        issues.append(f"Clipping tespit edildi ({clipped_samples} örnek >= {th.max_peak_linear}).")
    if rms_db < th.min_rms_db:
        issues.append(f"Ses seviyesi çok düşük ({rms_db:.1f} dB < {th.min_rms_db} dB).")
    if rms_db > th.max_rms_db:
        issues.append(f"Ses seviyesi aşırı yüksek ({rms_db:.1f} dB > {th.max_rms_db} dB).")
    if expected_word_count <= th.short_sentence_max_words:
        applied_max_wpm = round(60.0 / max(0.05, th.short_sentence_min_sec_per_word), 1)
    elif expected_word_count <= th.medium_short_max_words:
        applied_max_wpm = th.medium_short_max_wpm
    else:
        applied_max_wpm = th.max_wpm

    if expected_word_count >= 1 and len(active_idx) > 0 and wpm > 0:
        if expected_word_count <= th.short_sentence_max_words:
            min_dur = max(
                th.short_sentence_min_duration_sec,
                expected_word_count * th.short_sentence_min_sec_per_word,
            )
            if speech_span < min_dur or wpm > applied_max_wpm:
                issues.append(
                    f"Kısa cümle anormal hızlı/kısa ({wpm:.1f} WPM > {applied_max_wpm:.1f} veya süre {speech_span:.2f}s < {min_dur:.2f}s)."
                )
            elif expected_word_count >= 3 and wpm < th.min_wpm:
                issues.append(f"Konuşma hızı anormal yavaş ({wpm:.1f} WPM < {th.min_wpm}).")
        else:
            if wpm < th.min_wpm:
                issues.append(f"Konuşma hızı anormal yavaş ({wpm:.1f} WPM < {th.min_wpm}).")
            elif wpm > applied_max_wpm:
                issues.append(
                    f"Konuşma hızı anormal hızlı ({wpm:.1f} WPM > {applied_max_wpm:.1f}, kelime={expected_word_count})."
                )
    if max_int_sil > th.max_internal_silence_sec:
        issues.append(f"Cümle içinde anormal sessizlik ({max_int_sil:.2f} sn > {th.max_internal_silence_sec} sn).")
    if abrupt_tail:
        issues.append("Ses sonunda ani kesilme (abrupt tail cut) tespit edildi.")

    return AudioAcousticMetrics(
        duration_sec=round(duration_sec, 3),
        rms_db=rms_db,
        peak_linear=round(peak_linear, 4),
        true_peak_dbtp=true_peak_dbtp,
        clipped_samples=clipped_samples,
        wpm=wpm,
        max_internal_silence_sec=round(max_int_sil, 3),
        abrupt_tail_cut=abrupt_tail,
        leading_silence_sec=round(leading_sil, 3),
        trailing_silence_sec=round(trailing_sil, 3),
        passed=len(issues) == 0,
        issues=issues,
        applied_max_wpm=applied_max_wpm,
        word_count=expected_word_count,
    )


def check_splice_boundary(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    sr: int,
    max_click_delta: float = 0.35,
    max_level_jump_db: float = 6.0,
) -> Tuple[bool, List[str]]:
    """
    İki cümle birleştirme noktasında tıklama (click/DC jump) ve ani ses seviyesi değişimini denetler.
    """
    issues: List[str] = []
    if len(seg_a) == 0 or len(seg_b) == 0:
        return True, issues

    click_delta = float(abs(float(seg_a[-1]) - float(seg_b[0])))
    if click_delta > max_click_delta:
        issues.append(f"Birleştirme noktasında tıklama (click) riski: delta={click_delta:.3f}")

    rms_a = float(np.sqrt(np.mean(seg_a ** 2) + 1e-12))
    rms_b = float(np.sqrt(np.mean(seg_b ** 2) + 1e-12))
    db_diff = abs(20.0 * math.log10((rms_a + 1e-9) / (rms_b + 1e-9)))
    if db_diff > max_level_jump_db:
        issues.append(f"Ardışık cümleler arası ani seviye değişimi: {db_diff:.1f} dB")

    return len(issues) == 0, issues


def safe_trim_trailing_ghost(
    seg: np.ndarray,
    sr: int,
    last_word_end_sec: float | None = None,
    min_silence_gap_sec: float = 0.22,
    max_ghost_burst_sec: float = 0.35,
    safety_pad_sec: float = 0.14,
) -> Tuple[np.ndarray, bool]:
    """
    P1-01 DÜZELTMESİ:
    Cümle sonu gerçek konuşmayı ASLA kesmeden yalnızca cümle bittikten sonra
    (>=220ms tam sessizlik ardından) ortaya çıkan kısa hayalet darbeleri ('bı-', tıkırtı) temizler.
    - last_word_end_sec biliniyorsa, o noktadan önce KESİNLİKLE kesim yapmaz.
    - Kesim yapılırsa yumuşak 35ms fade-out uygular.
    Döndürür: (temiz_ses, kesim_uygulandi_mi)
    """
    w = np.asarray(seg, dtype=np.float32)
    if len(w) < int(0.8 * sr):
        return w, False

    win = max(1, int(0.020 * sr))
    env = np.sqrt(uniform_filter1d(w ** 2, size=win) + 1e-12)
    act = np.where(env > 0.005)[0]
    if len(act) == 0:
        return w, False

    # Arama yalnızca bilinen son kelimeden (veya toplam uzunluğun %70'inden) SONRA başlar
    if last_word_end_sec is not None:
        search_start = min(len(w), int((last_word_end_sec + safety_pad_sec) * sr))
    else:
        search_start = max(act[0] + int(0.55 * sr), int(len(w) * 0.70))

    if search_start >= len(w):
        return w, False

    sub_sil = env[search_start:] < 0.0030
    d = np.diff(np.concatenate(([0], sub_sil.astype(int), [0])))
    r_starts = np.where(d == 1)[0]
    r_ends = np.where(d == -1)[0]

    trimmed = False
    for rs, re_ in zip(r_starts, r_ends):
        run_len = (re_ - rs) / float(sr)
        abs_start = search_start + rs
        abs_end = search_start + re_
        rem_active = float(np.sum(env[abs_end:] > 0.005) / float(sr))
        if run_len >= min_silence_gap_sec and 0.0 < rem_active <= max_ghost_burst_sec:
            cut_idx = min(len(w), abs_start + int(0.04 * sr))
            w = w[:cut_idx].copy()
            fo = min(int(0.035 * sr), len(w) // 4)
            if fo > 0:
                w[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
            trimmed = True
            break

    return w, trimmed
