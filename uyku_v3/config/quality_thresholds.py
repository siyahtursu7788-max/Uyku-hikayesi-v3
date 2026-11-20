"""
Tüm aşamalar için katı kalite kontrol eşikleri.
Hiçbir aşama bu eşikleri geçmeden APPROVED durumuna geçemez.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioThresholds:
    # Çoklu metin doğrulama eşikleri (SequenceMatcher tek başına karar vermez)
    max_wer: float = 0.12
    max_cer: float = 0.08
    min_word_alignment_ratio: float = 0.90
    min_last_word_similarity: float = 0.78
    min_second_last_word_similarity: float = 0.74
    max_missing_words: int = 0
    max_extra_words: int = 1
    # Akustik ve süre eşikleri
    min_wpm: float = 42.0
    max_wpm: float = 145.0
    # Kısa cümle WPM güvenilirlik eşikleri (uzun cümlelerde max_wpm=145.0 korunur)
    short_sentence_max_words: int = 3
    short_sentence_min_sec_per_word: float = 0.25
    short_sentence_min_duration_sec: float = 0.60
    medium_short_max_words: int = 6
    medium_short_max_wpm: float = 180.0
    max_peak_linear: float = 0.985  # Clipping eşiği
    max_true_peak_dbtp: float = -2.5
    min_rms_db: float = -34.0
    max_rms_db: float = -10.0
    max_internal_silence_sec: float = 1.6
    max_splice_click_delta: float = 0.35
    min_tail_energy_ratio: float = 0.02
    # Paragraf tamlık eşiği
    min_paragraph_word_coverage: float = 0.92


@dataclass(frozen=True)
class ImageThresholds:
    expected_width: int = 1920
    expected_height: int = 1080
    expected_aspect_ratio: float = 16.0 / 9.0
    aspect_ratio_tolerance: float = 0.02
    min_file_size_bytes: int = 15_000
    max_mean_brightness: float = 145.0  # Uyku atmosferi için gündüz patlaması engeli
    min_mean_brightness: float = 4.0    # Tamamen siyah bozuk görsel engeli
    min_sharpness_laplacian: float = 15.0


@dataclass(frozen=True)
class VideoThresholds:
    # CLIP bağımsız kosinüs benzerliği (softmax olasılığı DEĞİL!)
    min_clip_cosine_positive: float = 0.22
    max_clip_cosine_negative: float = 0.24
    min_clip_margin: float = 0.03  # positive_cos - max_negative_cos >= 0.03
    # OpenCV / NumPy görsel kararlılık eşikleri
    max_brightness_mean: float = 140.0
    max_flicker_delta: float = 9.5
    max_morphing_drift: float = 26.0
    max_frame_jump: float = 14.0
    min_motion_mean: float = 0.35
    max_motion_mean: float = 12.0
    min_sharpness_ratio: float = 0.45
    max_loop_seam_delta: float = 12.0


@dataclass(frozen=True)
class AssemblyThresholds:
    expected_width: int = 1920
    expected_height: int = 1080
    expected_fps: float = 24.0
    fps_tolerance: float = 0.2
    max_av_sync_diff_sec: float = 0.35
    max_black_pixel_mean: float = 1.0  # #000000 saf siyah doğrulaması
    outro_duration_tolerance_sec: float = 1.5


@dataclass(frozen=True)
class SRTThresholds:
    max_chars_per_line: int = 42
    max_lines_per_cue: int = 2
    min_cue_duration_sec: float = 0.8
    max_cue_duration_sec: float = 9.0
    min_gap_between_cues_sec: float = 0.0
    min_sentence_alignment_confidence: float = 0.72
    min_total_coverage_ratio: float = 0.92
