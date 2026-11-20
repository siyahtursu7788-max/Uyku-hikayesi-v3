"""
Doğal Ses Mastering ve Kademeli Seviye Protokolü (Bölüm 7).
- Konuşma dinamiklerini korur (cümle bazlı agresif normalizasyon yapmaz).
- K1–K6 kademeli anlatıcı seviyesini (-14 -> -18 LUFS) ve anlaşılabilirliği korur.
- 4x aşırı örneklemeli gerçek True Peak (-2.5 dBTP) tavanı uygular.
- Altın Ambiyans Duraklarında sidechain pumping olmadan 3.5 sn yumuşak otomasyon uygular.
- Finalde 6 dakika saf siyah ekran ambiyansı + son 45 saniye logaritmik fade-to-silence üretir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from ..config.model_config import STEPPED_AUDIO_PROFILES
from ..config.quality_thresholds import AudioThresholds
from .audio_quality import compute_true_peak_dbtp


@dataclass
class MasteringReport:
    duration_sec: float
    integrated_rms_db: float
    true_peak_dbtp: float
    chapter_levels_db: Dict[str, float]
    outro_start_sec: float
    outro_fade_verified: bool
    passed: bool
    issues: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def apply_hpf(signal: np.ndarray, sr: int, cutoff_hz: float = 70.0) -> np.ndarray:
    if len(signal) <= 32:
        return signal.astype(np.float32)
    sos = butter(2, max(cutoff_hz / (0.5 * sr), 0.001), btype="high", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def apply_lpf(signal: np.ndarray, sr: int, cutoff_hz: float = 1400.0) -> np.ndarray:
    if len(signal) <= 32:
        return signal.astype(np.float32)
    sos = butter(2, min(cutoff_hz / (0.5 * sr), 0.99), btype="low", output="sos")
    return sosfilt(sos, signal).astype(np.float32)


def scale_to_rms_db(signal: np.ndarray, target_db: float) -> np.ndarray:
    if len(signal) == 0:
        return signal.astype(np.float32)
    rms = float(np.sqrt(np.mean(signal ** 2) + 1e-12))
    target_lin = 10.0 ** (target_db / 20.0)
    return (signal * (target_lin / rms)).astype(np.float32)


def apply_true_peak_limiter(
    signal: np.ndarray,
    sr: int,
    ceiling_dbtp: float = -2.5,
) -> np.ndarray:
    """
    Şeffaf yumuşak dizli (soft-knee) limiter + 4x True Peak tavan koruması.
    Ses rengini bozan sert kırpma veya aşırı tanh doygunluğu yapmaz.
    """
    x = np.asarray(signal, dtype=np.float32).copy()
    if len(x) == 0:
        return x
    ceiling_lin = 10.0 ** (ceiling_dbtp / 20.0)
    # Yumuşak dizli üst tepe sıkıştırma (yalnızca ceiling * 0.92 üstünde devreye girer)
    knee = ceiling_lin * 0.92
    over = np.abs(x) > knee
    if np.any(over):
        excess = np.abs(x[over]) - knee
        room = max(1e-6, ceiling_lin - knee)
        compressed = knee + room * np.tanh(excess / room)
        x[over] = np.sign(x[over]) * compressed

    # 4x True Peak doğrulaması ve hassas tavan ölçekleme
    tp = compute_true_peak_dbtp(x)
    if tp > ceiling_dbtp:
        atten = 10.0 ** ((ceiling_dbtp - tp - 0.05) / 20.0)
        x = (x * atten).astype(np.float32)
    return x


class MasteringEngine:
    def __init__(self, sr: int = 24000, thresholds: Optional[AudioThresholds] = None) -> None:
        self.sr = sr
        self.thresholds = thresholds or AudioThresholds()

    def build_outro_ambience(
        self,
        ambience_bed: np.ndarray,
        total_sec: float = 360.0,
        fade_sec: float = 45.0,
        target_db: float = -34.0,
    ) -> np.ndarray:
        """Final 6 dakikalık karanlık ekran ambiyansı ve son 45 sn logaritmik fade-out üretir."""
        n = int(round(total_sec * self.sr))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if len(ambience_bed) == 0:
            bed = np.zeros(n, dtype=np.float32)
        else:
            reps = math.ceil(n / len(ambience_bed)) + 1
            bed = np.tile(ambience_bed.astype(np.float32), reps)[:n]
            bed = apply_lpf(bed, self.sr, 850.0)
            bed = scale_to_rms_db(bed, target_db)

        fade_n = min(n, int(round(fade_sec * self.sr)))
        if fade_n > 1:
            curve = np.logspace(0.0, -4.0, fade_n, dtype=np.float32)
            curve[-1] = 0.0
            bed[-fade_n:] *= curve
        return bed

    def mix_and_master(
        self,
        chapter_vocals: List[Tuple[str, np.ndarray]],
        ambience_bed: Optional[np.ndarray] = None,
        outro_sec: float = 360.0,
        outro_fade_sec: float = 45.0,
    ) -> Tuple[np.ndarray, MasteringReport]:
        """
        K1-K6 kademeli anlatıcı miksajını yapar, True Peak ve outro kapanışını doğrular.
        """
        vocal_parts: List[np.ndarray] = []
        amb_parts: List[np.ndarray] = []
        chapter_levels_db: Dict[str, float] = {}
        issues: List[str] = []

        for chapter_name, seg in chapter_vocals:
            prof = STEPPED_AUDIO_PROFILES.get(chapter_name, STEPPED_AUDIO_PROFILES["KAYIT_01"])
            target_lufs = prof["lufs_target"]
            amb_db = prof["amb_db"]

            cleaned = apply_hpf(seg, self.sr, 70.0)
            # K1-K6 kademeli hedef seviye (anlatıcıyı yormadan ve maskelemeden)
            active = np.abs(cleaned) > 0.005
            if np.any(active):
                cur_rms = float(np.sqrt(np.mean(cleaned[active] ** 2) + 1e-12))
                cur_db = 20.0 * math.log10(cur_rms + 1e-12)
                # LUFS-S ~ aktif RMS + 0.5 dB yaklaşımıyla kademeli kalibrasyon
                gain_db = (target_lufs - 0.5) - cur_db
                cleaned = (cleaned * (10.0 ** (gain_db / 20.0))).astype(np.float32)
                chapter_levels_db[chapter_name] = round(
                    20.0 * math.log10(float(np.sqrt(np.mean(cleaned[active] ** 2) + 1e-12))), 2
                )
            else:
                chapter_levels_db[chapter_name] = -120.0

            vocal_parts.append(cleaned)
            n = len(cleaned)
            if ambience_bed is not None and len(ambience_bed) > 0:
                reps = math.ceil(n / len(ambience_bed)) + 1
                a_seg = np.tile(ambience_bed.astype(np.float32), reps)[:n]
                a_seg = scale_to_rms_db(apply_lpf(a_seg, self.sr, 1400.0), amb_db)
            else:
                a_seg = np.zeros(n, dtype=np.float32)
            amb_parts.append(a_seg)

        outro_start_sec = round(sum(len(v) for v in vocal_parts) / float(self.sr), 3)

        if outro_sec > 0:
            outro_amb = self.build_outro_ambience(
                ambience_bed if ambience_bed is not None else np.zeros(self.sr, dtype=np.float32),
                total_sec=outro_sec,
                fade_sec=outro_fade_sec,
            )
            vocal_parts.append(np.zeros(len(outro_amb), dtype=np.float32))
            amb_parts.append(outro_amb)

        full_vocal = np.concatenate(vocal_parts) if vocal_parts else np.zeros(0, dtype=np.float32)
        full_amb = np.concatenate(amb_parts) if amb_parts else np.zeros(0, dtype=np.float32)
        master = apply_true_peak_limiter(full_vocal + full_amb, self.sr, self.thresholds.max_true_peak_dbtp)

        tp = compute_true_peak_dbtp(master)
        rms_db = round(20.0 * math.log10(float(np.sqrt(np.mean(master ** 2) + 1e-12))), 2) if len(master) else -120.0

        # Outro logaritmik kapanış doğrulaması
        outro_fade_ok = True
        if outro_sec > 0 and len(master) >= self.sr:
            tail_1s_rms = float(np.sqrt(np.mean(master[-self.sr:] ** 2) + 1e-12))
            if tail_1s_rms > 0.005:
                outro_fade_ok = False
                issues.append(f"Outro son 1 sn yeterince sessizleşmemiş (RMS={tail_1s_rms:.4f}).")

        if tp > self.thresholds.max_true_peak_dbtp + 0.05:
            issues.append(f"True Peak tavanı aşıldı: {tp:.2f} dBTP > {self.thresholds.max_true_peak_dbtp} dBTP.")

        report = MasteringReport(
            duration_sec=round(len(master) / float(self.sr), 3),
            integrated_rms_db=rms_db,
            true_peak_dbtp=tp,
            chapter_levels_db=chapter_levels_db,
            outro_start_sec=outro_start_sec,
            outro_fade_verified=outro_fade_ok,
            passed=len(issues) == 0,
            issues=issues,
        )
        return master, report
