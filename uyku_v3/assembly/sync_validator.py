"""
Ses-Video Senkron, Saf Siyah Ekran (#000000) ve Tam Decode Doğrulayıcısı (Bölüm 11).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config.quality_thresholds import AssemblyThresholds
from ..video.video_validator import probe_video, validate_video_file


@dataclass
class SyncValidationResult:
    passed: bool
    video_duration_sec: float
    audio_duration_sec: float
    sync_diff_sec: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    decode_ok: bool
    black_screen_verified: bool
    black_pixel_mean: float
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def probe_media_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stderr=subprocess.STDOUT,
    ).decode().strip()
    return float(out)


def verify_black_screen_pure(
    video_path: Path,
    sample_time_sec: float,
    max_pixel_mean: float = 1.0,
) -> Tuple[bool, float]:
    """
    Belirtilen saniyedeki kareyi okuyup gerçekten saf siyah (#000000) olduğunu doğrular.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 255.0
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, sample_time_sec) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, 255.0
        mean_val = float(np.mean(frame))
        return (mean_val <= max_pixel_mean), round(mean_val, 4)
    finally:
        cap.release()


def validate_final_assembly(
    final_mp4_path: Path,
    master_audio_path: Path,
    outro_start_sec: Optional[float] = None,
    thresholds: Optional[AssemblyThresholds] = None,
) -> SyncValidationResult:
    th = thresholds or AssemblyThresholds()
    issues: List[str] = []

    vid_res = validate_video_file(
        final_mp4_path,
        expected_width=th.expected_width,
        expected_height=th.expected_height,
        expected_fps=th.expected_fps,
        min_duration_sec=1.0,
        require_audio=True,
        run_full_decode=True,
    )
    if not vid_res.passed:
        issues.extend(vid_res.issues)

    info = vid_res.info
    v_dur = info.duration_sec if info else 0.0
    try:
        a_dur = probe_media_duration(master_audio_path)
    except Exception as exc:
        a_dur = 0.0
        issues.append(f"Master ses süresi okunamadı: {exc}")

    sync_diff = round(abs(v_dur - a_dur), 3)
    if sync_diff > th.max_av_sync_diff_sec:
        issues.append(f"Ses-video senkron farkı yüksek: {sync_diff:.3f} sn > {th.max_av_sync_diff_sec} sn.")

    if info and info.video_codec != "h264":
        issues.append(f"YouTube standart codec H.264 değil: {info.video_codec}")
    if info and info.audio_codec != "aac":
        issues.append(f"YouTube standart ses codec AAC değil: {info.audio_codec}")

    black_ok = True
    black_mean = 0.0
    if outro_start_sec is not None and v_dur > outro_start_sec + 1.0:
        sample_t = min(v_dur - 0.5, outro_start_sec + 2.0)
        black_ok, black_mean = verify_black_screen_pure(final_mp4_path, sample_t, th.max_black_pixel_mean)
        if not black_ok:
            issues.append(f"Outro siyah ekranı saf #000000 değil (ortalama piksel={black_mean:.2f}).")

    return SyncValidationResult(
        passed=len(issues) == 0,
        video_duration_sec=v_dur,
        audio_duration_sec=round(a_dur, 3),
        sync_diff_sec=sync_diff,
        width=info.width if info else 0,
        height=info.height if info else 0,
        fps=info.fps if info else 0.0,
        video_codec=info.video_codec if info else "",
        audio_codec=info.audio_codec if info else "",
        decode_ok=vid_res.decode_ok,
        black_screen_verified=black_ok,
        black_pixel_mean=black_mean,
        issues=issues,
    )
