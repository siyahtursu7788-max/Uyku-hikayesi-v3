"""
FFprobe + Tam Akış Decode Video Doğrulayıcısı (Bölüm 9 & 11).
Dosyanın varlığı veya boyutu tek başına geçerlilik değildir:
- Çözünürlük, FPS, codec, süre, ses kanalı varlığı ve tam decode edilebilirlik denetlenir.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class VideoStreamInfo:
    duration_sec: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    has_audio: bool
    file_size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoValidationResult:
    path: str
    passed: bool
    decode_ok: bool
    info: Optional[VideoStreamInfo]
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def probe_video(path: Union[str, Path]) -> Optional[VideoStreamInfo]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        raw = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_format", "-show_streams",
                "-of", "json", str(p),
            ],
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
        data = json.loads(raw)
    except Exception:
        return None

    streams = data.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v_stream is None:
        return None

    dur = float(data.get("format", {}).get("duration", 0.0) or v_stream.get("duration", 0.0) or 0.0)
    fps_str = v_stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except Exception:
        fps = 0.0

    return VideoStreamInfo(
        duration_sec=round(dur, 3),
        width=int(v_stream.get("width", 0)),
        height=int(v_stream.get("height", 0)),
        fps=round(fps, 3),
        video_codec=str(v_stream.get("codec_name", "")),
        audio_codec=str(a_stream.get("codec_name", "")) if a_stream else "",
        has_audio=a_stream is not None,
        file_size_bytes=p.stat().st_size,
    )


def verify_full_decode(path: Union[str, Path], ffmpeg_threads: int = 2) -> bool:
    """Videonun baştan sona tek bir decode hatası vermeden çözülebildiğini doğrular."""
    p = Path(path)
    if not p.exists():
        return False
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-threads", str(ffmpeg_threads), "-i", str(p), "-f", "null", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode == 0 and proc.stderr.strip() == ""


def validate_video_file(
    path: Union[str, Path],
    expected_width: int = 1920,
    expected_height: int = 1080,
    expected_fps: float = 24.0,
    min_duration_sec: float = 1.0,
    require_audio: bool = False,
    run_full_decode: bool = True,
) -> VideoValidationResult:
    p = Path(path)
    issues: List[str] = []
    info = probe_video(p)
    if info is None:
        return VideoValidationResult(str(p), False, False, None, ["MP4 bozuk, boş veya ffprobe ile okunamadı."])

    if info.duration_sec < min_duration_sec:
        issues.append(f"Video süresi çok kısa ({info.duration_sec:.2f} sn < {min_duration_sec} sn).")
    if info.width != expected_width or info.height != expected_height:
        issues.append(f"Çözünürlük uyumsuz: {info.width}x{info.height} != {expected_width}x{expected_height}.")
    if abs(info.fps - expected_fps) > 0.25:
        issues.append(f"FPS uyumsuz: {info.fps:.2f} != {expected_fps:.2f}.")
    if require_audio and not info.has_audio:
        issues.append("Videoda ses akışı bulunamadı.")

    decode_ok = True
    if run_full_decode:
        decode_ok = verify_full_decode(p)
        if not decode_ok:
            issues.append("Tam akış decode testinde hata oluştu (bozuk MP4 akışı).")

    return VideoValidationResult(
        path=str(p),
        passed=len(issues) == 0,
        decode_ok=decode_ok,
        info=info,
        issues=issues,
    )
