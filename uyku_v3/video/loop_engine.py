"""
Doğal ve Kesintisiz Video Döngü Motoru (Bölüm 10).
P1-03 DÜZELTMESİ:
- Tüm sahnelerde körlemesine ping-pong KULLANILMAZ!
- Sakin/simetrik iç mekan sahnelerinde 'ping_pong',
- Ateş, yağmur, dalga, buhar gibi yönlü hareketlerde tersine dönüşü önleyen
  'forward_crossfade' (ileri yönlü yumuşak geçiş) veya 'optical_blend' kullanılır.
- Döngü birleşim noktası (seam delta) ve hayalet görüntü (ghosting) riski ölçülerek doğrulanır.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from ..core.file_manager import atomic_copy_file


@dataclass
class LoopResult:
    output_path: str
    loop_mode: str
    input_frames: int
    loop_frames: int
    seam_delta: float
    ghosting_risk: float
    passed: bool
    issues: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_ping_pong_sequence(frames: List[Image.Image]) -> List[Image.Image]:
    """Simetrik sakin sahneler için ileri + geri (0..N-1 + N-2..1) dizilim."""
    if len(frames) <= 2:
        return list(frames)
    return list(frames) + list(reversed(frames[1:-1]))


def build_forward_crossfade_sequence(
    frames: List[Image.Image],
    overlap_frames: int = 6,
) -> Tuple_Seq:
    """
    Yönlü hareketler (ateş, yağmur, dalga, buhar) için tersine oynatmadan ileri yönlü
    döngü oluşturur: son `overlap_frames` kare ile ilk `overlap_frames` kareyi
    yumuşak kosinüs ağırlığıyla harmanlayarak kesintisiz ileri döngü kurar.
    Döndürür: (loop_frames, ghosting_risk_score)
    """
    n = len(frames)
    k = max(2, min(overlap_frames, n // 3))
    body = [np.array(im.convert("RGB"), dtype=np.float32) for im in frames[k:n - k]]
    head = [np.array(im.convert("RGB"), dtype=np.float32) for im in frames[:k]]
    tail = [np.array(im.convert("RGB"), dtype=np.float32) for im in frames[n - k:]]

    blended_seam: List[np.ndarray] = []
    ghost_deltas: List[float] = []
    for i in range(k):
        # Kosinüs S-eğrisi ile yumuşak geçiş (0 -> 1)
        alpha = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / float(k + 1)))
        mix = (1.0 - alpha) * tail[i] + alpha * head[i]
        blended_seam.append(mix)
        ghost_deltas.append(float(np.mean(np.abs(tail[i] - head[i]))))

    all_arrs = body + blended_seam
    out_imgs = [Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)) for a in all_arrs]
    ghosting_risk = float(np.mean(ghost_deltas)) if ghost_deltas else 0.0
    return out_imgs, round(ghosting_risk, 2)


Tuple_Seq = tuple[List[Image.Image], float]


def build_seamless_loop(
    frames: List[Image.Image],
    output_mp4: Path,
    local_tmp_dir: Path,
    loop_mode: str = "ping_pong",
    fps_source: int = 7,
    fps_target: int = 24,
    target_width: int = 1920,
    target_height: int = 1080,
    ffmpeg_threads: int = 2,
    max_seam_delta: float = 15.0,
) -> LoopResult:
    """
    Hareket türüne uygun döngüyü kurar, yerel SSD'de 1080p 24fps H.264 olarak kodlar,
    birleşim noktasını doğrular ve atomik olarak hedefe kopyalar.
    """
    issues: List[str] = []
    if len(frames) < 4:
        return LoopResult(str(output_mp4), loop_mode, len(frames), 0, 999.0, 0.0, False, ["Döngü için en az 4 kare gerekli."])

    ghosting_risk = 0.0
    if loop_mode == "ping_pong":
        seq = build_ping_pong_sequence(frames)
    elif loop_mode in ("forward_crossfade", "optical_blend"):
        seq, ghosting_risk = build_forward_crossfade_sequence(frames, overlap_frames=min(6, len(frames) // 3))
    else:
        issues.append(f"Bilinmeyen döngü modu: {loop_mode}")
        seq = build_ping_pong_sequence(frames)

    # Döngü birleşim noktası kontrolü (son kare -> ilk kare)
    first_arr = np.array(seq[0].convert("RGB").resize((256, 144)), dtype=np.float32)
    last_arr = np.array(seq[-1].convert("RGB").resize((256, 144)), dtype=np.float32)
    seam_delta = float(np.mean(np.abs(last_arr - first_arr)))
    if seam_delta > max_seam_delta:
        issues.append(f"Döngü birleşim noktası farkı yüksek ({seam_delta:.2f} > {max_seam_delta}).")

    frame_dir = local_tmp_dir / f"loop_frames_{output_mp4.stem}"
    shutil.rmtree(frame_dir, ignore_errors=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i, im in enumerate(seq):
            im.convert("RGB").resize((target_width, target_height), Image.Resampling.LANCZOS).save(
                frame_dir / f"f_{i:04d}.png"
            )

        local_mp4 = local_tmp_dir / f"{output_mp4.stem}_local.mp4"
        vf = (
            f"framerate=fps={fps_target}:interp_start=0:interp_end=255:scene=100,"
            "eq=brightness=-0.02:contrast=0.98,vignette=PI/5"
        )
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error", "-threads", str(ffmpeg_threads),
            "-framerate", str(fps_source),
            "-i", str(frame_dir / "f_%04d.png"),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "19",
            "-pix_fmt", "yuv420p", "-an",
            str(local_mp4),
        ])
        atomic_copy_file(local_mp4, output_mp4, verify_hash=True)
        local_mp4.unlink(missing_ok=True)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)

    return LoopResult(
        output_path=str(output_mp4),
        loop_mode=loop_mode,
        input_frames=len(frames),
        loop_frames=len(seq),
        seam_delta=round(seam_delta, 2),
        ghosting_risk=round(ghosting_risk, 2),
        passed=len(issues) == 0,
        issues=issues,
    )
