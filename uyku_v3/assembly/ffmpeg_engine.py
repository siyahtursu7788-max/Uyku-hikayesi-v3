"""
Yerel SSD Tabanlı FFmpeg Montaj Motoru (Bölüm 11).
- Büyük birleştirmeleri Drive FUSE üzerinde ASLA yapmaz; Colab yerel SSD'sinde (/content/...) çalışır.
- Onaylanmamış veya eksik sahne klibi varsa montajı BAŞLATMADAN durdurur.
- Tüm sahneleri tek tip 1920x1080, 24 FPS, libx264, yuv420p standardına getirir.
- Saf siyah ekran (#000000) outro parçasını ayrı üretir ve doğrular.
"""

from __future__ import annotations

import gc
import shutil
import subprocess
from pathlib import Path
from typing import List

from ..core.file_manager import atomic_copy_file
from ..video.video_validator import validate_video_file
from .timeline import TimelineSegment


class FFmpegAssemblyEngine:
    def __init__(
        self,
        local_work_dir: Path,
        width: int = 1920,
        height: int = 1080,
        fps: int = 24,
        ffmpeg_threads: int = 2,
    ) -> None:
        self.local_work_dir = Path(local_work_dir)
        self.local_work_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.ffmpeg_threads = ffmpeg_threads

    def render_black_outro(self, duration_sec: float, output_path: Path) -> Path:
        """Tamamen #000000 saf siyah ekran MP4 parçası üretir (yuv420p, 1920x1080, 24fps)."""
        local_out = self.local_work_dir / f"tmp_black_outro_{output_path.stem}.mp4"
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error", "-threads", str(self.ffmpeg_threads),
            "-f", "lavfi",
            "-i", f"color=c=#000000:s={self.width}x{self.height}:r={self.fps}:d={duration_sec:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-an",
            str(local_out),
        ])
        atomic_copy_file(local_out, output_path, verify_hash=True)
        local_out.unlink(missing_ok=True)
        return output_path

    def render_extended_scene(
        self,
        loop_clip_path: Path,
        duration_sec: float,
        output_path: Path,
        fade_sec: float = 2.0,
    ) -> Path:
        """Onaylı döngü klibini sahne süresine uzatır, giriş/çıkış yumuşak fade uygular."""
        if not loop_clip_path.exists():
            raise FileNotFoundError(f"Montaja girecek sahne klibi eksik: {loop_clip_path}")

        clip_check = validate_video_file(
            loop_clip_path,
            expected_width=self.width,
            expected_height=self.height,
            expected_fps=float(self.fps),
            min_duration_sec=0.5,
            run_full_decode=True,
        )
        if not clip_check.passed:
            raise ValueError(f"Bozuk veya standart dışı klip montaja alınamaz ({loop_clip_path}): {clip_check.issues}")

        fade_d = min(fade_sec, max(0.2, duration_sec / 4.0))
        fade_out_st = max(0.1, duration_sec - fade_d)
        local_out = self.local_work_dir / f"tmp_scene_{output_path.stem}.mp4"
        vf = (
            f"scale={self.width}:{self.height}:flags=lanczos,"
            f"fps={self.fps},"
            f"fade=t=in:st=0:d={fade_d:.2f},"
            f"fade=t=out:st={fade_out_st:.2f}:d={fade_d:.2f}"
        )
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error", "-threads", str(self.ffmpeg_threads),
            "-stream_loop", "-1", "-i", str(loop_clip_path),
            "-vf", vf, "-t", f"{duration_sec:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-an",
            str(local_out),
        ])
        atomic_copy_file(local_out, output_path, verify_hash=True)
        local_out.unlink(missing_ok=True)
        gc.collect()
        return output_path

    def assemble_final_mp4(
        self,
        scene_files: List[Path],
        master_audio_path: Path,
        final_output_path: Path,
    ) -> Path:
        """
        Tüm parçaları yerel SSD'ye kopyalar, tek geçitte birleştirir (-movflags +faststart),
        hızlı copy hata verirse uyumlu yeniden kodlama yapar ve Drive'a atomik kopyalar.
        """
        if not scene_files:
            raise ValueError("Birleştirilecek sahne listesi boş.")
        if not master_audio_path.exists():
            raise FileNotFoundError(f"Master ses dosyası bulunamadı: {master_audio_path}")

        stage_dir = self.local_work_dir / "concat_stage"
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True, exist_ok=True)

        try:
            local_scenes: List[Path] = []
            for i, sf in enumerate(scene_files):
                if not sf.exists():
                    raise FileNotFoundError(f"Eksik sahne parçası ({i}): {sf}")
                dst = stage_dir / f"part_{i:03d}.mp4"
                shutil.copy2(sf, dst)
                local_scenes.append(dst)

            local_audio = stage_dir / f"master_audio{master_audio_path.suffix}"
            shutil.copy2(master_audio_path, local_audio)

            concat_txt = stage_dir / "concat.txt"
            with open(concat_txt, "w", encoding="utf-8") as f:
                for lp in local_scenes:
                    f.write(f"file '{lp}'\n")

            local_final = stage_dir / "final_assembled.mp4"
            cmd_copy = [
                "ffmpeg", "-y", "-loglevel", "error", "-threads", str(self.ffmpeg_threads),
                "-f", "concat", "-safe", "0", "-i", str(concat_txt),
                "-i", str(local_audio),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                "-shortest", "-movflags", "+faststart",
                str(local_final),
            ]
            proc = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                # Stream copy uyuşmazlığı durumunda güvenli yeniden kodlama
                subprocess.check_call([
                    "ffmpeg", "-y", "-loglevel", "error", "-threads", str(self.ffmpeg_threads),
                    "-f", "concat", "-safe", "0", "-i", str(concat_txt),
                    "-i", str(local_audio),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-r", str(self.fps),
                    "-c:a", "aac", "-b:a", "256k",
                    "-shortest", "-movflags", "+faststart",
                    str(local_final),
                ])

            atomic_copy_file(local_final, final_output_path, verify_hash=True)
            return final_output_path
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)
