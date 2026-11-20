"""
Ortam ayarları, Google Colab Pro T4 GPU / bellek / yerel SSD korumaları ve proje dizin yapısı.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


def enable_line_buffering() -> None:
    """Mobil Colab tarayıcısında çıktıların takılmaması için satır tamponlamayı açar."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass


@dataclass
class EnvironmentCheckResult:
    has_cuda: bool
    device: str
    gpu_name: str
    vram_gb: float
    local_free_gb: float
    drive_mounted: bool
    ffmpeg_available: bool
    ffprobe_available: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


@dataclass
class PipelineSettings:
    project_id: str
    project_dir: str
    script_path: str
    reference_audio_path: str
    local_work_dir: str = "/content/uyku_v2_work"
    sample_rate: int = 24000
    target_fps: int = 24
    target_width: int = 1920
    target_height: int = 1080
    outro_black_seconds: float = 360.0
    outro_fade_seconds: float = 45.0
    ffmpeg_threads: int = 2
    min_local_free_gb: float = 4.0
    require_gpu_for_ai: bool = True
    pronunciation_dict_path: Optional[str] = None
    extra_params: Dict[str, object] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return Path(self.project_dir)

    @property
    def backup_dir(self) -> Path:
        return self.root / "bolum_yedekleri"

    @property
    def paragraphs_dir(self) -> Path:
        return self.backup_dir / "paragraflar_v2"

    @property
    def sfx_dir(self) -> Path:
        return self.root / "sfx_ambiyans_audioldm"

    @property
    def visuals_dir(self) -> Path:
        return self.root / "gorseller"

    @property
    def clips_dir(self) -> Path:
        return self.root / "video_klipler_svd"

    @property
    def scenes_dir(self) -> Path:
        return self.root / "video_parca_yedekleri_hareketli"

    @property
    def quarantine_dir(self) -> Path:
        return self.root / "karantina_hatali_dosyalar"

    @property
    def reports_dir(self) -> Path:
        return self.root / "raporlar"

    @property
    def logs_dir(self) -> Path:
        return self.root / "loglar"

    @property
    def manifest_path(self) -> Path:
        return self.backup_dir / "uretim_manifesti_v2.json"

    @property
    def timestamps_path(self) -> Path:
        return self.backup_dir / "zaman_damgalari_v2.json"

    def ensure_directories(self) -> None:
        for d in [
            self.root,
            self.backup_dir,
            self.paragraphs_dir,
            self.sfx_dir,
            self.visuals_dir,
            self.clips_dir,
            self.scenes_dir,
            self.quarantine_dir,
            self.reports_dir,
            self.logs_dir,
            Path(self.local_work_dir),
        ]:
            d.mkdir(parents=True, exist_ok=True)


def check_runtime_environment(
    require_gpu: bool = True,
    local_work_dir: str = "/tmp",
    min_local_free_gb: float = 2.0,
    drive_mount_point: Optional[str] = None,
) -> EnvironmentCheckResult:
    """
    GPU, VRAM, yerel SSD boş alanı, Drive bağlantısı ve FFmpeg durumunu denetler.
    GPU gerektiren aşamalarda CPU üzerinde sessizce başlamayı ENGELLEMEK için hata üretir.
    """
    warnings: list[str] = []
    errors: list[str] = []

    has_cuda = False
    gpu_name = "CPU"
    vram_gb = 0.0
    try:
        import torch  # type: ignore

        has_cuda = bool(torch.cuda.is_available())
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024 ** 3), 2)
    except ImportError:
        warnings.append("PyTorch yüklü değil (yalnızca CPU/yerel test modu kullanılabilir).")

    device = "cuda" if has_cuda else "cpu"
    if require_gpu and not has_cuda:
        errors.append(
            "GPU GEREKLİ: CUDA/T4 GPU algılanamadı! Sessizce CPU modunda çalıştırma "
            "güvenlik kuralı gereği engellendi. Colab menüsünden 'Çalışma Zamanı -> T4 GPU' seçin."
        )

    if has_cuda and vram_gb < 12.0:
        warnings.append(f"Düşük VRAM tespit edildi ({vram_gb} GB). T4 optimizasyonları (fp16, cpu_offload) zorunlu.")

    check_path = local_work_dir if os.path.exists(local_work_dir) else "/tmp"
    usage = shutil.disk_usage(check_path)
    local_free_gb = round(usage.free / (1024 ** 3), 2)
    if local_free_gb < min_local_free_gb:
        errors.append(
            f"Yetersiz yerel SSD alanı: {local_free_gb:.2f} GB boş (minimum {min_local_free_gb:.2f} GB gerekli)."
        )

    drive_mounted = True
    if drive_mount_point:
        drive_mounted = os.path.exists(drive_mount_point) and os.path.isdir(drive_mount_point)
        if not drive_mounted:
            errors.append(f"Google Drive bağlı değil veya erişilemiyor: {drive_mount_point}")

    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    if not ffmpeg_ok or not ffprobe_ok:
        errors.append("FFmpeg veya FFprobe sistemde bulunamadı.")

    return EnvironmentCheckResult(
        has_cuda=has_cuda,
        device=device,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        local_free_gb=local_free_gb,
        drive_mounted=drive_mounted,
        ffmpeg_available=ffmpeg_ok,
        ffprobe_available=ffprobe_ok,
        warnings=warnings,
        errors=errors,
    )
