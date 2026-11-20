"""
uyku_v3 — Birleşik Komut Satırı ve Google Colab Giriş Noktası.
Örnek Colab kullanımı:
    %run "/content/drive/MyDrive/UYKU_BORU_HATTI_PROFESYONEL_V2/02_GUNCELLENMIS_KODLAR/main.py" \
        --project-id kul_koyu_01_06 \
        --project-dir "/content/drive/MyDrive/Kul_Koyu_Siginagi_Studyo" \
        --script "/content/drive/MyDrive/Kul_Koyu_Siginagi_Studyo/Kul_Koyu_Siginagi_01_06_Final.txt" \
        --reference-audio "/content/drive/MyDrive/Kul_Koyu_Siginagi_Studyo/referans_ses_15sn.wav" \
        --stage check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Paket kök dizinini sys.path'e ekle (Colab %run uyumluluğu)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from uyku_v3.config.settings import (
    PipelineSettings,
    check_runtime_environment,
    enable_line_buffering,
)
from uyku_v3.core.logging_system import PipelineLogger
from uyku_v3.core.manifest import ManifestManager
from uyku_v3.core.recovery import RecoveryManager


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Türkçe Uyku Hikâyesi Profesyonel Üretim Boru Hattı V2"
    )
    parser.add_argument("--project-id", required=True, help="Benzersiz proje kimliği (örn: kul_koyu_01_06)")
    parser.add_argument("--project-dir", required=True, help="Google Drive proje klasörü yolu")
    parser.add_argument("--script", required=True, help="Orijinal .txt senaryo dosyası")
    parser.add_argument("--reference-audio", required=True, help="15-18 sn referans_ses_15sn.wav dosyası")
    parser.add_argument("--local-work-dir", default="/content/uyku_v2_work", help="Colab yerel SSD çalışma alanı")
    parser.add_argument(
        "--stage",
        choices=["check", "recover", "status"],
        default="check",
        help="Çalıştırılacak operasyon (check, recover, status)",
    )
    parser.add_argument("--allow-cpu", action="store_true", help="Yalnızca yerel test/denetim için CPU izni")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    enable_line_buffering()
    args = parse_args(argv)

    settings = PipelineSettings(
        project_id=args.project_id,
        project_dir=args.project_dir,
        script_path=args.script,
        reference_audio_path=args.reference_audio,
        local_work_dir=args.local_work_dir,
        require_gpu_for_ai=not args.allow_cpu,
    )
    settings.ensure_directories()
    logger = PipelineLogger(settings.logs_dir, settings.project_id)
    manifest = ManifestManager(settings.manifest_path, settings.project_id)

    if args.stage == "check":
        env = check_runtime_environment(
            require_gpu=settings.require_gpu_for_ai,
            local_work_dir=settings.local_work_dir,
            min_local_free_gb=settings.min_local_free_gb,
        )
        logger.info("ENV", f"Cihaz: {env.device} ({env.gpu_name}, {env.vram_gb} GB VRAM) | SSD Boş: {env.local_free_gb} GB")
        for w in env.warnings:
            logger.warning("ENV", w)
        for e in env.errors:
            logger.error("ENV", e)
        return 0 if env.ok else 2

    if args.stage == "recover":
        rec = RecoveryManager(manifest, settings.quarantine_dir)
        res = rec.recover_after_interruption([settings.paragraphs_dir, settings.clips_dir, settings.scenes_dir])
        logger.info("RECOVERY", f"Kurtarma tamamlandı: {json.dumps(res, ensure_ascii=False)}")
        return 0

    if args.stage == "status":
        summary = {
            "project_id": settings.project_id,
            "counts": manifest.summary_counts(),
            "retry_queue": manifest.data.get("retry_queue", []),
            "manual_review_queue": manifest.data.get("manual_review_queue", []),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
