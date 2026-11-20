"""
Colab oturumu kesintisi ve Drive bağlantı kopması kurtarma motoru (Bölüm 13).
- GENERATING / VALIDATING durumunda yarıda kalan kayıtları INTERRUPTED işaretler.
- Yarım yazılmış (.tmp veya 0 byte veya hash'i tutmayan) dosyaları karantinaya alır.
- Yalnızca doğrulanmış APPROVED çıktıları korur, başarısız/eksik aşamaları yeniden kuyruğa alır.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

from .file_manager import sha256_file
from .manifest import ManifestManager
from .state_manager import ArtifactStatus


class RecoveryManager:
    def __init__(self, manifest: ManifestManager, quarantine_dir: Path) -> None:
        self.manifest = manifest
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def recover_after_interruption(self, search_dirs: List[Path] | None = None) -> Dict[str, List[str]]:
        """
        Kesinti sonrası sistemi güvenli başlangıç durumuna getirir:
        1. Geçici (.tmp) yarım dosyaları temizler/karantinaya alır.
        2. GENERATING veya VALIDATING durumunda kalmış çıktıları INTERRUPTED yapar.
        3. APPROVED görünen ancak diskte eksik, 0 byte veya hash'i bozulmuş dosyaları INTERRUPTED yapar.
        """
        report: Dict[str, List[str]] = {
            "interrupted_records": [],
            "quarantined_files": [],
            "preserved_approved": [],
        }

        # 1. Geçici yarım dosyaları karantinaya taşı
        for d in (search_dirs or []):
            dp = Path(d)
            if not dp.exists():
                continue
            for tmp_file in dp.rglob("*.tmp"):
                dst = self.quarantine_dir / f"partial_{tmp_file.name}"
                try:
                    shutil.move(str(tmp_file), str(dst))
                    report["quarantined_files"].append(str(dst))
                except OSError:
                    pass

        # 2. Manifest kayıtlarını denetle
        for aid, raw in list(self.manifest.data["artifacts"].items()):
            rec = self.manifest.get(aid)
            if rec is None:
                continue
            st = rec.get_status()
            out_p = Path(rec.output_path)

            if st in (ArtifactStatus.GENERATING, ArtifactStatus.VALIDATING):
                if out_p.exists():
                    q_dst = self.quarantine_dir / f"interrupted_{out_p.name}"
                    try:
                        shutil.move(str(out_p), str(q_dst))
                        report["quarantined_files"].append(str(q_dst))
                    except OSError:
                        pass
                self.manifest.set_status(
                    aid,
                    ArtifactStatus.INTERRUPTED,
                    error_message="Oturum kesintisi sırasında yarım kaldı; yeniden üretime alındı.",
                )
                report["interrupted_records"].append(aid)

            elif st == ArtifactStatus.APPROVED:
                valid_file = out_p.exists() and out_p.stat().st_size > 0
                if valid_file and rec.output_hash:
                    try:
                        valid_file = sha256_file(out_p) == rec.output_hash
                    except OSError:
                        valid_file = False
                if valid_file:
                    report["preserved_approved"].append(aid)
                else:
                    if out_p.exists():
                        q_dst = self.quarantine_dir / f"corrupt_{out_p.name}"
                        try:
                            shutil.move(str(out_p), str(q_dst))
                            report["quarantined_files"].append(str(q_dst))
                        except OSError:
                            pass
                    rec.status = ArtifactStatus.INTERRUPTED.value
                    rec.output_hash = ""
                    rec.error_messages.append("Onaylı çıktı diskte bozuk veya eksik bulundu; kurtarma kuyruğuna alındı.")
                    self.manifest.upsert(rec)
                    report["interrupted_records"].append(aid)

        self.manifest.save()
        return report
