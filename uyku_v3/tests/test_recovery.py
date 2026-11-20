"""
Bölüm 15 Kesinti, Drive Kopması, Yarım Dosya ve Manifest Kurtarma Testleri (Test 29–33).
Tamamı gerçek yerel dosya sistemi üzerinde çalışır (REAL_LOCAL).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uyku_v3.core.file_manager import atomic_copy_file, atomic_write_text, sha256_text
from uyku_v3.core.manifest import ManifestManager
from uyku_v3.core.recovery import RecoveryManager
from uyku_v3.core.state_manager import ArtifactStatus


class TestRecoveryAndManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest_path = self.root / "manifest.json"
        self.quarantine = self.root / "karantina"
        self.out_dir = self.root / "outputs"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_29_colab_session_interruption_recovery(self) -> None:
        """[REAL_LOCAL] Colab oturumu GENERATING sırasında kesildiğinde dosya karantinaya alınır ve INTERRUPTED olur."""
        m = ManifestManager(self.manifest_path, "test_proj")
        wav_p = self.out_dir / "paragraf_001.wav"
        m.register_or_validate_cache("p001", "audio", str(wav_p), sha256_text("metin"), "xtts", "2.0", {"t": 0.22})
        m.set_status("p001", ArtifactStatus.GENERATING, increment_attempt=True)
        wav_p.write_bytes(b"HALF_WRITTEN_BYTES")

        # Yeni oturum açıldı
        m2 = ManifestManager(self.manifest_path, "test_proj")
        rec = RecoveryManager(m2, self.quarantine)
        rep = rec.recover_after_interruption([self.out_dir])
        self.assertIn("p001", rep["interrupted_records"])
        self.assertEqual(m2.get("p001").get_status(), ArtifactStatus.INTERRUPTED)
        self.assertFalse(wav_p.exists())
        self.assertTrue(len(rep["quarantined_files"]) >= 1)

    def test_30_drive_disconnect_resilience_atomic_copy(self) -> None:
        """[REAL_LOCAL] Drive kopyalamasında geçici I/O hatası sonrası yeniden deneme ve veri kaybı olmaması."""
        src = self.root / "local_master.bin"
        dst = self.out_dir / "drive_master.bin"
        src.write_bytes(b"VALID_AUDIO_DATA_1234567890")

        call_count = {"n": 0}
        real_copy = __import__("shutil").copy2

        def flaky_copy(a, b):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("Simulated Google Drive FUSE disconnect")
            return real_copy(a, b)

        with patch("uyku_v3.core.file_manager.shutil.copy2", side_effect=flaky_copy):
            h = atomic_copy_file(src, dst, verify_hash=True, retries=3, retry_delay_sec=0.01)

        self.assertEqual(call_count["n"], 2)
        self.assertTrue(dst.exists())
        self.assertEqual(dst.read_bytes(), src.read_bytes())
        self.assertTrue(len(h) == 64)

    def test_31_partial_tmp_file_quarantined(self) -> None:
        """[REAL_LOCAL] Yarım kalmış .tmp dosyaları kurtarma sırasında karantinaya taşınır."""
        tmp_file = self.out_dir / ".klip_03.mp4.1234.tmp"
        tmp_file.write_bytes(b"PARTIAL")
        m = ManifestManager(self.manifest_path, "test_proj")
        rec = RecoveryManager(m, self.quarantine)
        rep = rec.recover_after_interruption([self.out_dir])
        self.assertFalse(tmp_file.exists())
        self.assertEqual(len(rep["quarantined_files"]), 1)

    def test_32_corrupt_manifest_recovery(self) -> None:
        """[REAL_LOCAL] Bozuk JSON manifest dosyası yedeğe alınıp sistem çökmeden yeniden başlar."""
        self.manifest_path.write_text("{CORRUPTED JSON...", encoding="utf-8")
        m = ManifestManager(self.manifest_path, "test_proj")
        self.assertEqual(m.data["project_id"], "test_proj")
        self.assertTrue(self.manifest_path.with_suffix(".corrupt.bak").exists())

    def test_33_rerun_preserves_approved_and_invalidates_on_source_change(self) -> None:
        """[REAL_LOCAL] Yeniden çalıştırmada geçerli APPROVED çıktılar korunur; kaynak metin değişirse geçersizleşir."""
        m = ManifestManager(self.manifest_path, "test_proj")
        wav_p = self.out_dir / "paragraf_000.wav"
        atomic_write_text(wav_p, "FAKE_WAV_CONTENT")

        h1 = sha256_text("Orijinal cümle.")
        m.register_or_validate_cache("p000", "audio", str(wav_p), h1, "xtts", "2.0", {"t": 0.22})
        m.set_status("p000", ArtifactStatus.GENERATING)
        m.set_status("p000", ArtifactStatus.VALIDATING)
        m.set_status("p000", ArtifactStatus.APPROVED, compute_output_hash=True)

        # Aynı hash ile tekrar kayıt -> APPROVED korunmalı
        r_same = m.register_or_validate_cache("p000", "audio", str(wav_p), h1, "xtts", "2.0", {"t": 0.22})
        self.assertEqual(r_same.get_status(), ArtifactStatus.APPROVED)

        # Kaynak metin değişti -> otomatik PENDING'e düşmeli!
        h2 = sha256_text("Değiştirilmiş yeni cümle.")
        r_changed = m.register_or_validate_cache("p000", "audio", str(wav_p), h2, "xtts", "2.0", {"t": 0.22})
        self.assertEqual(r_changed.get_status(), ArtifactStatus.PENDING)


if __name__ == "__main__":
    unittest.main()
