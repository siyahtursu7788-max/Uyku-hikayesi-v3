"""UYKU_HIKAYESI_V3_FINAL_PRODUCTION — Kapsamlı V3 Platform & 30 Dakikalık E2E Test Paketi."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from uyku_v3.ambience.audioldm_engine import AMBIENCE_CATALOG, AudioLDMAmbienceEngine
from uyku_v3.assembly.ffmpeg_engine import FFmpegAssemblyEngine
from uyku_v3.assembly.sync_validator import validate_final_assembly
from uyku_v3.audio.xtts_engine import XTTSEngine
from uyku_v3.config.model_config import XTTSConfig
from uyku_v3.config.quality_thresholds import AssemblyThresholds, AudioThresholds, SRTThresholds
from uyku_v3.core.file_manager import sha256_file
from uyku_v3.core.protection import LegacyProtectionGuard
from uyku_v3.core.resource_monitor import snapshot_system_resources
from uyku_v3.core.state_manager import ArtifactStatus
from uyku_v3.core.v3_cache import SixFactorCacheKey, VerifiedV3Cache
from uyku_v3.subtitles.srt_generator import SRTCue, write_srt_file
from uyku_v3.subtitles.srt_validator import validate_srt_file
from uyku_v3.video.motion_validator import audit_extra_frame_anomalies
from uyku_v3.visuals.continuity_manager import CharacterProfile, VisualContinuityManager, VisualStyleBible


class TestV3PlatformSuite(unittest.TestCase):
    def test_v3_01_six_factor_cache_and_fail_closed(self) -> None:
        """[REAL_LOCAL] 6 faktörlü SHA-256 önbellek isabeti, geçersizleştirme ve bozulma reddi."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache = VerifiedV3Cache(root / "cache")
            sample = root / "ok.wav"
            sample.write_bytes(b"RIFF" + b"\x00" * 128)

            base = SixFactorCacheKey("t1", "m1", "r1", "p1", "c1", "s1")
            with self.assertRaises(ValueError):
                cache.put_approved(base, sample, status="MANUAL_REVIEW")

            saved = cache.put_approved(base, sample, {"sentence": "Hoş geldin."})
            hit = cache.get(base, ext=".wav")
            self.assertIsNotNone(hit)
            self.assertEqual(cache.hits, 1)

            # 6 faktörden herhangi biri değişirse önbellek geçersizleşmeli
            for field_name in (
                "text_hash", "model_hash", "reference_audio_hash",
                "params_hash", "thresholds_config_hash", "source_file_hash",
            ):
                kwargs = {
                    "text_hash": "t1", "model_hash": "m1", "reference_audio_hash": "r1",
                    "params_hash": "p1", "thresholds_config_hash": "c1", "source_file_hash": "s1",
                }
                kwargs[field_name] = "CHANGED"
                self.assertIsNone(cache.get(SixFactorCacheKey(**kwargs), ext=".wav"))

            # Dosya bozulursa SHA-256 reddi
            Path(saved["path"]).write_bytes(b"CORRUPT")
            self.assertIsNone(cache.get(base, ext=".wav"))
            self.assertGreaterEqual(cache.corrupt_rejections, 1)

    def test_v3_02_profile_success_stats_root_cause_fixed(self) -> None:
        """[SIMULATED_GPU] xtts_engine history alanında needs_manual_review kaydedilir ve passes doğru sayılır."""
        sr = 24000
        t = np.linspace(0, 2.0, int(sr * 2.0), dtype=np.float32)
        env = np.ones_like(t)
        env[:int(sr * 0.05)] = np.linspace(0.0, 1.0, int(sr * 0.05), dtype=np.float32)
        env[-int(sr * 0.15):] = np.linspace(1.0, 0.0, int(sr * 0.15), dtype=np.float32)
        clean_wav = np.concatenate([
            np.zeros(int(sr * 0.08), dtype=np.float32),
            (0.14 * np.sin(2 * np.pi * 220 * t) * env).astype(np.float32),
            np.zeros(int(sr * 0.25), dtype=np.float32),
        ])

        def fake_synth(text, temp, speed, top_p):
            return clean_wav.copy()

        def fake_whisper(audio, sr_in):
            return "Orada hiçbir anı kaybolmaz.", [
                {"word": "Orada", "start": 0.1, "end": 0.4},
                {"word": "hiçbir", "start": 0.5, "end": 0.9},
                {"word": "anı", "start": 1.0, "end": 1.3},
                {"word": "kaybolmaz.", "start": 1.4, "end": 1.9},
            ]

        eng = XTTSEngine(
            config=XTTSConfig(),
            thresholds=AudioThresholds(),
            synthesizer_fn=fake_synth,
            transcriber_fn=fake_whisper,
        )
        with tempfile.TemporaryDirectory() as td:
            res = eng.synthesize_sentence_v22(
                sentence_text="Orada hiçbir anı kaybolmaz.",
                reference_wav=str(Path(td) / "ref.wav"),
                temp_dir=Path(td) / "q",
                enable_atempo_compensation=True,
                enable_adaptive_profiles=True,
            )
            self.assertEqual(res.status, ArtifactStatus.APPROVED)
            self.assertEqual(len(res.history), 1)
            self.assertIn("needs_manual_review", res.history[0])
            self.assertTrue(res.history[0]["passed"])
            self.assertFalse(res.history[0]["needs_manual_review"])

    def test_v3_03_legacy_protection_guard_per_file_sha256(self) -> None:
        """[REAL_LOCAL] Tekil dosya SHA-256 koruma bekçisi değişikliği yakalar."""
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "live_manifest.json"
            p2 = Path(td) / "approved.wav"
            p1.write_text('{"ok": true}', encoding="utf-8")
            p2.write_bytes(b"WAVDATA")
            guard = LegacyProtectionGuard({"manifest": p1, "wav": p2})
            guard.snapshot_before(strict_missing=True)
            rep_ok = guard.verify_after()
            self.assertTrue(rep_ok["all_protected_ok"])
            self.assertEqual(rep_ok["checked_count"], 2)

            p2.write_bytes(b"TAMPERED")
            rep_bad = guard.verify_after()
            self.assertFalse(rep_bad["all_protected_ok"])
            self.assertEqual(len(rep_bad["mismatches"]), 1)

    def test_v3_04_audioldm_ambience_catalog_and_timeline(self) -> None:
        """[REAL_LOCAL_DSP+SIMULATED_AUDIOLDM] 10 ambiyans profili, kesintisiz döngü ve anlatıcı dostu seviye."""
        for required_key in ("yagmur", "ruzgar", "deniz", "dalga", "somine", "orman", "gece_bocekleri", "uzak_gok_gurultusu"):
            self.assertIn(required_key, AMBIENCE_CATALOG)

        with tempfile.TemporaryDirectory() as td:
            eng = AudioLDMAmbienceEngine(Path(td) / "amb")
            tl = eng.render_timeline_ambience(
                segments=[
                    {"ambience_key": "yagmur", "duration_sec": 4.0},
                    {"ambience_key": "somine", "duration_sec": 4.0, "boost_db": 2.0},
                ],
                out_wav=Path(td) / "timeline_amb.wav",
                allow_synthetic_test_fallback=True,
                device="cpu",
            )
            self.assertTrue(Path(tl["output_path"]).exists())
            self.assertTrue(tl["narrator_safe"])
            self.assertLessEqual(tl["rms_db"], -24.0)
            self.assertEqual(len(tl["sha256"]), 64)

    def test_v3_05_visual_continuity_and_opencv_anomalies(self) -> None:
        """[REAL_LOCAL] Görsel devamlılık rehberi ve OpenCV siyah/boş/tekrar kare tespiti."""
        bible = VisualStyleBible(
            characters={
                "Arin": CharacterProfile("Arin", "late 20s", "dark wavy hair", "thick dark wool coat", "calm eyes")
            }
        )
        cm = VisualContinuityManager(bible)
        prompt_data = cm.build_scene_prompt(
            "sahne_01", "Misty midnight forest footpath", characters_present=["Arin"],
            objects_present=["pirinc_anahtar"], allow_characters=True,
        )
        self.assertIn("Arin", prompt_data["positive_prompt"])
        self.assertIn("brass key", prompt_data["positive_prompt"])

        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "s1.png"
            p2 = Path(td) / "s2.png"
            rng = np.random.default_rng(7)
            img1 = np.clip(rng.normal(40, 12, (144, 256, 3)), 0, 255).astype(np.uint8)
            img2 = np.clip(rng.normal(44, 12, (144, 256, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(img1).save(p1)
            Image.fromarray(img2).save(p2)
            cont = cm.evaluate_cross_scene_continuity([p1, p2])
            self.assertTrue(cont["passed"])

            # Siyah ve tekrarlı kare yakalama
            black_frames = [Image.new("RGB", (256, 144), (0, 0, 0)) for _ in range(5)]
            bad_audit = audit_extra_frame_anomalies(black_frames)
            self.assertFalse(bad_audit["passed"])
            self.assertEqual(bad_audit["black_frames"], 5)

            good_frames = [
                Image.fromarray(np.clip(rng.normal(40 + i * 2, 18, (144, 256, 3)), 0, 255).astype(np.uint8))
                for i in range(5)
            ]
            ok_audit = audit_extra_frame_anomalies(good_frames)
            self.assertTrue(ok_audit["passed"])

    def test_v3_06_125_paragraph_script_dry_run_integration(self) -> None:
        """[REAL_LOCAL] Gerçek 125 paragraflık senaryo ile v3_runner --dry-run doğrulaması."""
        script_p = Path("/working_dir/c_7b28fca70930cf34/user_uploaded_files/ttl=63d/user_uploaded/Unutulmuş_Zamanlar_Arşivi_-_Kusursuz_Final_v2.txt")
        if not script_p.exists():
            self.skipTest("Senaryo dosyası bu ortamda bulunamadı.")
        runner_p = Path(__file__).resolve().parents[2] / "v3_runner.py"
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "v3_out"
            rc = subprocess.run(
                [
                    "python3", str(runner_p),
                    "--script", str(script_p),
                    "--output-dir", str(out_dir),
                    "--local-work", str(Path(td) / "work"),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(rc.returncode, 0, rc.stderr + rc.stdout)
            reps = list((out_dir / "raporlar").glob("v3_*dry_run_*.json"))
            self.assertEqual(len(reps), 1)
            rep = json.loads(reps[0].read_text(encoding="utf-8"))
            self.assertEqual(rep["paragraph_count"], 125)
            self.assertGreater(rep["sentence_count"], 400)

    def test_v3_07_real_30_minute_ffmpeg_e2e_assembly_and_integrity(self) -> None:
        """[REAL_LOCAL_FFMPEG_30MIN] Gerçek 30 dakikalık (1800 sn) 1080p/24fps MP4 + AAC + 160 sn siyah kapanış + SRT bütünlük testi."""
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # 1. 30 dakikalık (1800 sn) gerçek AAC ses üret (düşük seviyeli sakin ton, FFmpeg lavfi)
            audio_m4a = root / "master_30min.m4a"
            audio_10s = root / "master_10s.m4a"
            subprocess.check_call(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "sine=frequency=110:sample_rate=24000:duration=10",
                    "-af", "volume=-24dB", "-c:a", "aac", "-b:a", "128k", str(audio_10s),
                ]
            )
            subprocess.check_call(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-stream_loop", "179", "-i", str(audio_10s),
                    "-t", "1800", "-c:a", "copy", str(audio_m4a),
                ]
            )
            # 2. 1080p/24fps 10 sn sahne klibini hızlı stream-copy loop ile 1640 sn sahne + 160 sn saf siyah kapanış = 1800 sn
            clip_10s = root / "scene_10s.mp4"
            subprocess.check_call(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=#101824:s=1920x1080:r=24:d=10",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
                    "-threads", "2", str(clip_10s),
                ]
            )
            scene_1640s = root / "scene_1640s.mp4"
            subprocess.check_call(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-stream_loop", "163", "-i", str(clip_10s),
                    "-t", "1640", "-c:v", "copy", str(scene_1640s),
                ]
            )
            outro_160s = root / "outro_160s.mp4"
            subprocess.check_call(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=#000000:s=1920x1080:r=24:d=10",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p",
                    "-threads", "2", str(root / "black_10s.mp4"),
                ]
            )
            subprocess.check_call(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-stream_loop", "15", "-i", str(root / "black_10s.mp4"),
                    "-t", "160", "-c:v", "copy", str(outro_160s),
                ]
            )

            # 3. FFmpegAssemblyEngine ile 30 dk final MP4 birleştir
            asm = FFmpegAssemblyEngine(local_work_dir=root / "work", ffmpeg_threads=2)
            final_mp4 = root / "final_30min_1080p.mp4"
            asm.assemble_final_mp4([scene_1640s, outro_160s], audio_m4a, final_mp4)

            # 4. Baştan sona senkronizasyon, 1080p/24fps ve saf siyah outro doğrulaması
            sync_res = validate_final_assembly(
                final_mp4_path=final_mp4,
                master_audio_path=audio_m4a,
                outro_start_sec=1640.0,
                thresholds=AssemblyThresholds(),
            )
            self.assertTrue(sync_res.passed, f"30 dk final video doğrulaması başarısız: {sync_res.issues}")
            self.assertAlmostEqual(sync_res.video_duration_sec, 1800.0, delta=2.0)

            # 5. Türkçe SRT üretimi ve doğrulaması (siyah ekranda 0 hayalet altyazı)
            sent_texts = [f"Türkçe uyku hikâyesi cümlesi {i + 1}." for i in range(250)]
            cues = [
                SRTCue(i + 1, i * 6.0 + 0.5, i * 6.0 + 5.0, [sent_texts[i]])
                for i in range(250)
            ]
            srt_p = root / "final_30min.srt"
            write_srt_file(cues, srt_p)
            srt_res = validate_srt_file(
                srt_path=srt_p,
                                outro_start_sec=1640.0,
                expected_sentences=sent_texts,
                thresholds=SRTThresholds(),
            )
            self.assertTrue(srt_res.passed, f"30 dk SRT doğrulaması başarısız: {srt_res.issues}")
            self.assertEqual(len(sha256_file(final_mp4)), 64)
            _ = snapshot_system_resources(root)
            self.assertLess(time.time() - t0, 120.0)


if __name__ == "__main__":
    unittest.main()
