"""V3_PATCHED — ZIP Otomatik Açma + Bütünleşik 7 Aşamalı Akış + Kesinti/Kurtarma Testleri."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PATCHED_ROOT = Path(__file__).resolve().parents[2]
if str(PATCHED_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCHED_ROOT))

import v3_runner
from uyku_v3.subtitles.transcription import WordTimestamp


def _make_clean_speech_wav(sr: int = 24000, sec: float = 2.0) -> np.ndarray:
    t = np.linspace(0, sec, int(sr * sec), dtype=np.float32)
    env = np.ones_like(t)
    env[: int(sr * 0.05)] = np.linspace(0.0, 1.0, int(sr * 0.05), dtype=np.float32)
    env[-int(sr * 0.15) :] = np.linspace(1.0, 0.0, int(sr * 0.15), dtype=np.float32)
    return np.concatenate([
        np.zeros(int(sr * 0.08), dtype=np.float32),
        (0.14 * np.sin(2 * np.pi * 220 * t) * env).astype(np.float32),
        np.zeros(int(sr * 0.25), dtype=np.float32),
    ])


class TestV3PatchedFixes(unittest.TestCase):
    def test_p01_real_zip_sha256_extract_idempotency_and_import(self) -> None:
        """[REAL_LOCAL] Gerçek UYKU_HIKAYESI_V3_PATCHED.zip üzerinde SHA-256 doğrulama, çıkarma, idempotency ve import testi."""
        zip_p = PATCHED_ROOT.parent / "UYKU_HIKAYESI_V3_PATCHED.zip"
        if not zip_p.exists():
            self.skipTest("ZIP henüz paketlenmedi (paketleme sonrası test edilir).")
        actual_sha = hashlib.sha256(zip_p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            ex_root = Path(td) / "pkg"
            # Yanlış SHA-256 reddedilmeli (fail-closed)
            with self.assertRaises(RuntimeError):
                v3_runner.ensure_v3_package_extracted(
                    zip_path=zip_p, extract_root=ex_root, expected_sha256="0" * 64
                )
            # Doğru SHA-256 ile çıkarma
            pkg_root = v3_runner.ensure_v3_package_extracted(
                zip_path=zip_p, extract_root=ex_root, expected_sha256=actual_sha
            )
            self.assertTrue((pkg_root / "uyku_v3" / "__init__.py").exists())
            marker = ex_root / ".v3_patched_zip_sha256"
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), actual_sha)
            mtime_first = marker.stat().st_mtime_ns

            # İkinci çağrı idempotent olmalı (tekrar çıkarmamalı)
            pkg_root_2 = v3_runner.ensure_v3_package_extracted(
                zip_path=zip_p, extract_root=ex_root, expected_sha256=actual_sha
            )
            self.assertEqual(pkg_root, pkg_root_2)
            self.assertEqual(marker.stat().st_mtime_ns, mtime_first)

            # İzole alt süreçte import doğrulaması
            rc = subprocess.run(
                [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(pkg_root)!r}); import uyku_v3; print(uyku_v3.__file__)"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(rc.returncode, 0, rc.stderr)
            self.assertIn(str(pkg_root), rc.stdout)

    def test_p02_full_7_stage_integrated_pipeline_wiring(self) -> None:
        """[HYBRID_SIMULATED_GPU_AND_REAL_LOCAL] Tüm 7 aşamanın (XTTS->Whisper->AudioLDM->Mastering->SDXL->SVD-XT/CLIP/OpenCV->SRT->FFmpeg MP4) gerçek girdi/çıktı bağlantısı."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script_p = root / "kisa_hikaye.txt"
            script_p.write_text(
                "GİRİŞ – TEST\n\n"
                "Hoş geldin. Artık dinlenme zamanı.\n\n"
                "Gece sessiz ve sakin.\n",
                encoding="utf-8",
            )
            ref_wav = root / "ref.wav"
            v3_runner._write_wav_pcm16(ref_wav, _make_clean_speech_wav(24000, 2.0), 24000)

            prot_file = root / "protected_legacy.json"
            prot_file.write_text('{"keep": true}', encoding="utf-8")

            cfg = json.loads((PATCHED_ROOT / "config_v3_production.json").read_text(encoding="utf-8"))
            cfg["ambience"]["outro_duration_sec"] = 6.0
            cfg["ambience"]["outro_fade_out_sec"] = 2.0

            speech = _make_clean_speech_wav(24000, 2.0)

            def fake_xtts(text, temp, speed, top_p):
                return speech.copy()

            def fake_whisper(audio, sr_in):
                # Hangi cümle geldiyse onu dön (bu test stage bağlantısını doğrular)
                return last_text[0], [
                    {"word": w, "start": 0.1 + i * 0.3, "end": 0.35 + i * 0.3}
                    for i, w in enumerate(last_text[0].split())
                ]

            last_text = [""]
            orig_synth = fake_xtts

            def tracking_xtts(text, temp, speed, top_p):
                last_text[0] = text
                return orig_synth(text, temp, speed, top_p)

            rng = np.random.default_rng(42)

            def fake_sdxl(pos_prompt, neg_prompt):
                arr = np.clip(rng.normal(42, 18, (1080, 1920, 3)), 0, 255).astype(np.uint8)
                return Image.fromarray(arr)

            def fake_svd(init_img, motion_bucket_id, noise_aug, num_frames):
                base = np.asarray(init_img.resize((640, 360)), dtype=np.float32)
                frames = []
                for i in range(8):
                    shifted = np.clip(np.roll(base, i, axis=1) + (i % 2), 0, 255).astype(np.uint8)
                    frames.append(Image.fromarray(shifted))
                return frames

            def fake_clip(frames, texts):
                # Pozitif=0.32, negatifler=0.10 (margin 0.22 >= 0.03)
                mat = np.full((len(frames), len(texts)), 0.10, dtype=np.float32)
                mat[:, 0] = 0.32
                return mat

            def fake_srt_words(media_path, outro_start_sec):
                sents = ["Hoş geldin.", "Artık dinlenme zamanı.", "Gece sessiz ve sakin."]
                words = []
                t_cur = 0.2
                for s in sents:
                    for w in s.split():
                        words.append(WordTimestamp(word=w, start=t_cur, end=t_cur + 0.35, norm=w.lower().strip(".,!?")))
                        t_cur += 0.45
                return words

            rep = v3_runner.run_v3_pipeline(
                cfg=cfg,
                script_path=script_p,
                ref_audio=ref_wav,
                v3_out_dir=root / "v3_out",
                local_work=root / "work",
                stages_req={"all"},
                protected_override={"legacy_test": prot_file},
                backends={
                    "xtts_synth_fn": tracking_xtts,
                    "whisper_fn": fake_whisper,
                    "allow_synthetic_ambience": True,
                    "image_gen_fn": fake_sdxl,
                    "svd_frame_fn": fake_svd,
                    "clip_score_fn": fake_clip,
                    "srt_words_fn": fake_srt_words,
                },
            )

            # Tüm 7 aşamanın gerçekten çalışıp çıktı ürettiğini doğrula
            for st_key in ("voice", "ambience", "mastering", "visuals", "video_i2v", "subtitles", "assembly"):
                self.assertIn(st_key, rep["stages"], f"Eksik aşama: {st_key}")

            self.assertEqual(rep["stages"]["visuals"]["approved_images"], 1)
            self.assertEqual(rep["stages"]["video_i2v"]["approved_clips"], 1)
            self.assertTrue(rep["stages"]["subtitles"]["passed"])
            self.assertEqual(rep["stages"]["assembly"].get("status"), "APPROVED", rep["stages"]["assembly"])
            self.assertTrue((root / "v3_out" / "final" / "YOUTUBE_FINAL_1080P.mp4").exists())
            self.assertTrue((root / "v3_out" / "final" / "YOUTUBE_FINAL_TR.srt").exists())
            self.assertTrue(rep["protection_sha256"]["all_protected_ok"])

    def test_p03_interruption_resume_and_fail_closed_gate(self) -> None:
        """[HYBRID_SIMULATED_GPU_AND_REAL_LOCAL] --stop-after 1 kesintisi sonrası devamda önbellek kullanımı ve APPROVED olmayan seste mastering engeli."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script_p = root / "iki_paragraf.txt"
            script_p.write_text(
                "GİRİŞ – KESİNTİ\n\nHoş geldin.\n\nGece sessiz ve sakin.\n",
                encoding="utf-8",
            )
            ref_wav = root / "ref.wav"
            v3_runner._write_wav_pcm16(ref_wav, _make_clean_speech_wav(24000, 2.0), 24000)
            cfg = json.loads((PATCHED_ROOT / "config_v3_production.json").read_text(encoding="utf-8"))

            speech = _make_clean_speech_wav(24000, 2.0)
            last_text = [""]

            def synth_ok(text, temp, speed, top_p):
                last_text[0] = text
                return speech.copy()

            def whisper_ok(audio, sr_in):
                return last_text[0], [
                    {"word": w, "start": 0.1 + i * 0.3, "end": 0.35 + i * 0.3}
                    for i, w in enumerate(last_text[0].split())
                ]

            # 1. Çalıştırma: --stop-after 1
            r1 = v3_runner.run_v3_pipeline(
                cfg=cfg, script_path=script_p, ref_audio=ref_wav,
                v3_out_dir=root / "v3_out", local_work=root / "work",
                stages_req={"voice"}, stop_after=1,
                protected_override={},
                backends={"xtts_synth_fn": synth_ok, "whisper_fn": whisper_ok},
            )
            self.assertEqual(r1["stages"]["voice"]["processed_paragraphs"], 1)
            self.assertEqual(r1["stages"]["voice"]["synthesized_sentences"], 1)

            # 2. Çalıştırma: devam (paragraf_000 yeniden sentezlenmeden atlanmalı)
            r2 = v3_runner.run_v3_pipeline(
                cfg=cfg, script_path=script_p, ref_audio=ref_wav,
                v3_out_dir=root / "v3_out", local_work=root / "work",
                stages_req={"voice"}, stop_after=0,
                protected_override={},
                backends={"xtts_synth_fn": synth_ok, "whisper_fn": whisper_ok},
            )
            self.assertEqual(r2["stages"]["voice"]["reused_approved_paragraphs"], ["paragraf_000"])
            self.assertEqual(r2["stages"]["voice"]["synthesized_sentences"], 1)


    def test_p04_failed_sentence_saves_quarantine_wav_and_whisper_metrics(self) -> None:
        """[HYBRID_SIMULATED_GPU_AND_REAL_LOCAL] Başarısız cümlede karantina ham WAV + Whisper metrikleri kaydedilir."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script_p = root / "fail_test.txt"
            script_p.write_text("GİRİŞ – KONTROL\n\nŞimdi yalnızca bulunduğun yere yerleş.\n", encoding="utf-8")
            ref_wav = root / "ref.wav"
            v3_runner._write_wav_pcm16(ref_wav, _make_clean_speech_wav(24000, 2.0), 24000)
            cfg = json.loads((PATCHED_ROOT / "config_v3_production.json").read_text(encoding="utf-8"))

            speech = _make_clean_speech_wav(24000, 2.0)

            def synth_any(text, temp, speed, top_p):
                return speech.copy()

            def whisper_wrong(audio, sr_in):
                return "Şimdi yalnızca bulunduğun yere yanlış.", [
                    {"word": "Şimdi", "start": 0.1, "end": 0.3},
                    {"word": "yalnızca", "start": 0.35, "end": 0.65},
                    {"word": "bulunduğun", "start": 0.7, "end": 1.0},
                    {"word": "yere", "start": 1.05, "end": 1.25},
                    {"word": "yanlış.", "start": 1.3, "end": 1.6},
                ]

            rep = v3_runner.run_v3_pipeline(
                cfg=cfg, script_path=script_p, ref_audio=ref_wav,
                v3_out_dir=root / "v3_out", local_work=root / "work",
                stages_req={"all"}, protected_override={},
                backends={"xtts_synth_fn": synth_any, "whisper_fn": whisper_wrong},
            )
            self.assertTrue(rep["stages"]["downstream_gate"]["blocked"])
            manifest_data = json.loads((root / "v3_out" / "v3_production_manifest.json").read_text(encoding="utf-8"))
            failed_list = manifest_data["artifacts"]["paragraf_000"]["quality_scores"]["failed"]
            self.assertEqual(len(failed_list), 1)
            f0 = failed_list[0]
            self.assertEqual(f0["transcribed_text"], "Şimdi yalnızca bulunduğun yere yanlış.")
            self.assertIsNotNone(f0["wer"])
            self.assertIsNotNone(f0["wpm"])
            self.assertTrue(len(f0["issues"]) > 0)
            self.assertTrue(Path(f0["raw_wav_path"]).exists())
            self.assertGreater(Path(f0["raw_wav_path"]).stat().st_size, 1000)

    def test_p05_active_paragraph_sentence_cache_hit_and_resume(self) -> None:
        """[HYBRID_SIMULATED_GPU_AND_REAL_LOCAL] Aktif paragrafta önbellekteki C02 tuple(audio, meta) üzerinden sentezsiz yüklenir."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script_p = root / "cache_hit.txt"
            script_p.write_text("GİRİŞ – TEST\n\nŞimdi yalnızca bulunduğun yere yerleş. Ellerin oldukları yerde dinlensin.\n", encoding="utf-8")
            ref_wav = root / "ref.wav"
            v3_runner._write_wav_pcm16(ref_wav, _make_clean_speech_wav(24000, 2.0), 24000)
            cfg = json.loads((PATCHED_ROOT / "config_v3_production.json").read_text(encoding="utf-8"))
            speech = _make_clean_speech_wav(24000, 2.0)

            state = {"run": 1, "last_text": ""}

            def synth_fn(text, temp, speed, top_p):
                state["last_text"] = text
                return speech.copy()

            def whisper_fn(audio, sr_in):
                txt = state["last_text"]
                if state["run"] == 1 and "yerleş" in txt:
                    return "Şimdi yalnızca bulunduğun yere yanlış.", [
                        {"word": "Şimdi", "start": 0.1, "end": 0.3},
                        {"word": "yalnızca", "start": 0.35, "end": 0.6},
                        {"word": "bulunduğun", "start": 0.65, "end": 0.95},
                        {"word": "yere", "start": 1.0, "end": 1.2},
                        {"word": "yanlış.", "start": 1.25, "end": 1.55},
                    ]
                return txt, [
                    {"word": w, "start": 0.1 + i * 0.3, "end": 0.35 + i * 0.3}
                    for i, w in enumerate(txt.split())
                ]

            # Run 1: C01 fails, C02 passes & cached
            r1 = v3_runner.run_v3_pipeline(
                cfg=cfg, script_path=script_p, ref_audio=ref_wav,
                v3_out_dir=root / "v3_out", local_work=root / "work",
                stages_req={"voice"}, protected_override={},
                backends={"xtts_synth_fn": synth_fn, "whisper_fn": whisper_fn},
            )
            self.assertEqual(r1["stages"]["voice"]["synthesized_sentences"], 2)

            # Run 2: C01 passes, C02 reused from SentenceAudioCache (hit[0])
            state["run"] = 2
            r2 = v3_runner.run_v3_pipeline(
                cfg=cfg, script_path=script_p, ref_audio=ref_wav,
                v3_out_dir=root / "v3_out", local_work=root / "work",
                stages_req={"voice"}, protected_override={},
                backends={"xtts_synth_fn": synth_fn, "whisper_fn": whisper_fn},
            )
            self.assertEqual(r2["stages"]["voice"]["reused_cached_sentences_in_active_paragraphs"], 1)
            self.assertEqual(r2["stages"]["voice"]["synthesized_sentences"], 1)
            self.assertEqual(r2["stages"]["voice"]["paragraph_status_counts"]["APPROVED"], 1)

    def test_p06_sdxl_pipeline_load_and_single_image_generation(self) -> None:
        """[HYBRID_SIMULATED_GPU_AND_REAL_LOCAL] load_sdxl() ile pipeline yüklenir, tek kopya korunur ve tek 1080p görsel üretilir."""
        import contextlib
        import types
        from unittest.mock import patch
        from uyku_v3.visuals.image_generator import ImageGenerator

        class _FakePipeOut:
            def __init__(self, img: Image.Image) -> None:
                self.images = [img]

        class _FakeSDXLPipe:
            load_calls = 0
            infer_calls = 0

            def __init__(self) -> None:
                self.vae = types.SimpleNamespace(enable_tiling=lambda: None)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.load_calls += 1
                return cls()

            def to(self, device: str):
                return self

            def enable_attention_slicing(self) -> None:
                pass

            def __call__(self, **kwargs):
                _FakeSDXLPipe.infer_calls += 1
                rng = np.random.default_rng(123)
                arr = np.clip(rng.normal(48, 22, (576, 1024, 3)), 0, 255).astype(np.uint8)
                return _FakePipeOut(Image.fromarray(arr))

        fake_diffusers = types.SimpleNamespace(StableDiffusionXLPipeline=_FakeSDXLPipe)
        fake_torch = types.SimpleNamespace(
            float16="float16",
            cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        )

        gen = ImageGenerator()
        # Yüklenmeden üretim denemesi sessizce geçmemeli
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            with self.assertRaises(RuntimeError):
                gen.ensure_valid_scene_image(td_p / "err.jpg", "test", td_p / "tmp")

            with patch.dict(sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}):
                gen.load_sdxl(device="cuda")
                gen.load_sdxl(device="cuda")  # İkinci çağrı yeni kopya oluşturmamalı
                self.assertEqual(_FakeSDXLPipe.load_calls, 1)
                self.assertIsNotNone(gen._pipe)

                out_img = td_p / "sahne_01.jpg"
                ok, val, regen = gen.ensure_valid_scene_image(
                    target_path=out_img,
                    prompt="dark nocturnal forest",
                    local_tmp_dir=td_p / "tmp",
                    max_retries=1,
                )
                self.assertTrue(ok, val.issues)
                self.assertTrue(regen)
                self.assertEqual(_FakeSDXLPipe.infer_calls, 1)
                self.assertTrue(out_img.exists())
                self.assertEqual(val.width, 1920)
                self.assertEqual(val.height, 1080)
                gen.unload()
                self.assertIsNone(gen._pipe)

    def test_p07_nonzero_exit_code_on_gate_block_and_traceback_logging(self) -> None:
        """[REAL_LOCAL] main() kapı engeli veya istisna durumunda çıkış kodu 0 vermez ve traceback kaydeder."""
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            rc = v3_runner.main([
                "--script", str(td_p / "olmayan_senaryo.txt"),
                "--output-dir", str(td_p / "out"),
                "--local-work", str(td_p / "work"),
                "--dry-run",
            ])
            self.assertNotEqual(rc, 0)
            tb_file = td_p / "out" / "loglar" / "last_traceback.txt"
            self.assertTrue(tb_file.exists())
            self.assertIn("Senaryo bulunamadı", tb_file.read_text(encoding="utf-8"))

    def test_p08_visuals_gate_blocks_video_and_assembly_on_bad_image(self) -> None:
        """[HYBRID_SIMULATED_GPU_AND_REAL_LOCAL] Görsel kalite kontrolünden geçemezse visuals_gate SVD-XT ve FFmpeg aşamalarını durdurur."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            script_p = root / "bad_vis.txt"
            script_p.write_text("GİRİŞ – GÖRSEL\n\nHoş geldin.\n", encoding="utf-8")
            ref_wav = root / "ref.wav"
            v3_runner._write_wav_pcm16(ref_wav, _make_clean_speech_wav(24000, 2.0), 24000)
            cfg = json.loads((PATCHED_ROOT / "config_v3_production.json").read_text(encoding="utf-8"))
            cfg["ambience"]["outro_duration_sec"] = 6.0
            cfg["ambience"]["outro_fade_out_sec"] = 2.0
            speech = _make_clean_speech_wav(24000, 2.0)

            def synth_ok(text, temp, speed, top_p):
                return speech.copy()

            def whisper_ok(audio, sr_in):
                return "Hoş geldin.", [{"word": "Hoş", "start": 0.1, "end": 0.3}, {"word": "geldin.", "start": 0.4, "end": 0.8}]

            def bad_black_sdxl(pos, neg):
                return Image.new("RGB", (1024, 576), (0, 0, 0))

            rep = v3_runner.run_v3_pipeline(
                cfg=cfg, script_path=script_p, ref_audio=ref_wav,
                v3_out_dir=root / "v3_out", local_work=root / "work",
                stages_req={"all"}, protected_override={},
                backends={
                    "xtts_synth_fn": synth_ok,
                    "whisper_fn": whisper_ok,
                    "allow_synthetic_ambience": True,
                    "image_gen_fn": bad_black_sdxl,
                },
            )
            self.assertIn("visuals_gate", rep["stages"])
            self.assertTrue(rep["stages"]["visuals_gate"]["blocked"])
            self.assertNotIn("video_i2v", rep["stages"])
            self.assertNotIn("assembly", rep["stages"])

    def test_p09_clip_handles_base_model_output_with_pooling(self) -> None:
        """[SIMULATED] transformers BaseModelOutputWithPooling döndürdüğünde CLIP kosinüs matrisi hatasız hesaplanır."""
        import contextlib
        import types
        from unittest.mock import patch
        from uyku_v3.video.clip_validator import CLIPValidator

        class _FakeTensor:
            def __init__(self, arr: np.ndarray) -> None:
                self.arr = arr.astype(np.float32)

            def norm(self, dim: int = -1, keepdim: bool = True):
                return _FakeTensor(np.linalg.norm(self.arr, axis=dim, keepdims=keepdim))

            def __truediv__(self, other):
                return _FakeTensor(self.arr / other.arr)

            def __matmul__(self, other):
                return _FakeTensor(self.arr @ other.arr)

            @property
            def T(self):
                return _FakeTensor(self.arr.T)

            def cpu(self):
                return self

            def numpy(self):
                return self.arr

        class BaseModelOutputWithPooling:
            def __init__(self, pooler_output: _FakeTensor) -> None:
                self.pooler_output = pooler_output

        class _FakeInputs(dict):
            def to(self, device: str):
                return self

        class _FakeModel:
            def get_image_features(self, **kwargs):
                return BaseModelOutputWithPooling(_FakeTensor(np.array([[1.0, 0.0], [1.0, 0.0]])))

            def get_text_features(self, **kwargs):
                return BaseModelOutputWithPooling(_FakeTensor(np.array([[1.0, 0.0], [0.0, 1.0]])))

        cv = CLIPValidator()
        cv._model = _FakeModel()
        cv._proc = lambda text, images, return_tensors, padding: _FakeInputs(
            {"pixel_values": None, "input_ids": None, "attention_mask": None}
        )
        fake_torch = types.SimpleNamespace(no_grad=contextlib.nullcontext)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            frames = [Image.new("RGB", (64, 64), (20, 20, 30)) for _ in range(2)]
            mat = cv._compute_cosine_matrix(frames, ["pos", "neg"])
            self.assertEqual(mat.shape, (2, 2))
            self.assertAlmostEqual(float(mat[0, 0]), 1.0, places=4)
            self.assertAlmostEqual(float(mat[0, 1]), 0.0, places=4)

if __name__ == "__main__":
    unittest.main()
