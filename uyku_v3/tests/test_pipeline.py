"""
Bölüm 15 Montaj, Senkron, Siyah Ekran, Decode Testleri (Test 22–25, 34) +
Bölüm 18 Aşama 8: Küçük Örnek Türkçe Uyku Hikâyesi ile Uçtan Uca (E2E) Entegrasyon Testi.
Gerçek FFmpeg, OpenCV, PIL, NumPy/SciPy ile 1920x1080 24fps H.264/AAC MP4,
gerçek SRT altyazı ve gerçek Bölüm 17 Son Kalite Raporu üretir ve doğrular!
"""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from uyku_v3.audio.mastering import MasteringEngine
from uyku_v3.audio.pronunciation_dictionary import PronunciationDictionary
from uyku_v3.audio.text_normalizer import normalize_for_tts, tr_norm_tokens
from uyku_v3.audio.xtts_engine import XTTSEngine
from uyku_v3.assembly.ffmpeg_engine import FFmpegAssemblyEngine
from uyku_v3.assembly.sync_validator import validate_final_assembly, verify_black_screen_pure
from uyku_v3.core.manifest import ManifestManager
from uyku_v3.core.state_manager import ArtifactStatus
from uyku_v3.reports.quality_report import build_final_quality_report
from uyku_v3.subtitles.alignment import align_sentences_to_words
from uyku_v3.subtitles.srt_generator import generate_srt_cues, write_srt_file
from uyku_v3.subtitles.srt_validator import validate_srt_file
from uyku_v3.subtitles.transcription import WordTimestamp
from uyku_v3.video.clip_validator import CLIPValidator
from uyku_v3.video.svd_engine import SVDEngine
from uyku_v3.video.video_validator import validate_video_file
from uyku_v3.visuals.image_generator import ImageGenerator
from uyku_v3.visuals.scene_manager import SceneSpec


def _write_wav_pcm16(path: Path, audio: np.ndarray, sr: int = 24000) -> None:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _dark_scene_img(w: int = 640, h: int = 360, shift: int = 0) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    pat = np.sin((xx + shift) * 0.04) * 16 + np.cos((yy + shift) * 0.05) * 16 + 44
    rng = np.random.default_rng(100 + shift)
    noise = rng.normal(0, 9.0, size=(h, w))
    rgb = np.stack([
        np.clip(pat + noise + 6, 0, 255),
        np.clip(pat + noise, 0, 255),
        np.clip(pat + noise - 4, 0, 255),
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _speech_wav(dur: float = 4.2, sr: int = 24000) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
    s = 0.14 * (0.6 * np.sin(2 * np.pi * 170 * t) + 0.4 * np.sin(2 * np.pi * 340 * t))
    pad = np.zeros(int(0.12 * sr), dtype=np.float32)
    fo = int(0.05 * sr)
    s[:fo] *= np.linspace(0, 1, fo, dtype=np.float32)
    s[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
    return np.concatenate([pad, s, pad]).astype(np.float32)


class TestAssemblyAndE2EPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.work = self.root / "local_ssd"
        self.engine = FFmpegAssemblyEngine(self.work, width=1920, height=1080, fps=24)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_22_corrupt_mp4_rejected(self) -> None:
        """[REAL_LOCAL] Bozuk MP4 dosyası ffprobe + decode kontrolünden geçememeli."""
        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"NOT_A_REAL_MP4_STREAM" * 1500)
        res = validate_video_file(corrupt)
        self.assertFalse(res.passed)
        self.assertFalse(res.decode_ok)

    def test_23_missing_scene_blocks_assembly(self) -> None:
        """[REAL_LOCAL] Eksik sahne varsa FFmpeg montajı başlamadan hata vermeli."""
        missing = self.root / "olmayan_sahne.mp4"
        audio = self.root / "audio.wav"
        _write_wav_pcm16(audio, _speech_wav(2.0))
        with self.assertRaises(FileNotFoundError):
            self.engine.assemble_final_mp4([missing], audio, self.root / "out.mp4")

    def test_24_25_34_black_screen_av_sync_and_full_decode(self) -> None:
        """[REAL_LOCAL] Saf #000000 siyah ekran, ses-video senkronu ve tam akış decode testi."""
        black_mp4 = self.root / "black.mp4"
        self.engine.render_black_outro(3.0, black_mp4)
        ok_black, px_mean = verify_black_screen_pure(black_mp4, sample_time_sec=1.5)
        self.assertTrue(ok_black, f"Siyah ekran piksel ortalaması yüksek: {px_mean}")

        audio_wav = self.root / "audio_3s.wav"
        _write_wav_pcm16(audio_wav, np.zeros(24000 * 3, dtype=np.float32))
        final_mp4 = self.root / "final_3s.mp4"
        self.engine.assemble_final_mp4([black_mp4], audio_wav, final_mp4)

        val = validate_final_assembly(final_mp4, audio_wav, outro_start_sec=0.5)
        self.assertTrue(val.passed, f"Montaj hataları: {val.issues}")
        self.assertTrue(val.decode_ok)
        self.assertTrue(val.black_screen_verified)
        self.assertLessEqual(val.sync_diff_sec, 0.35)

    def test_e2e_small_turkish_sleep_story_pipeline(self) -> None:
        """
        [E2E_INTEGRATION: REAL_LOCAL_DSP_FFMPEG + SIMULATED_GPU_MODELS]
        Küçük 2 cümlelik Türkçe uyku hikâyesi ile uçtan uca boru hattı testi:
        Metin Normalizasyon -> Ses QC -> Mastering -> Görsel QC -> SVD Klip QC ->
        FFmpeg 1080p Montaj + Siyah Outro -> SRT + Doğrulama -> Son Kalite Raporu (APPROVED).
        """
        story_sentences = [
            "Arin karanlık taş kulübede yağmur sesini dinliyordu.",
            "Liora eski bakır çaydanlığı yavaşça ocağa bıraktı.",
        ]
        _ = ManifestManager(self.root / "manifest_e2e.json", "e2e_sample_story")
        pdict = PronunciationDictionary()

        # 1. Ses Üretimi & QC (Simüle XTTS/Whisper + Gerçek Akustik & Metin QC)
        xtts = XTTSEngine(
            synthesizer_fn=lambda txt, t, s, p: _speech_wav(4.2),
            transcriber_fn=lambda wav, sr: ("", []),
            known_proper_nouns=["Arin", "Liora"],
        )
        approved_wavs = []
        for idx, s in enumerate(story_sentences):
            _ = normalize_for_tts(s, pdict)
            xtts._transcriber_fn = lambda wav, sr, _s=s: (_s, [{"word": _s.split()[-1].strip("."), "start": 3.4, "end": 4.1}])
            res = xtts.synthesize_sentence_with_qc(s)
            self.assertEqual(res.status, ArtifactStatus.APPROVED, f"Cümle {idx} başarısız: {res.validation.issues}")
            approved_wavs.append(res.audio)

        # 2. Mastering (Gerçek SciPy DSP + 4x True Peak)
        master_eng = MasteringEngine(sr=24000)
        combined_vocal = np.concatenate(approved_wavs)
        master_audio, m_rep = master_eng.mix_and_master(
            chapter_vocals=[("KAYIT_01", combined_vocal)],
            ambience_bed=_speech_wav(2.0) * 0.1,
            outro_sec=3.0,
            outro_fade_sec=1.5,
        )
        self.assertTrue(m_rep.passed, f"Mastering hatası: {m_rep.issues}")
        master_wav_path = self.root / "e2e_master.wav"
        _write_wav_pcm16(master_wav_path, master_audio, sr=24000)

        # 3. Görsel Üretimi & QC (Gerçek PIL + OpenCV 1920x1080 doğrulama)
        img_gen = ImageGenerator(generator_fn=lambda p, np_: _dark_scene_img(1024, 576))
        img_path = self.root / "sahne_01.jpg"
        img_ok, img_val, _ = img_gen.ensure_valid_scene_image(img_path, "dark stone cabin", self.work)
        self.assertTrue(img_ok, f"Görsel hatası: {img_val.issues}")

        # 4. SVD-XT Video Klip & QC (Gerçek OpenCV + FFmpeg Loop + Decode)
        base = np.array(_dark_scene_img(320, 180), dtype=np.float32)
        frames = [Image.fromarray(np.clip(np.roll(base, i * 2, axis=1), 0, 255).astype(np.uint8)) for i in range(8)]
        clip_val = CLIPValidator(score_fn=lambda fr, tx: np.tile([0.33, 0.10, 0.08, 0.09, 0.07], (len(fr), 1)))
        svd = SVDEngine(clip_validator=clip_val, frame_generator_fn=lambda im, mb, na, sd: frames)
        scene = SceneSpec(0, "sahne_01.jpg", "P00", "OUTRO", "indoor_calm", "dark stone cabin", "prompt")
        klip_path = self.root / "klip_00.mp4"
        clip_res = svd.generate_and_validate_clip(scene, img_path, klip_path, self.work)
        self.assertEqual(clip_res.status, ArtifactStatus.APPROVED, f"Klip hatası: {clip_res.error_summary}")

        # 5. FFmpeg 1080p Montaj + Saf Siyah Ekran Outro
        scene_00 = self.root / "sahne_00.mp4"
        outro_mp4 = self.root / "sahne_99_siyah.mp4"
        self.engine.render_extended_scene(klip_path, m_rep.outro_start_sec, scene_00)
        self.engine.render_black_outro(3.0, outro_mp4)
        final_mp4 = self.root / "e2e_final.mp4"
        self.engine.assemble_final_mp4([scene_00, outro_mp4], master_wav_path, final_mp4)
        asm_val = validate_final_assembly(final_mp4, master_wav_path, outro_start_sec=m_rep.outro_start_sec)
        self.assertTrue(asm_val.passed, f"Montaj hatası: {asm_val.issues}")

        # 6. Türkçe SRT Üretimi + Bağımsız Doğrulama
        whisper_words = []
        cursor = 0.2
        for sent in story_sentences:
            for w in sent.split():
                norm = tr_norm_tokens(w)[0]
                whisper_words.append(WordTimestamp(w, norm, cursor, cursor + 0.42))
                cursor += 0.45
            cursor += 0.35
        aligned = align_sentences_to_words(story_sentences, whisper_words, outro_start_sec=m_rep.outro_start_sec)
        cues = generate_srt_cues(aligned, outro_start_sec=m_rep.outro_start_sec)
        srt_path = self.root / "e2e_final.srt"
        write_srt_file(cues, srt_path)
        srt_val = validate_srt_file(srt_path, expected_sentences=story_sentences, outro_start_sec=m_rep.outro_start_sec)
        self.assertTrue(srt_val.passed, f"SRT hatası: {srt_val.issues}")

        # 7. Bölüm 17 Son Kalite Raporu
        final_report = build_final_quality_report(
            project_id="e2e_sample_story",
            audio_metrics={
                "passed": True,
                "paragraph_count": 1,
                "approved_sentences": 2,
                "retried_sentences": 0,
                "manual_review_count": 0,
                "whisper_match_ratio": 1.0,
                "missing_word_checks": "OK",
                "loudness_db": m_rep.integrated_rms_db,
                "true_peak_dbtp": m_rep.true_peak_dbtp,
                "duration_sec": m_rep.duration_sec,
            },
            visual_metrics={
                "passed": True,
                "total_scenes": 1,
                "approved_images": 1,
                "regenerated_images": 1,
                "resolution": "1920x1080",
            },
            video_metrics={
                "passed": True,
                "generated_clips": 1,
                "approved_clips": 1,
                "rejected_clips": 0,
                "manual_review_count": 0,
                "mean_motion": clip_res.motion_report.mean_motion,
                "max_flicker": clip_res.motion_report.max_flicker_delta,
                "min_sharpness_ratio": clip_res.motion_report.sharpness_ratio,
                "loop_check": "OK",
            },
            assembly_metrics=asm_val.to_dict(),
            subtitle_metrics=srt_val.to_dict(),
            output_dir=self.root / "reports",
        )
        self.assertEqual(final_report["final_status"], ArtifactStatus.APPROVED.value)


if __name__ == "__main__":
    unittest.main()
