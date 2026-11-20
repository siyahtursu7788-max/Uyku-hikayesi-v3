"""
Bölüm 15 Görsel ve Video Test Paketi (Test 10–21).
OpenCV, PIL, FFmpeg döngü ve durum makinesi testleri gerçek yerel motorla çalışır;
SDXL, SVD-XT ve CLIP GPU çağrıları açıkça SIMULATED_GPU enjeksiyonuyla doğrulanır.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from uyku_v3.visuals.image_validator import validate_image_file
from uyku_v3.visuals.image_generator import ImageGenerator
from uyku_v3.visuals.scene_manager import SceneSpec
from uyku_v3.video.clip_validator import CLIPValidator
from uyku_v3.video.motion_validator import analyze_video_frames
from uyku_v3.video.loop_engine import build_seamless_loop
from uyku_v3.video.svd_engine import SVDEngine
from uyku_v3.core.state_manager import ArtifactStatus


def _make_textured_dark_image(w: int = 1920, h: int = 1080, brightness: int = 45, shift: int = 0) -> Image.Image:
    """Uyku atmosferine uygun dokulu karanlık sinematik test görüntüsü üretir."""
    yy, xx = np.mgrid[0:h, 0:w]
    pattern = ((np.sin((xx + shift) * 0.04) * 18 + np.cos((yy + shift) * 0.05) * 18) + brightness)
    rng = np.random.default_rng(42 + shift)
    noise = rng.normal(0, 6.0, size=(h, w))
    r = np.clip(pattern + noise + 8, 0, 255).astype(np.uint8)
    g = np.clip(pattern + noise, 0, 255).astype(np.uint8)
    b = np.clip(pattern + noise - 4, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


def _make_smooth_video_frames(n: int = 12, w: int = 320, h: int = 180) -> List[Image.Image]:
    base = np.array(_make_textured_dark_image(w, h, brightness=50), dtype=np.float32)
    frames: List[Image.Image] = []
    for i in range(n):
        shifted = np.roll(base, shift=i * 2, axis=1)
        frames.append(Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8)))
    return frames


class TestVisualsAndVideo(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_10_single_image_generation_and_validation(self) -> None:
        """[SIMULATED_GPU+REAL_LOCAL] Tek görsel üretim ve 1920x1080 geçerlilik testi."""
        gen = ImageGenerator(generator_fn=lambda p, np_: _make_textured_dark_image(1024, 576, brightness=48))
        out_path = self.tmp_dir / "sahne_01.jpg"
        passed, val, regen = gen.ensure_valid_scene_image(out_path, "dark stone cabin", self.tmp_dir / "work")
        self.assertTrue(passed, f"Görsel hatası: {val.issues}")
        self.assertTrue(regen)
        self.assertEqual((val.width, val.height), (1920, 1080))

        # İkinci çağrıda geçerli görseli yeniden üretmemeli
        passed2, _, regen2 = gen.ensure_valid_scene_image(out_path, "dark stone cabin", self.tmp_dir / "work")
        self.assertTrue(passed2)
        self.assertFalse(regen2)

    def test_11_wrong_aspect_ratio_rejected(self) -> None:
        """[REAL_LOCAL] Yanlış en-boy oranı (4:3) reddedilmeli."""
        bad_img = self.tmp_dir / "bad_4_3.jpg"
        _make_textured_dark_image(1024, 768, brightness=48).save(bad_img, quality=95)
        val = validate_image_file(bad_img)
        self.assertFalse(val.passed)
        self.assertTrue(any("en-boy oranı" in x.lower() for x in val.issues))

    def test_12_single_scene_svd_xt_simulated_gpu(self) -> None:
        """[SIMULATED_GPU+REAL_LOCAL] Tek sahne SVD-XT üretimi + CLIP/OpenCV/FFmpeg onay testi."""
        src_img = self.tmp_dir / "src.jpg"
        frames = _make_smooth_video_frames(10, 320, 180)
        frames[0].resize((1920, 1080)).save(src_img, quality=95)

        good_clip = CLIPValidator(score_fn=lambda fr, tx: np.tile([0.34, 0.10, 0.08, 0.09, 0.07], (len(fr), 1)))
        engine = SVDEngine(
            clip_validator=good_clip,
            frame_generator_fn=lambda im, mb, na, sd: frames,
        )
        scene = SceneSpec(0, "src.jpg", "P00", "P01", "indoor_calm", "dark cabin", "prompt")
        out_mp4 = self.tmp_dir / "klip_00.mp4"
        res = engine.generate_and_validate_clip(scene, src_img, out_mp4, self.tmp_dir / "work")
        self.assertEqual(res.status, ArtifactStatus.APPROVED, f"Hata: {res.error_summary}")
        self.assertTrue(out_mp4.exists())

    def test_13_clip_wrong_scene_rejected(self) -> None:
        """[SIMULATED_GPU] CLIP yanlış sahne (düşük pozitif kosinüs) reddi testi."""
        frames = _make_smooth_video_frames(6)
        validator = CLIPValidator(score_fn=lambda fr, tx: np.tile([0.12, 0.14, 0.11, 0.10, 0.09], (len(fr), 1)))
        res = validator.validate_frames(frames, "dark basalt stone cabin")
        self.assertFalse(res.passed)

    def test_14_clip_human_figure_negative_risk_rejected(self) -> None:
        """[SIMULATED_GPU] İnsan yüzü/figürü negatif kavram riski reddi testi."""
        frames = _make_smooth_video_frames(6)
        # 2. sütun (indeks 2): 'human face portrait or person figure' yüksek
        validator = CLIPValidator(score_fn=lambda fr, tx: np.tile([0.28, 0.10, 0.31, 0.09, 0.08], (len(fr), 1)))
        res = validator.validate_frames(frames, "cozy empty stone cabin")
        self.assertFalse(res.passed)
        self.assertIn("human", res.worst_negative_concept)

    def test_15_excessive_brightness_rejected(self) -> None:
        """[REAL_LOCAL] Uyku için aşırı parlak gündüz sahnesi reddi testi."""
        bright_frames = [_make_textured_dark_image(320, 180, brightness=210, shift=i * 2) for i in range(6)]
        res = analyze_video_frames(bright_frames, bright_frames[0])
        self.assertFalse(res.passed)
        self.assertGreater(res.mean_brightness, 140.0)

    def test_16_flicker_detection(self) -> None:
        """[REAL_LOCAL] Ani parlaklık sıçraması (flicker) tespiti testi."""
        frames = _make_smooth_video_frames(6)
        frames[3] = _make_textured_dark_image(320, 180, brightness=115)  # +65 ani flaş
        res = analyze_video_frames(frames, frames[0])
        self.assertFalse(res.passed)
        self.assertGreater(res.max_flicker_delta, 9.5)

    def test_17_blur_detection(self) -> None:
        """[REAL_LOCAL] Karelerde bulanıklaşma (Laplacian keskinlik düşüşü) tespiti testi."""
        frames = _make_smooth_video_frames(6)
        # Orta kareyi tamamen bulanıklaştır (tek renk)
        frames[3] = Image.fromarray(np.full((180, 320, 3), 50, dtype=np.uint8))
        res = analyze_video_frames(frames, frames[0])
        self.assertFalse(res.passed)
        self.assertLess(res.sharpness_ratio, 0.45)

    def test_18_morphing_drift_detection(self) -> None:
        """[REAL_LOCAL] Aşırı şekil bozulması / morphing drift tespiti testi."""
        src = _make_textured_dark_image(320, 180, brightness=45)
        drifted = [src] + [_make_textured_dark_image(320, 180, brightness=105, shift=i * 15) for i in range(1, 6)]
        res = analyze_video_frames(drifted, src)
        self.assertFalse(res.passed)
        self.assertGreater(res.morphing_drift, 26.0)

    def test_19_three_failed_svd_attempts_never_approved(self) -> None:
        """[SIMULATED_GPU+REAL_LOCAL] P0-03 testi: 3 başarısız video denemesi ASLA kaydedilmez ve MANUAL_REVIEW olur."""
        src_img = self.tmp_dir / "src_fail.jpg"
        bad_frames = [_make_textured_dark_image(320, 180, brightness=210) for _ in range(6)]
        bad_frames[0].resize((1920, 1080)).save(src_img, quality=95)

        bad_clip = CLIPValidator(score_fn=lambda fr, tx: np.tile([0.10, 0.35, 0.30, 0.28, 0.25], (len(fr), 1)))
        engine = SVDEngine(
            clip_validator=bad_clip,
            frame_generator_fn=lambda im, mb, na, sd: bad_frames,
        )
        scene = SceneSpec(1, "src_fail.jpg", "P00", "P01", "indoor_calm", "dark room", "prompt")
        out_mp4 = self.tmp_dir / "klip_01_fail.mp4"
        res = engine.generate_and_validate_clip(scene, src_img, out_mp4, self.tmp_dir / "work")
        self.assertEqual(res.status, ArtifactStatus.MANUAL_REVIEW)
        self.assertEqual(res.attempts, 3)
        self.assertFalse(out_mp4.exists(), "Başarısız video klibi hedef MP4 olarak ASLA kaydedilmemeli!")

    def test_20_ping_pong_loop_generation(self) -> None:
        """[REAL_LOCAL] Simetrik iç mekan sahnesi için ping-pong döngü testi."""
        frames = _make_smooth_video_frames(8, 320, 180)
        out = self.tmp_dir / "loop_pingpong.mp4"
        res = build_seamless_loop(frames, out, self.tmp_dir / "w1", loop_mode="ping_pong")
        self.assertTrue(res.passed, f"Ping-pong hatası: {res.issues}")
        self.assertEqual(res.loop_frames, 14)  # 8 + 6
        self.assertTrue(out.exists())

    def test_21_directional_motion_forward_crossfade_loop(self) -> None:
        """[REAL_LOCAL] Ateş/dalga/yağmur gibi yönlü hareketler için tersine dönmeyen ileri döngü testi."""
        frames = _make_smooth_video_frames(12, 320, 180)
        out = self.tmp_dir / "loop_forward.mp4"
        res = build_seamless_loop(frames, out, self.tmp_dir / "w2", loop_mode="forward_crossfade")
        self.assertTrue(res.passed, f"Forward crossfade hatası: {res.issues}")
        self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
