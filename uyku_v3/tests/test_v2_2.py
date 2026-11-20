"""
V2.2 Bağımsız Otomatik Test Paketi:
- Cümle bazlı kalıcı WAV + SHA-256 önbelleği ve kesintiden devam testi
- Uzun cümlelerde virgül/nefes grubu akıllı bölme (orijinal metin korunur)
- Kontrollü otomatik atempo hız telafisi (telaffuz hatasını gizlemez)
- Uyarlanabilir XTTS yeniden deneme profilleri (fail-closed)
- Özel isim farklarının otomatik APPROVED olmaması (#5 kapsam dışı doğrulaması)
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from uyku_v3.audio.sentence_cache import SentenceAudioCache
from uyku_v3.audio.text_normalizer import split_long_sentence_for_tts
from uyku_v3.audio.whisper_validator import validate_sentence_audio
from uyku_v3.audio.xtts_engine import XTTSEngine, is_pure_fast_wpm_only
from uyku_v3.core.state_manager import ArtifactStatus


def _speech_sig(dur_sec: float, sr: int = 24000, amp: float = 0.18) -> np.ndarray:
    t = np.linspace(0, dur_sec, int(dur_sec * sr), endpoint=False, dtype=np.float32)
    sig = amp * (0.6 * np.sin(2 * np.pi * 180 * t) + 0.4 * np.sin(2 * np.pi * 340 * t))
    sig *= 0.8 + 0.2 * np.sin(2 * np.pi * 3.5 * t)
    pad = np.zeros(int(0.08 * sr), dtype=np.float32)
    return np.concatenate([pad, sig, pad]).astype(np.float32)


class TestV22Modules(unittest.TestCase):
    def test_01_sentence_cache_sha256_and_corrupt_rejection(self) -> None:
        """[REAL_LOCAL] Cümle önbelleği SHA-256 doğrulaması ve bozuk dosya reddi."""
        sr = 24000
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            cache = SentenceAudioCache(td_p / "cache", sample_rate=sr)
            sig = _speech_sig(1.5, sr=sr)
            params = {"speed": 0.84, "v": "2.2"}

            # İlk sorgu -> miss
            self.assertIsNone(cache.get_verified("paragraf_001", 1, "Test cümlesi.", params))
            self.assertEqual(cache.misses, 1)

            # Kaydet -> ikinci sorgu hit
            entry = cache.store_approved("paragraf_001", 1, "Test cümlesi.", params, sig, {"wer": 0.0}, td_p / "tmp")
            res = cache.get_verified("paragraf_001", 1, "Test cümlesi.", params)
            self.assertIsNotNone(res)
            self.assertEqual(cache.hits, 1)

            # Metin değişirse -> miss
            self.assertIsNone(cache.get_verified("paragraf_001", 1, "Farklı cümle.", params))

            # Diskteki WAV bozulursa (SHA-256 uyuşmazlığı) -> fail-closed miss!
            wav_p = Path(entry.wav_path)
            raw = bytearray(wav_p.read_bytes())
            raw[-1] ^= 0xFF
            wav_p.write_bytes(bytes(raw))
            self.assertIsNone(cache.get_verified("paragraf_001", 1, "Test cümlesi.", params))

    def test_02_long_sentence_comma_breath_split_preserves_original(self) -> None:
        """[REAL_LOCAL] 9+ kelimelik virgüllü cümle iki nefes grubuna bölünür; orijinal metin değişmez."""
        orig = "Sen olduğun yerde dinlenirken, bir yerlerde Arin sisli bir patikada yürüyor."
        copy_orig = str(orig)
        groups = split_long_sentence_for_tts(orig)
        self.assertEqual(orig, copy_orig)
        self.assertEqual(len(groups), 2)
        self.assertTrue(groups[0].startswith("Sen olduğun yerde dinlenirken"))
        self.assertTrue(groups[1].startswith("bir yerlerde Arin"))

        # Kısa cümle bölünmemeli
        short = "Orada hiçbir anı kaybolmaz."
        self.assertEqual(len(split_long_sentence_for_tts(short)), 1)

    def test_03_controlled_atempo_only_on_zero_wer_fast_wpm(self) -> None:
        """[SIMULATED_GPU] Kontrollü atempo yalnızca WER=0, CER=0 hızlı cümlede çalışır; telaffuz hatasını gizlemez."""
        sr = 24000
        # 7 kelime 2.65 sn -> ~158 WPM > 145 WPM
        fast_sig = _speech_sig(2.65, sr=sr)
        slow_sig = _speech_sig(3.20, sr=sr)

        def fake_synth(text, temp, speed, top_p):
            return fast_sig.copy()

        def fake_transcribe_clean(audio, s_rate):
            return "Üzerinde ne tokmak vardı ne de kilit.", [{"word": "kilit", "start": 2.1, "end": 2.4}]

        def fake_atempo(audio, s_rate, factor, t_dir):
            return slow_sig.copy()

        eng = XTTSEngine(synthesizer_fn=fake_synth, transcriber_fn=fake_transcribe_clean)
        res_ok = eng.synthesize_sentence_v22(
            "Üzerinde ne tokmak vardı ne de kilit.",
            enable_atempo_compensation=True,
            atempo_fn=fake_atempo,
        )
        self.assertEqual(res_ok.status, ArtifactStatus.APPROVED)
        self.assertIsNotNone(res_ok.atempo_applied)

        # Telaffuz hatası (WER > 0) varsa atempo ASLA devreye girmemeli ve reddedilmeli!
        def fake_transcribe_bad(audio, s_rate):
            return "Üzerinde ne tokmak vardı ne de kilidi.", [{"word": "kilidi", "start": 2.1, "end": 2.4}]

        eng_bad = XTTSEngine(synthesizer_fn=fake_synth, transcriber_fn=fake_transcribe_bad)
        res_bad = eng_bad.synthesize_sentence_v22(
            "Üzerinde ne tokmak vardı ne de kilit.",
            enable_atempo_compensation=True,
            atempo_fn=fake_atempo,
        )
        self.assertEqual(res_bad.status, ArtifactStatus.MANUAL_REVIEW)
        self.assertIsNone(res_bad.atempo_applied)

    def test_04_adaptive_retry_profiles_switch_and_fail_closed(self) -> None:
        """[SIMULATED_GPU] Son hece hatasında uyarlanabilir profillere geçer; 4 deneme başarısızsa MANUAL_REVIEW kalır."""
        sr = 24000
        sig = _speech_sig(2.4, sr=sr)

        def fake_synth(text, temp, speed, top_p):
            return sig.copy()

        def fake_transcribe_suffix_err(audio, s_rate):
            return "Orada hiçbir anı kaybolmazdım.", [{"word": "kaybolmazdım", "start": 1.8, "end": 2.2}]

        eng = XTTSEngine(synthesizer_fn=fake_synth, transcriber_fn=fake_transcribe_suffix_err)
        res = eng.synthesize_sentence_v22("Orada hiçbir anı kaybolmaz.", enable_adaptive_profiles=True)
        self.assertEqual(res.status, ArtifactStatus.MANUAL_REVIEW)
        self.assertEqual(res.attempts, 4)
        used_profiles = [h["profile_name"] for h in res.history]
        self.assertEqual(used_profiles[0], "std_primary")
        self.assertTrue(used_profiles[1].startswith("adaptive_low_rep_"))

    def test_05_proper_noun_still_manual_review_excluded(self) -> None:
        """[REAL_LOCAL] #5 kuralı gereği özel isim farkı otomatik APPROVED yapılmaz, MANUAL_REVIEW kalır."""
        val = validate_sentence_audio(
            expected_text="Arin yosunların üzerine adım attı.",
            transcribed_text="Eren yosunların üzerine adım attı.",
            audio_signal=_speech_sig(2.6),
            known_proper_nouns=["Arin", "Liora", "Elias"],
        )
        self.assertTrue(val.needs_manual_review)

    def test_06_interruption_resume_reuses_cached_approved(self) -> None:
        """[REAL_LOCAL] Kesinti sonrası devamda onaylı cümleler önbellekten gelir, yeniden üretilmez."""
        sr = 24000
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            cache = SentenceAudioCache(td_p / "cache", sample_rate=sr)
            params = {"speed": 0.84}
            synth_calls = 0

            sents = ["Birinci cümle burada.", "İkinci cümle burada.", "Üçüncü cümle burada."]
            # 1. çalıştırma: ilk 2 cümle üretilip önbelleğe yazılır, 3. cümlede kesinti olur
            for idx, s in enumerate(sents[:2], 1):
                if cache.get_verified("paragraf_000", idx, s, params) is None:
                    synth_calls += 1
                    cache.store_approved("paragraf_000", idx, s, params, _speech_sig(1.8), {"wer": 0.0}, td_p / "tmp")
            self.assertEqual(synth_calls, 2)

            # 2. çalıştırma (kesinti sonrası devam): ilk 2 cümle önbellekten gelir, yalnızca 3. üretilir
            cache2 = SentenceAudioCache(td_p / "cache", sample_rate=sr)
            for idx, s in enumerate(sents, 1):
                if cache2.get_verified("paragraf_000", idx, s, params) is None:
                    synth_calls += 1
                    cache2.store_approved("paragraf_000", idx, s, params, _speech_sig(1.8), {"wer": 0.0}, td_p / "tmp")
            self.assertEqual(cache2.hits, 2)
            self.assertEqual(cache2.misses, 1)
            self.assertEqual(synth_calls, 3)


    def test_07_validating_interruption_safe_transition(self) -> None:
        """[REAL_LOCAL] Regresyon 1: VALIDATING durumunda kesinti -> INTERRUPTED -> PENDING -> GENERATING."""
        from uyku_v3.core.manifest import ManifestManager
        with tempfile.TemporaryDirectory() as td:
            m = ManifestManager(Path(td) / "test_m.json", "p_test")
            m.register_or_validate_cache("paragraf_001", "audio_v22_test", str(Path(td) / "p1.wav"), "h1", "xtts_v2", "v2.2", {})
            m.set_status("paragraf_001", ArtifactStatus.GENERATING, increment_attempt=True)
            m.set_status("paragraf_001", ArtifactStatus.VALIDATING)
            # Güvenli geçiş: VALIDATING -> INTERRUPTED -> PENDING -> GENERATING
            m.set_status("paragraf_001", ArtifactStatus.INTERRUPTED, error_message="Kesinti")
            m.set_status("paragraf_001", ArtifactStatus.PENDING, error_message="Yeniden deneme")
            m.set_status("paragraf_001", ArtifactStatus.GENERATING, increment_attempt=True)
            self.assertEqual(m.get("paragraf_001").get_status(), ArtifactStatus.GENERATING)

    def test_08_profile_stats_excludes_needs_manual_review(self) -> None:
        """[SIMULATED_GPU] Regresyon 3: Özel isim vakasında passed=True, needs_manual_review=True profil başarısı sayılmamalı."""
        sr = 24000
        sig = _speech_sig(2.6, sr=sr)

        def fake_synth(text, temp, speed, top_p):
            return sig.copy()

        def fake_transcribe_proper(audio, s_rate):
            return "Eren yosunların üzerine adım attı.", [{"word": "attı", "start": 2.1, "end": 2.4}]

        eng = XTTSEngine(synthesizer_fn=fake_synth, transcriber_fn=fake_transcribe_proper, known_proper_nouns=["Arin"])
        res = eng.synthesize_sentence_v22("Arin yosunların üzerine adım attı.")
        self.assertEqual(res.status, ArtifactStatus.MANUAL_REVIEW)
        passes = sum(1 for h in res.history if (h.get("passed") is True and h.get("needs_manual_review") is False))
        self.assertEqual(passes, 0)

    def test_09_cache_invalidates_on_xtts_config_or_threshold_change(self) -> None:
        """[REAL_LOCAL] Regresyon 4: XTTSConfig veya AudioThresholds değiştiğinde cümle önbelleği geçersizleşmeli."""
        from dataclasses import asdict
        from uyku_v3.config.model_config import XTTSConfig
        from uyku_v3.config.quality_thresholds import AudioThresholds

        sr = 24000
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            cache = SentenceAudioCache(td_p / "cache", sample_rate=sr)
            p1 = {"v": "2.2", "xtts": asdict(XTTSConfig()), "thresholds": asdict(AudioThresholds())}
            cache.store_approved("paragraf_003", 1, "Örnek cümle.", p1, _speech_sig(1.6), {"wer": 0.0}, td_p / "tmp")
            self.assertIsNotNone(cache.get_verified("paragraf_003", 1, "Örnek cümle.", p1))

            # XTTSConfig.speed değişirse -> miss
            p2 = {"v": "2.2", "xtts": asdict(XTTSConfig(speed=0.79)), "thresholds": asdict(AudioThresholds())}
            self.assertIsNone(cache.get_verified("paragraf_003", 1, "Örnek cümle.", p2))

            # AudioThresholds.max_wer değişirse -> miss
            p3 = {"v": "2.2", "xtts": asdict(XTTSConfig()), "thresholds": asdict(AudioThresholds(max_wer=0.08))}
            self.assertIsNone(cache.get_verified("paragraf_003", 1, "Örnek cümle.", p3))

    def test_10_two_run_interruption_resume_and_cumulative_10p_metrics(self) -> None:
        """[REAL_LOCAL] Regresyon 2: 10 paragrafta 3 paragraf sonrası kesinti + 2. çalıştırmada önbellek ve kümülatif metrikler."""
        from uyku_v3.core.manifest import ManifestManager
        sr = 24000
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            cache = SentenceAudioCache(td_p / "cache", sample_rate=sr)
            m = ManifestManager(td_p / "m.json", "test_10p")
            params = {"v": "2.2"}
            synth_calls = 0

            # 10 paragraf x 2 cümle = 20 cümle; P000-P002 Run 1'de üretilir (P000,P001 APPROVED; P002'de C01 APPROVED, C02 MANUAL_REVIEW)
            parags = [f"paragraf_{i:03d}" for i in range(10)]

            def run_pass(stop_after: int, fix_p002: bool):
                nonlocal synth_calls
                reused_approved_p = 0
                reused_approved_p_sents = 0
                c_cache = SentenceAudioCache(td_p / "cache", sample_rate=sr)
                done = 0
                for pid in parags:
                    if stop_after > 0 and done >= stop_after:
                        break
                    out_w = td_p / f"{pid}.wav"
                    rec = m.register_or_validate_cache(pid, "audio_v22_test", str(out_w), f"h_{pid}", "xtts_v2", "v2.2", {})
                    if rec.get_status() == ArtifactStatus.APPROVED:
                        reused_approved_p += 1
                        reused_approved_p_sents += 2
                        done += 1
                        continue
                    if rec.get_status() in (ArtifactStatus.MANUAL_REVIEW, ArtifactStatus.APPROVED):
                        m.set_status(pid, ArtifactStatus.PENDING)
                    m.set_status(pid, ArtifactStatus.GENERATING, increment_attempt=True)
                    failed = False
                    s_recs = []
                    for c_i in (1, 2):
                        txt = f"{pid} cümle {c_i}."
                        hit = c_cache.get_verified(pid, c_i, txt, params)
                        if hit is not None:
                            s_recs.append({"c_idx": c_i, "status": "APPROVED", "from_cache": True})
                            continue
                        synth_calls += 1
                        if pid == "paragraf_002" and c_i == 2 and not fix_p002:
                            failed = True
                            s_recs.append({"c_idx": c_i, "status": "MANUAL_REVIEW", "from_cache": False})
                        else:
                            c_cache.store_approved(pid, c_i, txt, params, _speech_sig(1.5), {"wer": 0.0}, td_p / "tmp")
                            s_recs.append({"c_idx": c_i, "status": "APPROVED", "from_cache": False})
                    m.set_status(pid, ArtifactStatus.VALIDATING)
                    if failed:
                        m.set_status(pid, ArtifactStatus.MANUAL_REVIEW, quality_scores={"sentences": s_recs})
                    else:
                        out_w.write_bytes(b"WAVDATA")
                        m.set_status(pid, ArtifactStatus.APPROVED, quality_scores={"sentences": s_recs}, compute_output_hash=True)
                    done += 1
                return reused_approved_p, reused_approved_p_sents, c_cache.hits

            # Run 1 (--stop-after 3): 6 cümle üretilir (P000 APPROVED, P001 APPROVED, P002 C01 önbellekte, C02 MANUAL_REVIEW)
            r1_p, r1_ps, r1_ch = run_pass(stop_after=3, fix_p002=False)
            self.assertEqual((r1_p, r1_ps, r1_ch, synth_calls), (0, 0, 0, 6))

            # Run 2 (10 paragraf devam): P000 ve P001 atlanır (4 cümle yeniden üretilmez), P002 C01 önbellekten gelir (1 hit),
            # yalnızca P002 C02 + P003..P009 (1 + 14 = 15 cümle) sentezlenir -> toplam 6+15=21 (sıfır onaylı tekrar!)
            r2_p, r2_ps, r2_ch = run_pass(stop_after=0, fix_p002=True)
            self.assertEqual(r2_p, 2)
            self.assertEqual(r2_ps, 4)
            self.assertEqual(r2_ch, 1)
            self.assertEqual(synth_calls, 21)
            self.assertEqual(m.summary_counts().get("APPROVED"), 10)

if __name__ == "__main__":
    unittest.main()
