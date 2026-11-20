"""
Bölüm 15 Ses Test Paketi (Test 1–9).
Gerçek DSP/Normalizasyon/Kalite kontrol testleri yerel ortamda doğrudan çalışır;
XTTS-v2 ve Whisper GPU çağrıları açıkça SIMULATED_GPU olarak etiketlenerek test edilir.
"""

from __future__ import annotations

import unittest
import numpy as np

from uyku_v3.audio.pronunciation_dictionary import PronunciationDictionary
from uyku_v3.audio.text_normalizer import (
    tr_lower,
    tr_norm_tokens,
    number_to_turkish_words,
    normalize_for_tts,
)
from uyku_v3.audio.audio_quality import analyze_audio_acoustics, compute_true_peak_dbtp, safe_trim_trailing_ghost
from uyku_v3.audio.whisper_validator import validate_sentence_audio, validate_paragraph_audio
from uyku_v3.audio.xtts_engine import XTTSEngine
from uyku_v3.audio.mastering import MasteringEngine
from uyku_v3.core.state_manager import ArtifactStatus


def _make_speech_like_signal(duration_sec: float = 3.0, sr: int = 24000, amp: float = 0.14) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    sig = amp * (0.6 * np.sin(2 * np.pi * 160 * t) + 0.4 * np.sin(2 * np.pi * 320 * t))
    # Başa ve sona 120ms yumuşak sessizlik ekle
    pad = np.zeros(int(0.12 * sr), dtype=np.float32)
    fo = int(0.05 * sr)
    sig[:fo] *= np.linspace(0, 1, fo, dtype=np.float32)
    sig[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
    return np.concatenate([pad, sig, pad]).astype(np.float32)


class TestAudioModules(unittest.TestCase):
    def test_01_single_sentence_xtts_simulated_gpu(self) -> None:
        """[SIMULATED_GPU] Tek cümle XTTS üretim + 3 aşamalı QC testi."""
        sentence = "Gece rüzgârı taş duvarların arasından yavaşça geçiyordu."
        sig = _make_speech_like_signal(3.8)
        engine = XTTSEngine(
            synthesizer_fn=lambda text, temp, spd, tp: sig.copy(),
            transcriber_fn=lambda wav, sr: (sentence, [{"word": "geçiyordu", "start": 3.0, "end": 3.7}]),
        )
        res = engine.synthesize_sentence_with_qc(sentence)
        self.assertEqual(res.status, ArtifactStatus.APPROVED)
        self.assertEqual(res.attempts, 1)
        self.assertTrue(res.validation.passed)

    def test_02_turkish_proper_noun_pronunciation_dict(self) -> None:
        """[REAL_LOCAL] Türkçe özel isim (Arin, Liora, Elias) ve kelime-sınırı güvenliği testi."""
        pdict = PronunciationDictionary()
        raw = "Arin, Liora ve Elias hala eski halat ile rüzgar sesini dinliyordu."
        spoken = normalize_for_tts(raw, pronunciation_dict=pdict, preserve_circumflex=True)
        self.assertIn("Liyora", spoken)
        self.assertIn("Eliyas", spoken)
        self.assertIn("hâlâ", spoken)
        self.assertIn("halat", spoken)  # 'hala' kuralı 'halat'ı bozmamalı!
        self.assertIn("rüzgâr", spoken)  # Şapkalı harf korunmalı!
        # Orijinal metin değişmemiş olmalı
        self.assertIn("Liora", raw)

    def test_03_number_date_and_turkish_i_normalization(self) -> None:
        """[REAL_LOCAL] Sayı, sıra sayısı, tarih ve Türkçe İ/I normalizasyon testi."""
        self.assertEqual(tr_lower("İSTANBUL IŞIK"), "istanbul ışık")
        self.assertEqual(number_to_turkish_words(400), "dört yüz")
        self.assertEqual(number_to_turkish_words(1924), "bin dokuz yüz yirmi dört")
        toks = tr_norm_tokens("400. akşamda 14.05.1924 tarihinde İstanbul'da ışık yandı.")
        self.assertIn("dört", toks)
        self.assertIn("yüzüncü", toks)
        self.assertIn("istanbulda", toks)
        self.assertIn("ışık", toks)

    def test_04_missing_word_detection(self) -> None:
        """[REAL_LOCAL] Eksik kelime tespiti testi."""
        exp = "Kül Koyu kıyısında ağır meşe kapı sessizce kapandı."
        act = "Kül Koyu kıyısında meşe kapı sessizce kapandı."
        res = validate_sentence_audio(exp, act, audio_signal=_make_speech_like_signal(3.5))
        self.assertFalse(res.passed)
        self.assertIn("ağır", res.missing_words)

    def test_05_extra_word_detection(self) -> None:
        """[REAL_LOCAL] Fazladan kelime/halüsinasyon tespiti testi."""
        exp = "Ocaktaki közler yavaşça kızarıyordu."
        act = "Ocaktaki közler yavaşça kızarıyordu bı mıva fazla."
        res = validate_sentence_audio(exp, act, audio_signal=_make_speech_like_signal(2.8))
        self.assertFalse(res.passed)
        self.assertGreater(len(res.extra_words), 1)

    def test_06_sentence_end_tail_cut_detection(self) -> None:
        """[REAL_LOCAL] Cümle sonu fiil hecesi düşmesi ve ani kesilme testi."""
        exp = "Yağmur damlaları çatıya yavaşça vuruyordu."
        act = "Yağmur damlaları çatıya yavaşça vuruyor."
        # Ani kesilmiş ses sinyali (sonunda fade-out ve sessizlik yok)
        sr = 24000
        t = np.linspace(0, 2.5, int(sr * 2.5), dtype=np.float32)
        abrupt = (0.18 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        res = validate_sentence_audio(exp, act, audio_signal=abrupt, sr=sr)
        self.assertFalse(res.passed)
        self.assertFalse(res.last_syllable_ok)
        self.assertTrue(res.acoustic["abrupt_tail_cut"])

        # Güvenli hayalet temizleyici son kelimeden önce ASLA kesmemeli
        clean_sig = _make_speech_like_signal(3.0)
        trimmed, was_cut = safe_trim_trailing_ghost(clean_sig, sr=sr, last_word_end_sec=2.9)
        self.assertFalse(was_cut)
        self.assertEqual(len(trimmed), len(clean_sig))

    def test_07_whisper_misrecognition_vs_audio_error(self) -> None:
        """[REAL_LOCAL] Whisper özel isim yanlış tanıma vs gerçek ses hatası ayrımı testi."""
        exp = "Liora eski taş kulübede sıcak çayını yudumluyordu."
        act = "Liyora eski taş kulübede sıcak çayını yudumluyordu."
        res = validate_sentence_audio(
            exp,
            act,
            audio_signal=_make_speech_like_signal(3.5),
            known_proper_nouns=["Liora", "Arin", "Elias"],
        )
        self.assertTrue(res.whisper_error_suspected)
        self.assertTrue(res.last_word_ok)
        self.assertTrue(res.last_syllable_ok)

    def test_08_low_paragraph_completeness_rejected(self) -> None:
        """[REAL_LOCAL] Düşük paragraf tamlık skoru (eski sistemdeki 0.075/0.15 hatası) kesinlikle reddedilmeli."""
        sentences = [
            "Kül Koyu akşam sisinin altında tamamen sessizleşmişti.",
            "Taş kulübenin içinde ocak hafifçe çıtırdıyordu.",
            "Dışarıda yağmur ince bir perde gibi yağıyordu.",
        ]
        bad_transcript = "Kül Koyu akşam."
        res = validate_paragraph_audio(sentences, bad_transcript)
        self.assertFalse(res.passed)
        self.assertLess(res.word_alignment_ratio, 0.50)

    def test_09_loudness_clipping_and_true_peak_mastering(self) -> None:
        """[REAL_LOCAL] Clipping tespiti ve -2.5 dBTP True Peak mastering koruması testi."""
        sr = 24000
        clipped = np.clip(_make_speech_like_signal(2.5, sr=sr, amp=1.5), -1.0, 1.0)
        ac = analyze_audio_acoustics(clipped, sr=sr, expected_word_count=5)
        self.assertFalse(ac.passed)
        self.assertGreater(ac.clipped_samples, 0)

        # Mastering motoru K1-K6 ve True Peak <= -2.5 dBTP sağlamalı
        master_eng = MasteringEngine(sr=sr)
        out, rep = master_eng.mix_and_master(
            chapter_vocals=[("KAYIT_01", _make_speech_like_signal(2.5, sr=sr)), ("KAYIT_06", _make_speech_like_signal(2.5, sr=sr))],
            ambience_bed=_make_speech_like_signal(2.0, sr=sr, amp=0.05),
            outro_sec=4.0,
            outro_fade_sec=2.0,
        )
        self.assertTrue(rep.passed, f"Mastering sorunları: {rep.issues}")
        self.assertLessEqual(compute_true_peak_dbtp(out), -2.45)
        self.assertTrue(rep.outro_fade_verified)


    def test_10_v21_real_missing_word_rejected(self) -> None:
        """[REAL_LOCAL] Gerçek eksik kelimeler (yün, palto) kesinlikle reddedilmeli."""
        res = validate_sentence_audio(
            expected_text="Üzerinde kalın koyu renkli bir yün palto vardı.",
            transcribed_text="Üzerinde kalın koyu renkli bir vardı.",
        )
        self.assertFalse(res.passed)
        self.assertIn("yün", res.missing_words)
        self.assertIn("palto", res.missing_words)

    def test_11_v21_wrong_last_word_and_turkish_suffix_rejected(self) -> None:
        """[REAL_LOCAL] Yanlış son kelime (et->etmiyor, kapanmaz->kapanmazlığa, kuşlar->var) reddedilmeli."""
        cases = [
            ("Başının yastığa bıraktığı ağırlığı fark et.", "Başının yastığa bıraktığı ağırlığı fark etmiyor."),
            ("Hiçbir kapı sertçe kapanmaz.", "Hiçbir kapı sertçe kapanmazlığa."),
            ("Kanatlarını kapatıp dinlenen kuşlar.", "Kanatlarını kapatıp dinlenen var."),
        ]
        for exp, act in cases:
            res = validate_sentence_audio(expected_text=exp, transcribed_text=act)
            self.assertFalse(res.passed, f"Hatalı son kelime onaylandı: {exp} vs {act}")

    def test_12_v21_short_sentence_natural_vs_rushed_wpm(self) -> None:
        """[REAL_LOCAL] Kısa cümle doğal hızda onaylanmalı, gerçek hızlı/kesik okuma reddedilmeli."""
        sr = 24000
        # 3 kelime 0.90 sn (200 WPM, 0.30 sn/kelime) -> doğal kısa cümle, geçmeli
        sig_ok = _make_speech_like_signal(0.95, sr=sr)
        ac_ok = analyze_audio_acoustics(sig_ok, sr=sr, expected_word_count=3)
        self.assertTrue(ac_ok.passed, f"Doğal kısa cümle reddedildi: {ac_ok.issues}")

        # 3 kelime 0.45 sn (400 WPM, 0.15 sn/kelime) -> gerçekten aşırı hızlı, reddedilmeli
        sig_fast = _make_speech_like_signal(0.45, sr=sr)
        ac_fast = analyze_audio_acoustics(sig_fast, sr=sr, expected_word_count=3)
        self.assertFalse(ac_fast.passed)

    def test_13_v21_long_sentence_fast_speech_still_rejected(self) -> None:
        """[REAL_LOCAL] Uzun cümlede (9 kelime) 165 WPM gerçek hızlı konuşma hala reddedilmeli (145 WPM korunmalı)."""
        sr = 24000
        # 9 kelime 3.25 sn -> ~166 WPM > 145 WPM
        sig_fast_long = _make_speech_like_signal(3.25, sr=sr)
        ac = analyze_audio_acoustics(sig_fast_long, sr=sr, expected_word_count=9)
        self.assertFalse(ac.passed)
        self.assertEqual(ac.applied_max_wpm, 145.0)

    def test_14_v21_compound_word_split_normalized_without_hiding_missing(self) -> None:
        """[REAL_LOCAL] Birleşik kelime bölünmesi (başparmağıyla / baş parmağıyla) normalize edilmeli; gerçek eksik kelime gizlenmemeli."""
        res_ok = validate_sentence_audio(
            expected_text="Başparmağıyla anahtarın sapını yokladı.",
            transcribed_text="Baş parmağıyla anahtarın sapını yokladı.",
        )
        self.assertTrue(res_ok.passed, f"Birleşik kelime bölünmesi geçmedi: {res_ok.issues}")
        self.assertEqual(res_ok.wer, 0.0)

        # Gerçek eksik kelime gizlenmemeli
        res_bad = validate_sentence_audio(
            expected_text="Başparmağıyla anahtarın sapını yokladı.",
            transcribed_text="Baş parmağıyla sapını yokladı.",
        )
        self.assertFalse(res_bad.passed)
        self.assertIn("anahtarın", res_bad.missing_words)

    def test_15_v21_cer_low_non_proper_not_auto_approved_and_proper_noun_manual(self) -> None:
        """[REAL_LOCAL] CER <= 0.05 genel yazım farkı otomatik onaylanmamalı; özel isim farkı MANUAL_REVIEW kalmalı."""
        # Genel kelime farkı (düşük CER ama özel isim değil) -> otomatik onay yok
        res_non_proper = validate_sentence_audio(
            expected_text="Patika hafifçe kıvrıldı.",
            transcribed_text="Patika hafifçe kıvrandı.",
            known_proper_nouns=["Arin", "Liora", "Elias"],
        )
        self.assertFalse(res_non_proper.passed)

        # Özel isim (Arin -> Eren) -> son kelime/hizalama sağlamsa whisper_error_suspected + needs_manual_review
        res_proper = validate_sentence_audio(
            expected_text="Arin yosunların üzerine adım attı.",
            transcribed_text="Eren yosunların üzerine adım attı.",
            known_proper_nouns=["Arin", "Liora", "Elias"],
        )
        self.assertTrue(res_proper.whisper_error_suspected)
        self.assertTrue(res_proper.needs_manual_review)

    def test_16_v21_manual_review_preserves_raw_audio(self) -> None:
        """[SIMULATED_GPU] MANUAL_REVIEW durumuna düşen cümlede ham XTTS sesi korunmalı."""
        sr = 24000
        raw_sig = _make_speech_like_signal(2.2, sr=sr)

        def fake_synth(text, temp, speed, top_p):
            return raw_sig.copy()

        def fake_transcribe(audio, s_rate):
            return "Başının yastığa bıraktığı ağırlığı fark etmiyor.", [{"word": "etmiyor", "start": 1.8, "end": 2.1}]

        eng = XTTSEngine(synthesizer_fn=fake_synth, transcriber_fn=fake_transcribe)
        res = eng.synthesize_sentence_with_qc("Başının yastığa bıraktığı ağırlığı fark et.")
        self.assertEqual(res.status, ArtifactStatus.MANUAL_REVIEW)
        self.assertIsNotNone(res.raw_audio)
        self.assertEqual(len(res.raw_audio), len(raw_sig))

if __name__ == "__main__":
    unittest.main()
