"""
Bölüm 15 Türkçe SRT Altyazı Test Paketi (Test 26–28 ve ek sınır testleri).
Tamamı gerçek yerel ortamda çalışır (REAL_LOCAL).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from uyku_v3.subtitles.transcription import WordTimestamp
from uyku_v3.subtitles.alignment import align_sentences_to_words
from uyku_v3.subtitles.srt_generator import (
    format_srt_timestamp,
    wrap_and_split_text,
    generate_srt_cues,
    write_srt_file,
)
from uyku_v3.subtitles.srt_validator import validate_srt_file
from uyku_v3.core.state_manager import ArtifactStatus


class TestSubtitles(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_26_srt_missing_word_flagged_for_manual_review(self) -> None:
        """[REAL_LOCAL] Ortadaki kelimeleri eksik cümle ilk/son kelime tutsa bile MANUAL_REVIEW olmalı."""
        sentences = ["Kül Koyu kıyısında eski taş kulübenin ağır meşe kapısı yavaşça kapandı."]
        # Yalnızca ilk ve son kelimeler mevcut, ortadaki 7 kelime eksik!
        whisper_words = [
            WordTimestamp("Kül", "kül", 0.5, 0.9),
            WordTimestamp("kapandı", "kapandı", 4.0, 4.6),
        ]
        aligned = align_sentences_to_words(sentences, whisper_words, outro_start_sec=60.0)
        self.assertEqual(aligned[0].status, ArtifactStatus.MANUAL_REVIEW)
        self.assertGreater(len(aligned[0].missing_words), 5)

    def test_27_srt_overlap_prevented_and_detected(self) -> None:
        """[REAL_LOCAL] SRT zaman çakışması üretimde engellenir ve bağımsız doğrulayıcı çakışmayı yakalar."""
        # Çakışmalı manuel SRT yazıp doğrulayıcının yakaladığını kanıtla
        bad_srt = self.tmp_dir / "overlap.srt"
        bad_srt.write_text(
            "1\n00:00:01,000 --> 00:00:04,500\nİlk cümle burada.\n\n"
            "2\n00:00:04,000 --> 00:00:07,000\nİkinci cümle çakışıyor.\n",
            encoding="utf-8",
        )
        bad_val = validate_srt_file(bad_srt)
        self.assertFalse(bad_val.passed)
        self.assertEqual(bad_val.overlap_count, 1)

        # Üreticinin aynı durumda çakışmayı otomatik giderdiğini kanıtla
        sentences = ["İlk cümle burada.", "İkinci cümle hemen başlıyor."]
        words = [
            WordTimestamp("İlk", "ilk", 1.0, 1.4),
            WordTimestamp("cümle", "cümle", 1.4, 1.8),
            WordTimestamp("burada", "burada", 1.8, 2.2),
            WordTimestamp("İkinci", "ikinci", 2.25, 2.7),
            WordTimestamp("cümle", "cümle", 2.7, 3.1),
            WordTimestamp("hemen", "hemen", 3.1, 3.5),
            WordTimestamp("başlıyor", "başlıyor", 3.5, 4.2),
        ]
        aligned = align_sentences_to_words(sentences, words, outro_start_sec=10.0)
        cues = generate_srt_cues(aligned, outro_start_sec=10.0)
        good_srt = self.tmp_dir / "clean.srt"
        write_srt_file(cues, good_srt)
        good_val = validate_srt_file(good_srt, expected_sentences=sentences, outro_start_sec=10.0)
        self.assertTrue(good_val.passed, f"SRT hataları: {good_val.issues}")
        self.assertEqual(good_val.overlap_count, 0)

    def test_28_srt_turkish_characters_and_no_truncation_and_ms_rounding(self) -> None:
        """[REAL_LOCAL] Türkçe karakterler (çğıöşüİ), 42 karakter x 2 satır kelime kayıpsız bölme ve ms=1000 engeli."""
        # ms=1000 taşma testi
        self.assertEqual(format_srt_timestamp(1.9999), "00:00:02,000")

        # Uzun Türkçe cümle 2 satırı aştığında hiçbir kelime silinmeden ardışık bloklara bölünmeli
        long_tr = (
            "İstanbul'un serin rüzgârı çatıdaki yağmur damlalarıyla birleşip "
            "ağır meşe kapının ardındaki sıcacık taş ocağın çevresinde uğulduyordu."
        )
        blocks = wrap_and_split_text(long_tr, max_chars_per_line=42, max_lines_per_cue=2)
        self.assertGreaterEqual(len(blocks), 2)
        reconstructed = " ".join(" ".join(b) for b in blocks)
        self.assertEqual(reconstructed, long_tr)
        for blk in blocks:
            self.assertLessEqual(len(blk), 2)
            for line in blk:
                self.assertLessEqual(len(line), 42)

        # Mojibake tespiti
        mojibake_srt = self.tmp_dir / "mojibake.srt"
        mojibake_srt.write_text(
            "1\n00:00:01,000 --> 00:00:03,000\nYaÄŸmur yaÄŸÄ±yordu.\n",
            encoding="utf-8",
        )
        val = validate_srt_file(mojibake_srt)
        self.assertFalse(val.passed)
        self.assertGreater(val.encoding_issues, 0)


if __name__ == "__main__":
    unittest.main()
