# Otomatik Test ve Kabul Kriterleri Raporu — UYKU_BORU_HATTI_V2
- **Çalıştırma Tarihi (UTC):** 2026-11-19T18:48:03.140292+00:00
- **Toplam Test:** 46 | **Başarılı:** 46 | **Başarısız:** 0 | **Süre:** 45.91 sn
- **Ortam Notu:** Yerel Linux CPU ortamında çalıştırıldı (Python 3.11, FFmpeg, FFprobe, OpenCV, NumPy, SciPy, PIL). T4 GPU gerektiren yapay zekâ modelleri (XTTS-v2, Whisper Turbo, SDXL 1.0, SVD-XT, CLIP) SIMULATED_GPU olarak enjekte edilmiş; tüm ses DSP, normalizasyon, kalite kontrol, OpenCV, FFmpeg montaj/decode, SRT ve kurtarma motorları GERÇEK yerel ortamda çalıştırılmıştır.

| # | Test | Çalışma Türü | Durum | Beklenen / Açıklama |
|---|---|---|---|---|
| 1 | `test_01_single_sentence_xtts_simulated_gpu` | `SIMULATED_GPU` | **PASS** | [SIMULATED_GPU] Tek cümle XTTS üretim + 3 aşamalı QC testi. |
| 2 | `test_02_turkish_proper_noun_pronunciation_dict` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Türkçe özel isim (Arin, Liora, Elias) ve kelime-sınırı güvenliği testi. |
| 3 | `test_03_number_date_and_turkish_i_normalization` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Sayı, sıra sayısı, tarih ve Türkçe İ/I normalizasyon testi. |
| 4 | `test_04_missing_word_detection` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Eksik kelime tespiti testi. |
| 5 | `test_05_extra_word_detection` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Fazladan kelime/halüsinasyon tespiti testi. |
| 6 | `test_06_sentence_end_tail_cut_detection` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Cümle sonu fiil hecesi düşmesi ve ani kesilme testi. |
| 7 | `test_07_whisper_misrecognition_vs_audio_error` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Whisper özel isim yanlış tanıma vs gerçek ses hatası ayrımı testi. |
| 8 | `test_08_low_paragraph_completeness_rejected` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Düşük paragraf tamlık skoru (eski sistemdeki 0.075/0.15 hatası) kesinlikle reddedilmeli. |
| 9 | `test_09_loudness_clipping_and_true_peak_mastering` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Clipping tespiti ve -2.5 dBTP True Peak mastering koruması testi. |
| 10 | `test_10_v21_real_missing_word_rejected` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Gerçek eksik kelimeler (yün, palto) kesinlikle reddedilmeli. |
| 11 | `test_11_v21_wrong_last_word_and_turkish_suffix_rejected` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Yanlış son kelime (et->etmiyor, kapanmaz->kapanmazlığa, kuşlar->var) reddedilmeli. |
| 12 | `test_12_v21_short_sentence_natural_vs_rushed_wpm` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Kısa cümle doğal hızda onaylanmalı, gerçek hızlı/kesik okuma reddedilmeli. |
| 13 | `test_13_v21_long_sentence_fast_speech_still_rejected` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Uzun cümlede (9 kelime) 165 WPM gerçek hızlı konuşma hala reddedilmeli (145 WPM korunmalı). |
| 14 | `test_14_v21_compound_word_split_normalized_without_hiding_missing` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Birleşik kelime bölünmesi (başparmağıyla / baş parmağıyla) normalize edilmeli; gerçek eksik kelime gizlenmemeli. |
| 15 | `test_15_v21_cer_low_non_proper_not_auto_approved_and_proper_noun_manual` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] CER <= 0.05 genel yazım farkı otomatik onaylanmamalı; özel isim farkı MANUAL_REVIEW kalmalı. |
| 16 | `test_16_v21_manual_review_preserves_raw_audio` | `SIMULATED_GPU` | **PASS** | [SIMULATED_GPU] MANUAL_REVIEW durumuna düşen cümlede ham XTTS sesi korunmalı. |
| 17 | `test_22_corrupt_mp4_rejected` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Bozuk MP4 dosyası ffprobe + decode kontrolünden geçememeli. |
| 18 | `test_23_missing_scene_blocks_assembly` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Eksik sahne varsa FFmpeg montajı başlamadan hata vermeli. |
| 19 | `test_24_25_34_black_screen_av_sync_and_full_decode` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Saf #000000 siyah ekran, ses-video senkronu ve tam akış decode testi. |
| 20 | `test_e2e_small_turkish_sleep_story_pipeline` | `E2E_INTEGRATION (REAL_LOCAL_DSP_FFMPEG + SIMULATED_GPU_MODELS)` | **PASS** | [E2E_INTEGRATION: REAL_LOCAL_DSP_FFMPEG + SIMULATED_GPU_MODELS]         Küçük 2 cümlelik Türkçe uyku hikâyesi ile uçtan uca boru hattı testi:         Metin Normalizasyon -> Ses QC -> Mastering -> Görsel QC -> SVD Klip QC ->         FFmpeg 1080p Montaj + Siyah Outro -> SRT + Doğrulama -> Son Kalite Raporu (APPROVED). |
| 21 | `test_29_colab_session_interruption_recovery` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Colab oturumu GENERATING sırasında kesildiğinde dosya karantinaya alınır ve INTERRUPTED olur. |
| 22 | `test_30_drive_disconnect_resilience_atomic_copy` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Drive kopyalamasında geçici I/O hatası sonrası yeniden deneme ve veri kaybı olmaması. |
| 23 | `test_31_partial_tmp_file_quarantined` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Yarım kalmış .tmp dosyaları kurtarma sırasında karantinaya taşınır. |
| 24 | `test_32_corrupt_manifest_recovery` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Bozuk JSON manifest dosyası yedeğe alınıp sistem çökmeden yeniden başlar. |
| 25 | `test_33_rerun_preserves_approved_and_invalidates_on_source_change` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Yeniden çalıştırmada geçerli APPROVED çıktılar korunur; kaynak metin değişirse geçersizleşir. |
| 26 | `test_26_srt_missing_word_flagged_for_manual_review` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Ortadaki kelimeleri eksik cümle ilk/son kelime tutsa bile MANUAL_REVIEW olmalı. |
| 27 | `test_27_srt_overlap_prevented_and_detected` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] SRT zaman çakışması üretimde engellenir ve bağımsız doğrulayıcı çakışmayı yakalar. |
| 28 | `test_28_srt_turkish_characters_and_no_truncation_and_ms_rounding` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Türkçe karakterler (çğıöşüİ), 42 karakter x 2 satır kelime kayıpsız bölme ve ms=1000 engeli. |
| 29 | `test_01_sentence_cache_sha256_and_corrupt_rejection` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Cümle önbelleği SHA-256 doğrulaması ve bozuk dosya reddi. |
| 30 | `test_02_long_sentence_comma_breath_split_preserves_original` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] 9+ kelimelik virgüllü cümle iki nefes grubuna bölünür; orijinal metin değişmez. |
| 31 | `test_03_controlled_atempo_only_on_zero_wer_fast_wpm` | `SIMULATED_GPU` | **PASS** | [SIMULATED_GPU] Kontrollü atempo yalnızca WER=0, CER=0 hızlı cümlede çalışır; telaffuz hatasını gizlemez. |
| 32 | `test_04_adaptive_retry_profiles_switch_and_fail_closed` | `SIMULATED_GPU` | **PASS** | [SIMULATED_GPU] Son hece hatasında uyarlanabilir profillere geçer; 4 deneme başarısızsa MANUAL_REVIEW kalır. |
| 33 | `test_05_proper_noun_still_manual_review_excluded` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] #5 kuralı gereği özel isim farkı otomatik APPROVED yapılmaz, MANUAL_REVIEW kalır. |
| 34 | `test_06_interruption_resume_reuses_cached_approved` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Kesinti sonrası devamda onaylı cümleler önbellekten gelir, yeniden üretilmez. |
| 35 | `test_10_single_image_generation_and_validation` | `HYBRID (SIMULATED_GPU + REAL_LOCAL)` | **PASS** | [SIMULATED_GPU+REAL_LOCAL] Tek görsel üretim ve 1920x1080 geçerlilik testi. |
| 36 | `test_11_wrong_aspect_ratio_rejected` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Yanlış en-boy oranı (4:3) reddedilmeli. |
| 37 | `test_12_single_scene_svd_xt_simulated_gpu` | `HYBRID (SIMULATED_GPU + REAL_LOCAL)` | **PASS** | [SIMULATED_GPU+REAL_LOCAL] Tek sahne SVD-XT üretimi + CLIP/OpenCV/FFmpeg onay testi. |
| 38 | `test_13_clip_wrong_scene_rejected` | `SIMULATED_GPU` | **PASS** | [SIMULATED_GPU] CLIP yanlış sahne (düşük pozitif kosinüs) reddi testi. |
| 39 | `test_14_clip_human_figure_negative_risk_rejected` | `SIMULATED_GPU` | **PASS** | [SIMULATED_GPU] İnsan yüzü/figürü negatif kavram riski reddi testi. |
| 40 | `test_15_excessive_brightness_rejected` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Uyku için aşırı parlak gündüz sahnesi reddi testi. |
| 41 | `test_16_flicker_detection` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Ani parlaklık sıçraması (flicker) tespiti testi. |
| 42 | `test_17_blur_detection` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Karelerde bulanıklaşma (Laplacian keskinlik düşüşü) tespiti testi. |
| 43 | `test_18_morphing_drift_detection` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Aşırı şekil bozulması / morphing drift tespiti testi. |
| 44 | `test_19_three_failed_svd_attempts_never_approved` | `HYBRID (SIMULATED_GPU + REAL_LOCAL)` | **PASS** | [SIMULATED_GPU+REAL_LOCAL] P0-03 testi: 3 başarısız video denemesi ASLA kaydedilmez ve MANUAL_REVIEW olur. |
| 45 | `test_20_ping_pong_loop_generation` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Simetrik iç mekan sahnesi için ping-pong döngü testi. |
| 46 | `test_21_directional_motion_forward_crossfade_loop` | `REAL_LOCAL` | **PASS** | [REAL_LOCAL] Ateş/dalga/yağmur gibi yönlü hareketler için tersine dönmeyen ileri döngü testi. |
