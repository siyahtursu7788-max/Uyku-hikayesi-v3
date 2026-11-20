# Eski Dosya Envanteri, Sınıflandırma ve Kalıcı Silme Onay Listesi

Bu envanter, Google Drive'daki mevcut `Unutulmus_Zamanlar_Arsivi_Studyo` ve `UYKU_BORU_HATTI_PROFESYONEL_V2` klasörlerindeki tüm dosya ve klasörleri sınıflandırır. **Hiçbir dosya kalıcı olarak silinmemiştir.** Kalıcı silme yalnızca aşağıdaki **QUARANTINE (Silme Adayı)** listesindeki öğeler için açık onayınız verildikten sonra yapılmalıdır.

## 1. KEEP — Kesinlikle Korunacak Kullanıcı Varlıkları (ASLA SİLİNMEZ)
| Dosya / Klasör | Drive ID | Açıklama |
| :--- | :--- | :--- |
| `Unutulmus_Zamanlar_Arsivi_Final_v2.txt` | `1XTxb6BHUh8QZIa_k_k_VpDfrt2poxXtP` | 125 paragraflık ana senaryo (39.4 KB) |
| `referans_ses_15sn.wav` | `1YpTXKlpJ7-xE_4vpsupwKJ7gsF-KkAMy` | Onaylı 15 sn anlatıcı referans sesi (804 KB) |
| `telaffuz_sozlugu.json` | `1YgSDPAAj2Z_fdglYJJMJ9UwOFd7wJFoF` | Türkçe özel isim & şapkalı kelime sözlüğü |
| `bolum_yedekleri/uretim_manifesti_v2.json` | `13up-iJio6GC0gJJ0Nybxf5zdpRyoL4GT` | Canlı V2 manifesti (7 APPROVED paragraf) |
| `bolum_yedekleri/paragraflar_v2/` | `1RsUmYLnhGgQr8ltXEQVB58HD4_2Rsx3D` | 7 onaylı WAV dosyası (`001,003,007,009,017,018,021`) |
| `sfx_ambiyans_audioldm/` | `1RGZwbTnlxXDP_ZQlpskqusfguPULMNda` | Üretilmiş AudioLDM ambiyans/SFX klasörü |
| `gorseller/` | `1QkrAQG3_JmVqwC9qGdC7jY3u5ai7KF6E` | Sahne görselleri klasörü |
| `video_klipler_svd/` | `1QCk-EO6Xiw16klEwkrMggbBEq3xb6LfN` | SVD-XT video klipleri klasörü |
| `video_parca_yedekleri_hareketli/` | `1JG3YrJpTNSq_LErEBlCv3w-PuqoNhJeA` | Uzatılmış hareketli sahne parçaları |
| `GENEL_UYKU_VIDEOSU_BORU_HATTI_REHBERI.md` | `1dcfDjM6BbZxK4WjaWo1RoWt9mfJXmFDST6xJQb4-uzs` | Ana kanal üretim rehberi |
| `UYKU_HIKAYESI_BORU_HATTI_ANAYASASI.md` | `1lSuFfpvVzBdXs55oq18xdtMoqPZ5L67kcJeOa23vr1Q` | Boru hattı anayasası |

## 2. REUSE — V3 İçine Aktarılan Doğrulanmış Modüller
- `core/` (`state_manager.py`, `manifest.py`, `file_manager.py`, `recovery.py`, `logging_system.py`)
- `audio/` (`text_normalizer.py`, `pronunciation_dictionary.py`, `whisper_validator.py`, `audio_quality.py`, `mastering.py`, `sentence_cache.py`, `xtts_engine.py` — `profile_success_stats` hatası düzeltilerek)
- `visuals/` (`image_generator.py`, `image_validator.py`, `scene_manager.py`)
- `video/` (`svd_engine.py`, `loop_engine.py`, `clip_validator.py`, `motion_validator.py` — siyah/boş/tekrarlı kare denetimi eklenerek, `video_validator.py`)
- `subtitles/` (`transcription.py`, `alignment.py`, `srt_generator.py`, `srt_validator.py`)
- `assembly/` (`ffmpeg_engine.py`, `sync_validator.py`, `timeline.py`, `final_export.py`)

## 3. REBUILD — V3'te Yeniden Yazılan veya Yeni Eklenen Modüller
- `v3_runner.py` (tek birleşik 20 aşamalı orkestratör)
- `uyku_v3/core/v3_cache.py` (6 faktörlü SHA-256 önbellek motoru)
- `uyku_v3/core/protection.py` (tekil dosya SHA-256 koruma raporu)
- `uyku_v3/core/resource_monitor.py` (T4 VRAM / RAM / SSD izleyici)
- `uyku_v3/ambience/audioldm_engine.py` (10 profilli AudioLDM + lisanslı ambiyans + zaman çizelgesi karıştırıcısı)
- `uyku_v3/visuals/continuity_manager.py` (karakter/mekân/ışık/nesne görsel devamlılık yöneticisi)
- `V3_Colab_T4_Uretim.ipynb`, `colab/cell_1_kurulum_ve_hazirlik.py`, `colab/cell_2_uretim_baslat_veya_devam.py`
- `requirements_v3_colab_py313.txt` (Python 3.13 uyumlu bağımlılıklar)

## 4. ARCHIVE — V3 Canlı T4 Doğrulaması Tamamlanana Kadar Geri Dönüş Arşivinde Saklanacaklar
| Dosya / Klasör | Drive ID | Sürüm |
| :--- | :--- | :--- |
| `unutulmus_zamanlar_ses_ve_master.py` | `1u8cIfVUje0dMwOTsL2a71RNw4hEknXKV` | V1 |
| `unutulmus_zamanlar_gorsel_video_srt.py` | `1aNX5UwTkCKtNTPbiM-fj563U4lrNcyZg` | V1 |
| `UYKU_BORU_HATTI_V2_PAKET.zip` | `1GLOA_bB7n_rQwkbxPXK_8te4i_d6XaDC` | V2.1 |
| `unutulmus_zamanlar_v2_runner.py` | `1XBsPHojhdCYzuR8j5jsCwkkkhMj_CFoJ` | V2.1 |
| `UYKU_BORU_HATTI_V2_2_PAKET.zip` | `1uF0OcKeus1dri_-gx9JUsI1fYlQH1eJJ` | V2.2 |
| `unutulmus_zamanlar_v2_2_test_runner.py` | `1DDG5jknkbGQ8HNWL8MF2ygCM0Irc9X2n` | V2.2 |
| `bolum_yedekleri/v2_2_test_10p/` | `190e1Ot832XMnSiJ3dZPx9EpNp3-D1h2N` | V2.2 test geçmişi |
| `bolum_yedekleri/guvenlik_yedekleri/` | `1hyaC289dxY36Llhol-QSr56UbZajshTF` | Güvenlik yedekleri |
| `YEDEK_UYKU_HIKAYESI_BORU_HATTI_ANAYASASI.md` | `1hGZbPoPwFEt0LLDVpQb6L2bdtFK6Cp0lJmF3iYcqoAo` | Yedek anayasa |

## 5. QUARANTINE — Kalıcı Silme İçin Açık Onayınızı Bekleyen Gereksiz / Eski Dosyalar
Onay verdiğinizde aşağıdaki 17 öğe silinebilir veya ayrı çöp/karantina arşivine taşınabilir:
1. `_eski_unutulmus_zamanlar_v2_runner.py` (`1fMhUEK2uj4NDbNQR5VxgNXHeDqB-ieH_`, 47.8 KB)
2. `_eski2_unutulmus_zamanlar_v2_runner.py` (`1_AafMIhuRY7A-5SUtlCZM5A7JXhKIDUJ`, 48.2 KB)
3. `_eski3_unutulmus_zamanlar_v2_runner.py` (`1vQcRfnGSLZvOaHCpEFIlFL1zhGNoh1u2`, 49.4 KB)
4. `_eski4_unutulmus_zamanlar_v2_runner.py` (`1m8uFl5iy3_qljKLCTIIN2jUDdjR1Rx6J`, 49.4 KB)
5. `_yedek_v2_0_unutulmus_zamanlar_v2_runner.py` (`1qSQqtNaiUyZwwYyTQKOuHG1c06RF_fIA`, 50.1 KB)
6. `_eski_v2_1_pre_patch_unutulmus_zamanlar_v2_runner.py` (`1_kpjGdrJgw0tWejhNZiLqyToOZwjbU22`, 54.9 KB)
7. `_yedek_v2_1_pre_patch_unutulmus_zamanlar_v2_runner.py` (`1uQMrdtuRfONordn6RqpRJpfEnworkrB2`, 54.9 KB)
8. `_eski_v2_2_pre_fix_unutulmus_zamanlar_v2_2_test_runner.py` (`1cGcWBtwoDTr60RBXZZfWfInmB38t6fi6`, 19.6 KB)
9. `_yedek_v2_2_pre_fix_unutulmus_zamanlar_v2_2_test_runner.py` (`11jSl0I9VaFil670nV72ooVIJs6dcw6-G`, 19.6 KB)
10. `_eski_v2_2_pre_fix_UYKU_BORU_HATTI_V2_2_PAKET.zip` (`1e_hTloGxeSTT2kICOPV-mrmy4WK6bSD1`, 103.2 KB)
11. `_yedek_v2_2_pre_fix_UYKU_BORU_HATTI_V2_2_PAKET.zip` (`14pt3dBLRo38P1fuvrTaO3E-oP3dPr3If`, 103.2 KB)
12. `Unutulmus_Zamanlar_Arsivi_Studyo/__pycache__/` (`1Lj6dqUl-JjcbyO6iRnAmZYyLeJ1y0uEi`)
13. `bolum_yedekleri/paragraflar_v10_master/` (`1lUQWUUJpI8AyGmhpkaxRQXCWdfChJM0J`)
14. `bolum_yedekleri/denetim_v10_master.json` (`13v0H1208shVj_xSZfVitWKee8WFmWztr`, 976 B)
15. `bolum_yedekleri/uretim_manifesti_v2_yedek_pre_v2_1.json` (`1tfVMBka_lHtWRmdDf_13odXpXcg2xiUL`, 140.6 KB)
16. `bolum_yedekleri/cumle_duzeltme_v2_1_20261119T182833_002239Z/` (`1Q7cKIRR1LWjoJgkSD5e-LXI-fmH57T-8`)
17. `bolum_yedekleri/cumle_duzeltme_v2_1_son_20261119T183730_315596Z/` (`1N-F06lG9JKy7wCygizDuu1l6hxJ7VPd0`)
18. `karantina_hatali_dosyalar/` (`1sZBX89_gK7saok7G2hc-YJwULBLQ5Asi`) — yalnızca bozuk ara dosyalar.
