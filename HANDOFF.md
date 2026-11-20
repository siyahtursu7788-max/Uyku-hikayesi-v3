# HANDOFF — UYKU_HIKAYESI V3_PATCHED (Ortak Çalışma Durum Dosyası)

Son güncelleme: 2026-11-20  
Aktif dal / paket: `V3_PATCHED`  
Hedef ortam: Google Colab Pro (NVIDIA Tesla T4, 15 GB VRAM, Python 3.13) + Google Drive  
Hedef çıktı: *Unutulmuş Zamanlar Arşivi* (125 paragraf) için 1080p / 24 fps Türkçe uyku hikâyesi MP4 üretim hattı. İlk canlı doğrulama hedefi: 3 paragraflık çalışan ilk uçtan uca MP4 (`YOUTUBE_FINAL_1080P.mp4`).

---

## 1. Değişmez Koruma Kuralları (Tüm Geliştiriciler / AI Asistanlar İçin)

1. Yeni proje, yeni klasör veya V4 oluşturma — tüm değişiklikler mevcut `V3_PATCHED` yapısında yapılır.
2. Google Drive'daki eski V1, V2, V2.1, V2.2, `UYKU_HIKAYESI_V3_FINAL_PRODUCTION` ve onaylanmış `uretim_ciktilari/` (WAV, önbellek, manifest) dosyalarını asla silme.
3. Bir dosyayı değiştirmeden önce yedek al.
4. Kalite eşiklerini sırf akışı ilerletmek için sessizce düşürme; eşik değişikliği gerekirse kullanıcı onayı al.
5. Yerel/simüle testleri canlı T4 GPU testi gibi raporlama (`REAL_LOCAL`, `SIMULATED`, `REAL_T4_GPU` ayrımını koru).
6. T4 VRAM yönetimini bozma: aşamalar arasında GPU belleğini serbest bırak (`unload_models`), aynı modelin ikinci kopyasını yükleme.
7. Public repoya asla token, API anahtarı, kişisel dosya veya büyük `.wav` / `.mp4` çıktıları ekleme (`.gitignore` korunmalı).

---

## 2. Canlı Tesla T4 Üretim Durumu (3 Paragraflık İlk Test)

`V3_PATCHED/uretim_ciktilari/v3_production_manifest.json` kaydına göre:

| Aşama / Artefakt | Durum | Doğrulanan Gerçek Çıktı |
|---|---|---|
| `paragraf_000` (XTTS-v2 + Whisper Turbo) | **APPROVED** | Onaylı WAV (`4132d17c...`) |
| `paragraf_001` (XTTS-v2 + Whisper Turbo) | **APPROVED** | 4/4 cümle `SentenceAudioCache` hit (`80b826cb...`) |
| `paragraf_002` (XTTS-v2 + Whisper Turbo) | **APPROVED** | Onaylı WAV (`d4baf02f...`) |
| `ambience_timeline` (AudioLDM `cvssp/audioldm-s-full-v2`) | **APPROVED** | `51.884s`, `-35.48 dB`, `narrator_safe: true` |
| `master_audio` (Mastering + Outro Fade) | **APPROVED** | `216.384s`, `-22.04 dB` RMS, `-2.55 dBTP`, `outro_fade_verified: true` |
| `visual_sahne_01` (SDXL 1.0 FP16) | **APPROVED** | `1920x1080`, `393,018 B`, `mean_brightness: 89.39`, `sharpness_laplacian: 32.82` |
| `video_clip_sahne_01` (SVD-XT + OpenCV + CLIP) | **MANUAL_REVIEW** | 3 deneme yapıldı; OpenCV `motion_passed: true` (3/3), CLIP negatif `'bright sunny daylight scene'` takıldı (`0.244`, `0.244`, `0.243 > 0.24`). Çıkış kodu `2` (fail-closed gate doğru çalıştı). |
| SRT + FFmpeg 1080p MP4 | **BEKLİYOR** | `video_clip_sahne_01` onaylanınca otomatik çalışacak. |

---

## 3. Açık Olan Tek Engel & Karar Seçenekleri

- **Sorun:** `video_clip_sahne_01` klibi OpenCV hareket ve görsel kontrollerden geçti, ancak CLIP negatif kontrolünde `'bright sunny daylight scene'` benzerliği `0.243` geldi (eşik: `VideoThresholds.max_clip_cosine_negative = 0.24`, fark yalnızca `+0.003`). Mevcut `visual_sahne_01` görselinin ortalama parlaklığı `89.39`.
- **Ek küçük not:** Manifestte `visual_sahne_01` için `output_path` eski çalıştırmadan `.png` kalmış olabilir, diskte `.jpg` var (`output_hash` boş kalmış). Görsel yenilenirse `register_or_validate_cache` içinde `output_path` `.jpg` olarak güncellenmeli.
- **Seçenek 1 (Önerilen):** CLIP eşiğini `0.24` olarak koru, `v3_runner.py` içindeki `_build_scene_specs()` prompt'unu daha karanlık/gece atmosferli hâle getir, `visual_sahne_01` görselini yeniden üret ve `SVD-XT -> SRT -> FFmpeg 1080p MP4` zincirini tamamla.
- **Seçenek 2:** Mevcut `visual_sahne_01` görselini koruyarak `uyku_v3/config/quality_thresholds.py` içinde `VideoThresholds.max_clip_cosine_negative` değerini `0.24 -> 0.26` olarak kalibre et (`min_clip_margin = 0.03` ve OpenCV kontrolleri aynen kalsın) ve `SVD-XT -> SRT -> FFmpeg 1080p MP4` zincirini tamamla.

---

## 4. Canlı T4'te Çözülüp Kilitlenmiş Kritik Hatalar (Geri Bozma!)

1. `SentenceAudioCache.get_verified()` tuple döndürür (`hit[0]`), sözlük (`hit["audio"]`) değil (`v3_runner.py`).
2. `SDXL` lazy yükleme bağlantısı (`img_gen.load_sdxl(device="cuda")`) + çift yükleme koruması (`v3_runner.py`, `uyku_v3/visuals/image_generator.py`).
3. `MANUAL_REVIEW` cümlelerinin karantina WAV + Whisper metrik kayıtları (`v3_runner.py`).
4. `transformers` güncel sürümlerinde CLIP `get_image_features` / `get_text_features` dönüşünün `BaseModelOutputWithPooling` olması (`uyku_v3/video/clip_validator.py` -> `_as_tensor()`).
5. `ClipGenerationResult` alanında `.issues` değil `.error_summary` ve `.history` kullanılması (`v3_runner.py`).
6. Kapı (gate) bloklandığında çıkış kodunun `0` değil `2`, beklenmeyen hatada `1` + `loglar/last_traceback.txt` üretmesi (`v3_runner.py`).

---

## 5. Çoklu AI Devir (Handoff) Protokolü

Her anlamlı değişiklik sonrası bu dosyada şu 4 satırı güncelle:
- **Değiştiren & Tarih:**
- **Değiştirilen dosyalar:**
- **Çalıştırılan testler (`REAL_LOCAL` / `REAL_T4_GPU`):**
- **Sıradaki adım:**
